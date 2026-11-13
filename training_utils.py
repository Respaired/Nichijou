import argparse
import json
import logging
import math
import os
import shutil
from collections import OrderedDict
from glob import glob
from typing import Optional

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataset import build_multi_tts_dataloader, build_tts_dataloader

logger = logging.getLogger("nichijou.training")


def module_filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
    if not isinstance(mod, nn.Linear):
        return False

    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False

    if mod.in_features < 256 or mod.out_features < 256:
        return False

    skip_exact = {
        "in_proj",
        "out_proj",
        "cond_proj",
        "span_mask_proj",
    }

    if fqn in skip_exact:
        return False

    if fqn.startswith("cond_module."):
        return False

    if fqn.startswith("speaker_film_adapter."):
        return False

    if fqn.startswith("speaker_reg_head."):
        return False

    if fqn.startswith("reference_proj."):
        return False

    if "_adaln." in fqn:
        return False

    if fqn.startswith(("patch_merge", "patch_expand", "local_")):
        return False

    return True


def get_num_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())

    if n >= 1e9:
        return f"{n / 1e9:.2f}B"

    return f"{n / 1e6:.1f}M"


def format_eta(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m" if h > 0 else f"{m}m"


def build_lr_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    schedule_name = str(train_cfg.get("lr_schedule", "cosine")).lower()

    if schedule_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if schedule_name == "cosine":
        warmup_steps = int(train_cfg["warmup_steps"])
        max_steps = int(train_cfg["max_steps"])
        min_lr = float(train_cfg.get("min_lr", 0.0))

        if not 0 <= warmup_steps < max_steps:
            raise ValueError("cosine requires 0 <= warmup_steps < max_steps")

        if min_lr < 0 or any(min_lr > group["lr"] for group in optimizer.param_groups):
            raise ValueError(
                "min_lr must be nonnegative and no greater than each initial learning rate"
            )

        def schedule(base_lr):
            floor = min_lr / base_lr if base_lr else 0.0

            def factor(step):
                if warmup_steps and step < warmup_steps:
                    return float(step) / warmup_steps

                progress = min(1.0, max(0.0, (step - warmup_steps) / (max_steps - warmup_steps)))
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

            return factor

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[schedule(group["lr"]) for group in optimizer.param_groups]
        )

    if schedule_name != "wsd":
        raise ValueError(
            f"Unsupported training.lr_schedule={schedule_name!r}; expected 'constant', 'wsd' or 'cosine'"
        )

    warmup_steps = int(train_cfg["warmup_steps"])
    decay_start = int(train_cfg["wsd_decay_start"])
    decay_steps = int(train_cfg["wsd_decay_steps"])
    min_lr_ratio = float(train_cfg["wsd_min_lr_ratio"])
    max_steps = int(train_cfg["max_steps"])
    decay_end = decay_start + decay_steps

    if decay_start < warmup_steps:
        raise ValueError("wsd_decay_start must be >= warmup_steps")

    if decay_steps <= 0:
        raise ValueError("wsd_decay_steps must be > 0")

    if decay_end > max_steps:
        raise ValueError("wsd_decay_start + wsd_decay_steps must be <= max_steps")

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("wsd_min_lr_ratio must be in [0, 1]")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)

        if step < decay_start:
            return 1.0

        if step < decay_end:
            progress = float(step - decay_start) / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def reset_constant_lr(optimizer, scheduler, learning_rate):
    if learning_rate <= 0:
        raise ValueError("constant learning rate must be positive")

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = float(learning_rate)

    fresh = build_lr_scheduler(optimizer, {"lr_schedule": "constant"})
    scheduler.load_state_dict(fresh.state_dict())


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module

    while hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def save_nonfinite_diagnostic(exp_dir, model, batch, step, phase):
    bad_parameters, bad_gradients = [], []

    for name, parameter in unwrap_model(model).named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            bad_parameters.append(name)

        if parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all():
            bad_gradients.append(name)

    inputs = {
        key: value.detach().cpu() for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    finite = {key: bool(torch.isfinite(value).all()) for key, value in inputs.items()}
    path = os.path.join(exp_dir, "nonfinite_batch.pt")
    torch.save(
        dict(
            step=step,
            phase=phase,
            batch=inputs,
            finite_inputs=finite,
            bad_parameters=bad_parameters,
            bad_gradients=bad_gradients,
            text_encoder_state=(
                {
                    name: value.detach().cpu()
                    for name, value in unwrap_model(model).text_encoder.state_dict().items()
                }
                if hasattr(unwrap_model(model), "text_encoder")
                else {}
            ),
        ),
        path,
    )
    logger.error(
        "Non-finite %s at step %d | finite inputs=%s | bad parameters=%s | "
        "bad gradients=%s | failing batch saved: %s",
        phase,
        step,
        finite,
        bad_parameters[:8],
        bad_gradients[:8],
        path,
    )


def check_backward_finite(model, batch, exp_dir, step):
    try:
        parameters = list(model.parameters())
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error).lower():
            raise

        save_nonfinite_diagnostic(exp_dir, model, batch, step, "gradient norm")
        raise FloatingPointError(
            f"Non-finite gradient norm at optimizer step {step}; no update or checkpoint saved"
        ) from error


def checked_grad_norm(accelerator, model, batch, exp_dir, step, limit):
    if accelerator.distributed_type.name in {"NO", "MULTI_GPU", "MULTI_CPU"}:
        accelerator.unscale_gradients()

        try:
            return float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), limit, error_if_nonfinite=True, foreach=True
                )
            )
        except RuntimeError as error:
            if "non-finite" not in str(error).lower():
                raise

            save_nonfinite_diagnostic(exp_dir, model, batch, step, "gradient norm")
            raise FloatingPointError(
                f"Non-finite gradient norm at optimizer step {step}; no update or checkpoint saved"
            ) from error

    check_backward_finite(model, batch, exp_dir, step)
    return float(accelerator.clip_grad_norm_(model.parameters(), limit))


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def _muon_like_param(name: str, p: torch.Tensor) -> bool:
    if not p.requires_grad or p.ndim != 2:
        return False

    lname = name.lower()
    excluded = (
        "in_proj.weight",
        "out_proj.weight",
        "cond_proj.weight",
        "span_mask_proj.weight",
        "reference_proj.weight",
        "speaker_reg_head",
        "embedding",
        "embed_tokens",
        "tok_embeddings",
    )
    return not any(part in lname for part in excluded)


def _tensor_group_stats(named_tensors):
    items = [(name, t) for name, t in named_tensors if t is not None]

    if not items:
        return {"rms": 0.0, "max": 0.0, "l2": 0.0, "peak_l2": 0.0, "peak_name": "none"}

    sumsq, numel, maxima = [], 0, []

    for _, t in items:
        x = t.detach().float()
        sumsq.append(x.square().sum())
        maxima.append(x.abs().max())
        numel += x.numel()

    sums = torch.stack(sumsq)
    ss = sums.sum()
    mx = torch.stack(maxima).max()
    peak_idx = int(sums.argmax().item())
    return {
        "rms": (ss / max(numel, 1)).sqrt().item(),
        "max": mx.item(),
        "l2": ss.sqrt().item(),
        "peak_l2": sums[peak_idx].sqrt().item(),
        "peak_name": items[peak_idx][0],
    }


def collect_muon_norm_stats(model: nn.Module, grads: bool = False) -> dict:
    raw = unwrap_model(model)
    muon, q, k = [], [], []

    for name, p in raw.named_parameters():
        if not _muon_like_param(name, p):
            continue

        t = p.grad if grads else p

        if t is None:
            continue

        muon.append((name, t))
        lname = name.lower()

        if any(tok in lname for tok in (".wq.", ".q_proj.", ".query.")) or lname.endswith(
            "wq.weight"
        ):
            q.append((name, t))

        if any(tok in lname for tok in (".wk.", ".k_proj.", ".key.")) or lname.endswith(
            "wk.weight"
        ):
            k.append((name, t))

    out = {}

    for prefix, group in (("muon", muon), ("q", q), ("k", k)):
        stats = _tensor_group_stats(group)
        out.update(
            {
                f"{prefix}_rms": stats["rms"],
                f"{prefix}_max": stats["max"],
                f"{prefix}_l2": stats["l2"],
                f"{prefix}_peak_l2": stats["peak_l2"],
                f"{prefix}_peak_name": stats["peak_name"],
            }
        )

    return out


def build_muon_optimizers(model: nn.Module, train_cfg: dict):
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "optimizer='muon' requires a PyTorch build with torch.optim.Muon. "
            f"Installed torch version: {torch.__version__}"
        )

    raw = unwrap_model(model)
    embedding_ids = {
        id(p)
        for module in raw.modules()
        if isinstance(module, nn.Embedding)
        for p in module.parameters(recurse=False)
    }
    excluded_parts = (
        "in_proj.weight",
        "out_proj.weight",
        "cond_proj.weight",
        "span_mask_proj.weight",
        "reference_proj.weight",
        "speaker_reg_head",
        "embedding",
        "embed_tokens",
        "tok_embeddings",
    )
    muon_params, muon_ids = [], set()

    for name, p in raw.named_parameters():
        if not p.requires_grad or p.ndim != 2 or id(p) in embedding_ids:
            continue

        lname = name.lower()

        if any(part in lname for part in excluded_parts):
            continue

        muon_params.append(p)
        muon_ids.add(id(p))

    adamw_params = [p for p in raw.parameters() if p.requires_grad and id(p) not in muon_ids]

    if not muon_params:
        raise RuntimeError("Muon selected but no eligible hidden 2D weights were found")

    if not adamw_params:
        raise RuntimeError("Muon selected but no auxiliary AdamW parameters were found")

    muon_lr = float(train_cfg.get("muon_lr", 0.02))
    adamw_lr = float(train_cfg.get("muon_adamw_lr", train_cfg["lr"]))
    wd = float(train_cfg["weight_decay"])
    betas = tuple(train_cfg["betas"])
    momentum = float(train_cfg.get("muon_momentum", 0.95))
    ns_steps = int(train_cfg.get("muon_ns_steps", 5))
    adjust_lr_fn = train_cfg.get("muon_adjust_lr_fn", "original")
    muon_optimizer = torch.optim.Muon(
        muon_params,
        lr=muon_lr,
        weight_decay=wd,
        momentum=momentum,
        nesterov=True,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=betas,
        weight_decay=wd,
    )
    logger.info(
        "Optimizer: torch.optim.Muon + torch.optim.AdamW | muon_tensors=%d params=%s lr=%g | "
        "aux_adamw_tensors=%d params=%s lr=%g | wd=%g | momentum=%g | ns_steps=%d | adjust_lr_fn=%s",
        len(muon_params),
        f"{sum(p.numel() for p in muon_params):,}",
        muon_lr,
        len(adamw_params),
        f"{sum(p.numel() for p in adamw_params):,}",
        adamw_lr,
        wd,
        momentum,
        ns_steps,
        adjust_lr_fn,
    )
    return muon_optimizer, adamw_optimizer


def load_silence_latent(path: str, latent_dim: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(obj, dict):
        for key in ("latents", "latent", "codes", "fsq"):
            if key in obj:
                obj = obj[key]
                break
        else:
            tensor_values = [v for v in obj.values() if isinstance(v, torch.Tensor)]

            if len(tensor_values) != 1:
                raise ValueError(
                    f"Could not identify silence latent tensor in {path}; keys={list(obj.keys())}"
                )

            obj = tensor_values[0]

    if not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)

    while obj.dim() > 2 and obj.shape[0] == 1:
        obj = obj.squeeze(0)

    if obj.dim() != 2:
        raise ValueError(
            f"Silence latent must be rank-2 [T, D] (or singleton-batched), got {tuple(obj.shape)}"
        )

    if obj.shape[-1] != latent_dim and obj.shape[0] == latent_dim:
        obj = obj.transpose(0, 1)

    if obj.shape[-1] != latent_dim:
        raise ValueError(
            f"Silence latent dim mismatch: expected {latent_dim}, got {tuple(obj.shape)}"
        )

    if obj.shape[0] < 1:
        raise ValueError("Silence latent sequence is empty")

    return obj.contiguous().float()


def fill_bucket_tail_with_silence(
    latents: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_audio_mask: torch.Tensor,
    silence_latent: torch.Tensor,
) -> torch.Tensor:
    pad_mask = valid_audio_mask & ~speech_mask
    silence = silence_latent.to(device=latents.device, dtype=latents.dtype)
    pad_offsets = pad_mask.long().cumsum(dim=1) - 1
    silence_indices = pad_offsets[pad_mask].remainder(silence.shape[0])
    out = latents.clone()
    out[pad_mask] = silence[silence_indices]
    return out


def sample_stratified_logit_normal(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    offset = torch.rand((), device=device)
    shift = torch.randint(0, batch_size, (), device=device)
    idx = (torch.arange(batch_size, device=device) + shift) % batch_size
    q = (idx.float() + offset) / batch_size
    z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    return torch.sigmoid(z).to(dtype)


def sample_training_mask(
    speech_mask: torch.Tensor,
    valid_audio_mask: torch.Tensor,
    *,
    p_full: float = 0.70,
    p_suffix: float = 0.15,
    p_infill: float = 0.15,
    partial_frac_min: float = 0.70,
    partial_frac_max: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    speech_mask = speech_mask.to(torch.bool)
    valid_audio_mask = valid_audio_mask.to(torch.bool)
    B, _ = speech_mask.shape
    device = speech_mask.device
    probs = torch.tensor([p_full, p_suffix, p_infill], device=device, dtype=torch.float32)

    if torch.any(probs < 0) or probs.sum() <= 0:
        raise ValueError(
            f"Invalid objective probabilities: full={p_full}, suffix={p_suffix}, infill={p_infill}"
        )

    probs = probs / probs.sum()
    mode_ids = torch.multinomial(probs, num_samples=B, replacement=True)
    span_mask = valid_audio_mask & ~speech_mask
    valid_lens = speech_mask.sum(dim=1).long()

    for i in range(B):
        L = int(valid_lens[i].item())

        if L <= 0:
            continue

        mode = int(mode_ids[i].item())

        if mode == 0:
            span_mask[i] |= speech_mask[i]
            continue

        frac = float(
            torch.empty((), device=device).uniform_(partial_frac_min, partial_frac_max).item()
        )

        if mode == 1:
            if L == 1:
                span_mask[i, :L] = True
                mode_ids[i] = 0
                continue

            mask_len = min(max(int(frac * L), 1), L - 1)
            start = L - mask_len
            span_mask[i, start:L] = True
            continue

        if L < 3:
            mask_len = min(max(int(frac * L), 1), max(L - 1, 1))
            start = max(L - mask_len, 0)
            span_mask[i, start:L] = True
            mode_ids[i] = 1 if L > 1 else 0
            continue

        mask_len = min(max(int(frac * L), 1), L - 2)
        max_start = L - mask_len - 1
        start = int(torch.randint(1, max_start + 1, (1,), device=device).item())
        span_mask[i, start : start + mask_len] = True

    span_mask &= valid_audio_mask
    return span_mask, mode_ids


def extract_disc_features(
    model: nn.Module,
    latents: torch.Tensor,
    latent_lengths: torch.Tensor,
    text_ids: torch.Tensor,
    text_mask: torch.Tensor,
    detach_generator: bool,
    speaker_emb: Optional[torch.Tensor] = None,
    disc_t: Optional[torch.Tensor] = None,
    disc_noise: Optional[torch.Tensor] = None,
):
    max_len = int(latent_lengths.max().item())
    latents = latents[:, :max_len, :]
    valid_mask = torch.arange(max_len, device=latents.device)[None, :] < latent_lengths[:, None]
    B = latents.size(0)

    if disc_t is not None and disc_noise is not None:
        disc_noise = disc_noise[:, :max_len, :]
        t_exp = disc_t[:, None, None]
        noisy_latents = (1.0 - t_exp) * disc_noise + t_exp * latents
        noisy_latents = noisy_latents * valid_mask.unsqueeze(-1).to(noisy_latents.dtype)
        t_for_model = disc_t
    else:
        noisy_latents = latents
        t_for_model = torch.ones(B, device=latents.device, dtype=latents.dtype)

    kwargs = dict(
        x=noisy_latents,
        cond=noisy_latents,
        t=t_for_model,
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=None,
        reference_mask=None,
        text_cond_drop=torch.zeros(B, device=latents.device, dtype=torch.bool),
        speaker_cond_drop=None,
        use_checkpoint=False,
        latent_mask=valid_mask,
        valid_audio_mask=valid_mask,
        span_mask=torch.zeros_like(valid_mask),
        return_hidden_states=True,
        speaker_emb=speaker_emb,
    )

    if detach_generator:
        with torch.no_grad():
            _, gen_layers = model(**kwargs)

        gen_layers = [x.detach() for x in gen_layers]
    else:
        _, gen_layers = model(**kwargs)

    return gen_layers


def compute_discriminator_loss(
    model: nn.Module,
    discriminator: nn.Module,
    cache: dict,
) -> torch.Tensor:
    B = cache["fake_latents"].size(0)
    device = cache["fake_latents"].device
    dtype = cache["fake_latents"].dtype
    disc_t = torch.rand(B, device=device, dtype=dtype)
    disc_noise = torch.randn_like(cache["fake_latents"])
    fake_feats = extract_disc_features(
        model=model,
        latents=cache["fake_latents"],
        latent_lengths=cache["latent_lengths"],
        text_ids=cache["text_ids"],
        text_mask=cache["text_mask"],
        detach_generator=True,
        speaker_emb=cache.get("speaker_embs"),
        disc_t=disc_t,
        disc_noise=disc_noise,
    )
    real_feats = extract_disc_features(
        model=model,
        latents=cache["real_latents"],
        latent_lengths=cache["latent_lengths"],
        text_ids=cache["text_ids"],
        text_mask=cache["text_mask"],
        detach_generator=True,
        speaker_emb=cache.get("speaker_embs"),
        disc_t=disc_t,
        disc_noise=disc_noise,
    )
    fake_score = discriminator(fake_feats, cache["latent_lengths"])
    real_score = discriminator(real_feats, cache["latent_lengths"])
    return (fake_score**2).mean() + ((1.0 - real_score) ** 2).mean()


def compute_loss(
    model: nn.Module,
    batch: dict,
    use_checkpoint: bool,
    *,
    stage2: bool = False,
    use_speaker: bool = False,
    use_adversarial: bool = False,
    discriminator: Optional[nn.Module] = None,
    adv_lambda: float = 0.0,
    speaker_aux_weight: float = 0.1,
    silence_latent: Optional[torch.Tensor] = None,
    objective_mix: Optional[dict] = None,
):
    latents = batch["latents"]
    valid_audio_mask = batch["latent_mask"].to(torch.bool)
    speech_mask = batch["speech_mask"].to(torch.bool)
    text_ids = batch["text_ids"]
    text_mask = batch["text_mask"].to(torch.bool)
    B, T, D = latents.shape
    device = latents.device
    dtype = latents.dtype

    if silence_latent is not None:
        latents = fill_bucket_tail_with_silence(
            latents=latents,
            speech_mask=speech_mask,
            valid_audio_mask=valid_audio_mask,
            silence_latent=silence_latent,
        )

    x1 = latents * valid_audio_mask.unsqueeze(-1).to(latents.dtype)
    objective_cfg = objective_mix or {}
    span_mask, objective_mode = sample_training_mask(
        speech_mask=speech_mask,
        valid_audio_mask=valid_audio_mask,
        p_full=float(objective_cfg.get("p_full", 0.70)),
        p_suffix=float(objective_cfg.get("p_suffix", 0.15)),
        p_infill=float(objective_cfg.get("p_infill", 0.15)),
        partial_frac_min=float(objective_cfg.get("partial_frac_min", 0.70)),
        partial_frac_max=float(objective_cfg.get("partial_frac_max", 0.95)),
    )
    x0 = torch.randn_like(x1)

    t = sample_stratified_logit_normal(B, device=device, dtype=dtype)
    t_exp = t[:, None, None]
    x_t = (1.0 - t_exp) * x0 + t_exp * x1
    v_target = x1 - x0
    cond = torch.where(span_mask.unsqueeze(-1), torch.zeros_like(x1), x1)
    cond = cond * valid_audio_mask.unsqueeze(-1).to(cond.dtype)
    model_input = x_t * valid_audio_mask.unsqueeze(-1).to(x_t.dtype)
    p_drop_both = 0.1
    p_drop_text = 0.1
    p_drop_speaker = 0.0
    r = torch.rand(B, device=device)
    b0 = p_drop_both
    b1 = b0 + p_drop_text
    b2 = b1 + p_drop_speaker
    drop_both_bucket = r < b0
    drop_text_bucket = (r >= b0) & (r < b1)
    drop_speaker_bucket = (r >= b1) & (r < b2)
    text_cond_drop = drop_both_bucket | drop_text_bucket
    speaker_cond_drop = None
    speaker_emb = batch.get("speaker_embs", None)

    if speaker_emb is not None:
        speaker_emb = speaker_emb.to(device)

    if stage2 and use_speaker and speaker_emb is not None:
        speaker_cond_drop = drop_both_bucket | drop_speaker_bucket

    need_hidden = stage2 and use_speaker and speaker_emb is not None
    out = model(
        x=model_input,
        cond=cond,
        t=t,
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=None,
        reference_mask=None,
        text_cond_drop=text_cond_drop,
        speaker_cond_drop=speaker_cond_drop,
        use_checkpoint=use_checkpoint,
        latent_mask=valid_audio_mask,
        valid_audio_mask=valid_audio_mask,
        span_mask=span_mask,
        speaker_emb=speaker_emb if (stage2 and use_speaker) else None,
        return_hidden_states=need_hidden,
    )

    if need_hidden:
        v_pred, hidden_states = out
    else:
        v_pred = out

    per_pos_loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="none").mean(dim=-1)
    masked_loss = per_pos_loss[span_mask]
    fm_loss = masked_loss.mean() if masked_loss.numel() > 0 else per_pos_loss.new_zeros(())
    speech_loss_mask = span_mask & speech_mask
    silence_loss_mask = span_mask & valid_audio_mask & ~speech_mask
    speech_region_loss = (
        per_pos_loss[speech_loss_mask].mean()
        if speech_loss_mask.any()
        else per_pos_loss.new_zeros(())
    )
    silence_region_loss = (
        per_pos_loss[silence_loss_mask].mean()
        if silence_loss_mask.any()
        else per_pos_loss.new_zeros(())
    )
    visible_speech = speech_mask & ~span_mask
    zero_visible_frac = (visible_speech.sum(dim=1) == 0).float().mean()
    full_frac = (objective_mode == 0).float().mean()
    suffix_frac = (objective_mode == 1).float().mean()
    infill_frac = (objective_mode == 2).float().mean()
    gen_adv_loss_val = 0.0
    speaker_aux_loss_val = 0.0
    disc_cache = None

    if not stage2:
        stats = {
            "fm_loss": fm_loss.detach(),
            "speech_loss": speech_region_loss.detach(),
            "silence_loss": silence_region_loss.detach(),
            "full_tts_frac": full_frac.detach(),
            "suffix_frac": suffix_frac.detach(),
            "infill_frac": infill_frac.detach(),
            "zero_visible_frac": zero_visible_frac.detach(),
            "gen_adv_loss": torch.tensor(0.0, device=device),
            "speaker_aux_loss": torch.tensor(0.0, device=device),
        }
        return fm_loss, stats, None

    total_loss = fm_loss
    speaker_aux_loss = fm_loss.new_zeros(())

    if (
        use_speaker
        and speaker_emb is not None
        and need_hidden
        and hasattr(unwrap_model(model), "predict_speaker")
    ):
        raw = unwrap_model(model)
        last_hidden = hidden_states[-1]
        spk_pred = raw.predict_speaker(last_hidden, valid_audio_mask)

        if spk_pred is not None:
            valid_spk = speaker_emb.abs().sum(dim=-1) > 0

            if speaker_cond_drop is not None:
                valid_spk = valid_spk & ~speaker_cond_drop.to(torch.bool)

            if valid_spk.any():
                speaker_aux_loss = F.mse_loss(
                    spk_pred[valid_spk],
                    speaker_emb[valid_spk],
                )
                speaker_aux_loss_val = speaker_aux_loss.detach().item()

    total_loss = total_loss + speaker_aux_weight * speaker_aux_loss
    gen_adv_loss = fm_loss.new_zeros(())

    if use_adversarial and discriminator is not None and adv_lambda > 0.0:
        latent_lengths = valid_audio_mask.sum(dim=1).long()
        x0_gen = torch.randn_like(x1)
        t_gen = torch.rand(B, device=device, dtype=dtype)
        t_gen_exp = t_gen[:, None, None]
        x_t_gen = (1.0 - t_gen_exp) * x0_gen + t_gen_exp * x1
        v_gen = model(
            x=x_t_gen * valid_audio_mask.unsqueeze(-1).to(x_t_gen.dtype),
            cond=torch.zeros_like(x1),
            t=t_gen,
            text_ids=text_ids,
            text_mask=text_mask,
            reference_latent=None,
            reference_mask=None,
            text_cond_drop=torch.zeros(B, device=device, dtype=torch.bool),
            use_checkpoint=use_checkpoint,
            latent_mask=valid_audio_mask,
            valid_audio_mask=valid_audio_mask,
            span_mask=valid_audio_mask,
            speaker_emb=speaker_emb if use_speaker else None,
        )
        x1_hat = (x0_gen + v_gen) * valid_audio_mask.unsqueeze(-1).to(v_gen.dtype)
        disc_t = torch.rand(B, device=device, dtype=dtype)
        disc_noise = torch.randn_like(x1)
        fake_feats = extract_disc_features(
            model=model,
            latents=x1_hat,
            latent_lengths=latent_lengths,
            text_ids=text_ids,
            text_mask=text_mask,
            detach_generator=False,
            speaker_emb=speaker_emb if use_speaker else None,
            disc_t=disc_t,
            disc_noise=disc_noise,
        )
        fake_score = discriminator(fake_feats, latent_lengths)
        gen_adv_loss = ((1.0 - fake_score) ** 2).mean()
        gen_adv_loss_val = gen_adv_loss.detach().item()
        disc_cache = {
            "fake_latents": x1_hat.detach(),
            "real_latents": x1.detach(),
            "latent_lengths": latent_lengths.detach(),
            "text_ids": text_ids.detach(),
            "text_mask": text_mask.detach(),
            "speaker_embs": speaker_emb.detach() if speaker_emb is not None else None,
        }

    total_loss = total_loss + adv_lambda * gen_adv_loss
    stats = {
        "fm_loss": fm_loss.detach(),
        "speech_loss": speech_region_loss.detach(),
        "silence_loss": silence_region_loss.detach(),
        "full_tts_frac": full_frac.detach(),
        "suffix_frac": suffix_frac.detach(),
        "infill_frac": infill_frac.detach(),
        "zero_visible_frac": zero_visible_frac.detach(),
        "gen_adv_loss": torch.tensor(gen_adv_loss_val, device=device),
        "speaker_aux_loss": torch.tensor(speaker_aux_loss_val, device=device),
    }
    return total_loss, stats, disc_cache


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_bufs = OrderedDict(ema_model.named_buffers())
    model_bufs = OrderedDict(model.named_buffers())

    for name, buf in model_bufs.items():
        if name in ema_bufs:
            ema_bufs[name].copy_(buf)


def save_checkpoint(
    accelerator,
    model,
    ema,
    step,
    exp_dir,
    tag,
    checkpoint_bucket=None,
    keep_last_k=5,
    local_keep_last_k=None,
):
    ckpt_dir = os.path.join(exp_dir, "checkpoints", tag)

    if checkpoint_bucket and local_keep_last_k == 1:
        from accelerate.utils import broadcast_object_list

        preparation_error = [None]

        if accelerator.is_main_process:
            try:
                from bucket_checkpoints import upload_checkpoint

                for previous in sorted(glob(os.path.join(exp_dir, "checkpoints", "step_*"))):
                    if previous == ckpt_dir:
                        continue

                    upload_checkpoint(previous, checkpoint_bucket, keep_last_k)
                    shutil.rmtree(previous)
            except Exception as exc:
                preparation_error[0] = f"{type(exc).__name__}: {exc}"

        broadcast_object_list(preparation_error)

        if preparation_error[0] is not None:
            raise RuntimeError(f"Cannot prepare local checkpoint storage: {preparation_error[0]}")

        accelerator.wait_for_everyone()

    os.makedirs(ckpt_dir, exist_ok=True)
    accelerator.save_state(ckpt_dir)

    if accelerator.is_main_process:
        raw_model = unwrap_model(model)
        torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "model_state.pt"))
        extra = {
            "step": step,
            "ema": ema.state_dict(),
            "architecture": raw_model.architecture_signature(),
        }
        torch.save(extra, os.path.join(ckpt_dir, "extra_state.pt"))

    accelerator.wait_for_everyone()

    if checkpoint_bucket:
        from accelerate.utils import broadcast_object_list

        upload_error = [None]

        if accelerator.is_main_process:
            try:
                from bucket_checkpoints import upload_checkpoint

                upload_checkpoint(ckpt_dir, checkpoint_bucket, keep_last_k)
                logger.info("Uploaded checkpoint: %s/%s", checkpoint_bucket, tag)
            except Exception as exc:
                upload_error[0] = f"{type(exc).__name__}: {exc}"

        broadcast_object_list(upload_error)

        if upload_error[0] is not None:
            raise RuntimeError(
                f"Checkpoint bucket upload failed; local checkpoint retained: {upload_error[0]}"
            )

    accelerator.wait_for_everyone()
    logger.info(f"Saved checkpoint: {tag}")


def load_checkpoint(accelerator, model, ema, exp_dir, tag, drop_optimizer_state=False):
    ckpt_dir = os.path.join(exp_dir, "checkpoints", tag)

    extra_path = os.path.join(ckpt_dir, "extra_state.pt")
    extra = torch.load(extra_path, map_location="cpu", weights_only=True)
    raw_model = unwrap_model(model)
    expected = raw_model.architecture_signature()
    saved = extra.get("architecture")

    if saved is None and (raw_model.patch_size > 1 or raw_model.local_num_layers > 0):
        raise RuntimeError(
            "This checkpoint has no patch/refinement architecture metadata. "
            "It cannot be resumed into the coarse-to-fine model, including with "
            "--drop_optimizer_state. Use a new experiment directory without --resume."
        )

    if saved is not None and saved != expected:
        raise RuntimeError(
            f"Checkpoint architecture does not match this model. Saved={saved}; "
            f"requested={expected}. Use the matching config or start a new run."
        )

    if not drop_optimizer_state:
        accelerator.load_state(ckpt_dir)
    else:
        logger.info("drop_optimizer_state=True -> skipping optimizer/scheduler/RNG restore")

    model_state_path = os.path.join(ckpt_dir, "model_state.pt")

    if os.path.exists(model_state_path):
        raw_model = unwrap_model(model)
        state = torch.load(model_state_path, map_location="cpu", weights_only=True)
        raw_model.load_state_dict(state, strict=False)
        logger.info("Loaded model state_dict (strict=False)")

    try:
        ema.load_state_dict(extra["ema"])
    except RuntimeError:
        logger.warning(
            "EMA state_dict mismatch (likely stage1->stage2), "
            "re-initialising EMA from current model weights"
        )
        ema.load_state_dict(unwrap_model(model).state_dict(), strict=False)

    ema.to(accelerator.device)

    if drop_optimizer_state:
        step = 0
        logger.info(f"Loaded weights from {tag} | optimizer dropped | step reset to 0")
    else:
        step = extra["step"]
        logger.info(f"Resumed from {tag}  (step {step})")

    return step


def find_latest_checkpoint(exp_dir):
    ckpt_root = os.path.join(exp_dir, "checkpoints")

    if not os.path.exists(ckpt_root):
        return None

    dirs = sorted(glob(os.path.join(ckpt_root, "step_*")))
    return os.path.basename(dirs[-1]) if dirs else None


def cleanup_checkpoints(exp_dir, keep_last_k):
    ckpt_root = os.path.join(exp_dir, "checkpoints")

    if not os.path.exists(ckpt_root):
        return

    dirs = sorted(glob(os.path.join(ckpt_root, "step_*")))

    for d in dirs[:-keep_last_k]:
        shutil.rmtree(d, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str)
    dataset_group.add_argument(
        "--datasets", type=str, help="Path to JSON file containing a top-level datasets list"
    )
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--drop_optimizer_state", action="store_true")
    parser.add_argument(
        "--reset_lr_scheduler",
        action="store_true",
        help="Keep resumed optimizer moments and step count, but reset a constant LR schedule",
    )
    parser.add_argument(
        "--length_training_stage",
        action="store_true",
        help="Train on numbered neighboring clips, capped at 8 samples and 60 seconds (non-streaming)",
    )
    parser.add_argument(
        "--length_training_fraction",
        type=float,
        default=None,
        help="Fraction of random starting samples for length training (default 0.5, or training config)",
    )

    parser.add_argument("--stage2", action="store_true", help="Enable stage-2 fine-tuning")
    parser.add_argument(
        "--use_speaker",
        action="store_true",
        help="Enable TitaNet speaker-conditioned training (requires --stage2)",
    )
    parser.add_argument("--speaker_aux_weight", type=float, default=0.1)
    parser.add_argument("--speaker_drop_prob", type=float, default=0.1)

    parser.add_argument("--use_adversarial", action="store_true")
    parser.add_argument("--adv_lambda", type=float, default=1.0)
    parser.add_argument("--disc_lr", type=float, default=5e-5)
    parser.add_argument("--disc_channels", type=int, default=512)
    parser.add_argument("--disc_num_layers", type=int, default=3)
    parser.add_argument("--disc_num_heads", type=int, default=8)
    parser.add_argument("--disc_kernel", type=int, default=15)
    args = parser.parse_args()

    if args.length_training_stage and args.streaming:
        parser.error("--length_training_stage requires indexed datasets; remove --streaming")

    if args.length_training_fraction is not None and not args.length_training_stage:
        parser.error("--length_training_fraction requires --length_training_stage")

    return args


def load_training_tokenizer(args, model_cfg, data_cfg):
    text_pad_id = data_cfg["text_pad_id"]
    tokenizer_name = (
        args.tokenizer
        or data_cfg.get("tokenizer")
        or data_cfg.get("tokenizer_name")
        or data_cfg.get("text_tokenizer")
    )

    if not tokenizer_name:
        raise ValueError(
            "A tokenizer must be provided via --tokenizer or data.tokenizer/tokenizer_name/text_tokenizer"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        add_bos_token=True,
        add_eos_token=True,
    )
    text_encoder_cfg = model_cfg["text_encoder"]

    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} has no pad_token_id")

    if int(tokenizer.pad_token_id) != int(text_pad_id):
        raise ValueError(
            f"Tokenizer/config pad mismatch: tokenizer.pad_token_id={tokenizer.pad_token_id}, data.text_pad_id={text_pad_id}"
        )

    model_pad_id = int(text_encoder_cfg.get("pad_token_id", text_pad_id))

    if int(tokenizer.pad_token_id) != model_pad_id:
        raise ValueError(
            f"Tokenizer/model pad mismatch: tokenizer.pad_token_id={tokenizer.pad_token_id}, "
            f"model.text_encoder.pad_token_id={model_pad_id}"
        )

    for attr in ("add_bos_token", "add_eos_token"):
        if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not True:
            raise ValueError(
                f"Tokenizer {tokenizer_name} must have {attr}=True for this training run"
            )

    return tokenizer_name, tokenizer


def build_training_optimizers(args, model, train_cfg):
    optimizer_name = str(train_cfg.get("optimizer", "adamw8bit")).lower()

    if optimizer_name not in ("adamw8bit", "muon"):
        raise ValueError(
            f"training.optimizer must be 'adamw8bit' or 'muon', got {optimizer_name!r}"
        )

    if args.stage2 and optimizer_name == "muon":
        raise ValueError(
            "optimizer='muon' is currently Stage-1 only; Stage 2 keeps its explicit base/speaker AdamW8bit LR groups"
        )

    aux_optimizer = None
    aux_scheduler = None

    if args.stage2:
        raw = unwrap_model(model)
        speaker_params = []
        speaker_param_ids = set()

        if args.use_speaker:
            if raw.speaker_film_adapter is not None:
                for p in raw.speaker_film_adapter.parameters():
                    if p.requires_grad:
                        speaker_params.append(p)
                        speaker_param_ids.add(id(p))

            if raw.speaker_reg_head is not None:
                for p in raw.speaker_reg_head.parameters():
                    if p.requires_grad:
                        speaker_params.append(p)
                        speaker_param_ids.add(id(p))

        base_params = [
            p for p in model.parameters() if p.requires_grad and id(p) not in speaker_param_ids
        ]
        param_groups = []

        if base_params:
            param_groups.append(
                {"params": base_params, "lr": 2e-5, "weight_decay": train_cfg["weight_decay"]}
            )

        if speaker_params:
            param_groups.append(
                {"params": speaker_params, "lr": 1e-4, "weight_decay": train_cfg["weight_decay"]}
            )

        logger.info(
            f"Optimizer: AdamW8bit | base params={len(base_params)} @ lr=2e-5, speaker params={len(speaker_params)} @ lr=1e-4"
        )
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=5e-5,
            betas=tuple(train_cfg["betas"]),
            weight_decay=train_cfg["weight_decay"],
        )
    elif optimizer_name == "muon":
        optimizer, aux_optimizer = build_muon_optimizers(model, train_cfg)
    else:
        logger.info(f"Optimizer: AdamW8bit | lr={train_cfg['lr']} | wd={train_cfg['weight_decay']}")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=train_cfg["lr"],
            betas=tuple(train_cfg["betas"]),
            weight_decay=train_cfg["weight_decay"],
        )

    scheduler = build_lr_scheduler(optimizer, train_cfg)

    if aux_optimizer is not None:
        aux_scheduler = build_lr_scheduler(aux_optimizer, train_cfg)

    return optimizer_name, optimizer, aux_optimizer, scheduler, aux_scheduler


def build_training_dataloader(
    args,
    tokenizer,
    data_cfg,
    train_cfg,
    max_latent_len,
    latent_buckets,
    latent_pad_value,
):
    options = dict(
        latent_dim=data_cfg["latent_dim"],
        need_audio=args.stage2 and args.use_speaker,
        max_text_len=data_cfg["max_text_length"],
        max_latent_len=max_latent_len,
        text_pad_id=data_cfg["text_pad_id"],
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        streaming=args.streaming,
        dataset_split=args.dataset_split,
        buckets=latent_buckets,
        pad_value=latent_pad_value,
    )

    if args.datasets is None:
        logger.info(f"Loading dataset from {args.dataset}")
        return build_tts_dataloader(dataset_path=args.dataset, **options)

    with open(args.datasets, "r", encoding="utf-8") as file:
        dataset_configs = json.load(file).get("datasets")

    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError("--datasets JSON must contain a non-empty top-level 'datasets' list")

    logger.info(f"Loading {len(dataset_configs)} configured datasets from {args.datasets}")
    return build_multi_tts_dataloader(dataset_configs=dataset_configs, **options)


def configure_training_logging(exp_dir, accelerator):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(exp_dir, "log.txt"), mode="a"),
        ],
    )
    norm_logger = None

    if accelerator.is_main_process:
        norm_logger = logging.getLogger(f"norm_tracking.{os.path.abspath(exp_dir)}")
        norm_logger.setLevel(logging.INFO)
        norm_logger.propagate = False

        if not norm_logger.handlers:
            norm_handler = logging.FileHandler(os.path.join(exp_dir, "norms.log"), mode="a")
            norm_handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            norm_logger.addHandler(norm_handler)

    return norm_logger


def resolve_latent_buckets(data_cfg, max_latent_len):
    raw_latent_buckets = data_cfg.get("latent_length_buckets", [125, 250, 376])

    if not isinstance(raw_latent_buckets, (list, tuple)) or not raw_latent_buckets:
        raise ValueError(
            f"data.latent_length_buckets must be a non-empty list/tuple, got {raw_latent_buckets!r}"
        )

    latent_buckets = [int(b) for b in raw_latent_buckets]

    if any(b <= 0 for b in latent_buckets):
        raise ValueError(
            f"data.latent_length_buckets must contain only positive integers, got {latent_buckets}"
        )

    if latent_buckets != sorted(set(latent_buckets)):
        raise ValueError(
            f"data.latent_length_buckets must be strictly increasing with no duplicates, got {latent_buckets}"
        )

    if latent_buckets[-1] != max_latent_len:
        raise ValueError(
            f"largest data.latent_length_buckets entry must equal data.max_latent_length: "
            f"buckets={latent_buckets}, max_latent_length={max_latent_len}"
        )
    return latent_buckets


def resolve_objective_mix(train_cfg):
    objective_cfg = train_cfg.get("objective_mix", {})
    objective_cfg = {
        "p_full": float(objective_cfg.get("p_full", 0.70)),
        "p_suffix": float(objective_cfg.get("p_suffix", 0.15)),
        "p_infill": float(objective_cfg.get("p_infill", 0.15)),
        "partial_frac_min": float(objective_cfg.get("partial_frac_min", 0.70)),
        "partial_frac_max": float(objective_cfg.get("partial_frac_max", 0.95)),
    }
    objective_total = (
        objective_cfg["p_full"] + objective_cfg["p_suffix"] + objective_cfg["p_infill"]
    )

    if objective_total <= 0.0 or any(
        objective_cfg[k] < 0.0 for k in ("p_full", "p_suffix", "p_infill")
    ):
        raise ValueError(f"Invalid training.objective_mix probabilities: {objective_cfg}")

    if not (0.0 < objective_cfg["partial_frac_min"] <= objective_cfg["partial_frac_max"] <= 1.0):
        raise ValueError(f"Invalid partial mask fraction range: {objective_cfg}")
    return objective_cfg
