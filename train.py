import json
import logging
import math
import os
import time
from copy import deepcopy

import bitsandbytes as bnb
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from dataset import extract_speaker_embs_from_batch, load_titanet, configure_length_training_stage
from discriminator import ConformerDiscirminator
from model_transformer import load_config, model_from_config
from training_utils import (
    build_training_dataloader,
    build_training_optimizers,
    check_backward_finite,
    checked_grad_norm,
    cleanup_checkpoints,
    collect_muon_norm_stats,
    compute_discriminator_loss,
    compute_loss,
    configure_training_logging,
    find_latest_checkpoint,
    format_eta,
    get_num_params,
    load_checkpoint,
    load_silence_latent,
    load_training_tokenizer,
    logger,
    module_filter_fn,
    parse_args,
    reset_constant_lr,
    resolve_latent_buckets,
    resolve_objective_mix,
    save_checkpoint,
    save_nonfinite_diagnostic,
    set_requires_grad,
    unwrap_model,
    update_ema,
)

logging.getLogger("TileLang").setLevel(logging.ERROR)
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)
for _http_logger in ("httpx", "httpcore"):
    logging.getLogger(_http_logger).setLevel(logging.WARNING)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    train_cfg["length_training_stage"] = args.length_training_stage

    if args.length_training_stage:
        configure_length_training_stage(cfg, args.length_training_fraction)
        train_cfg["length_index_cache_dir"] = os.path.join(args.exp_dir, "length_indices")

    raw_max = data_cfg["max_audio_seconds"] * data_cfg["codec_rate_hz"]
    max_latent_len = int(data_cfg.get("max_latent_length", int(raw_max)))
    max_text_len = data_cfg["max_text_length"]
    latent_dim = data_cfg["latent_dim"]
    text_pad_id = data_cfg["text_pad_id"]
    codec_rate_hz = float(data_cfg["codec_rate_hz"])
    tokenizer_name, tokenizer = load_training_tokenizer(args, model_cfg, data_cfg)
    latent_buckets = resolve_latent_buckets(data_cfg, max_latent_len)

    latent_pad_value = data_cfg.get("latent_pad_value", data_cfg.get("pad_value", 0.0))
    objective_cfg = resolve_objective_mix(train_cfg)

    silence_latent_path = data_cfg.get(
        "silence_latent_path",
        "assets/silence.pt",
    )
    silence_latent = load_silence_latent(silence_latent_path, latent_dim)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        log_with=None,
    )
    os.makedirs(args.exp_dir, exist_ok=True)
    norm_logger = configure_training_logging(args.exp_dir, accelerator)

    if accelerator.is_main_process:
        with open(os.path.join(args.exp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")

    set_seed(train_cfg["seed"])
    logger.info(f"Experiment directory: {args.exp_dir}")
    logger.info(
        f"Stage: {'2' if args.stage2 else '1 (vanilla)'}"
        f"{' + speaker conditioning' if args.use_speaker else ''}"
        f"{' + adversarial' if args.use_adversarial else ''}"
    )
    logger.info(f"Max latent length: {max_latent_len}")

    if args.length_training_stage:
        logger.info(
            "Length training: cached, non-overlapping groups; %.1f%% random starting samples; "
            "2-8 consecutive clips / <=60s including Fish silence gaps; max_text_length=%s",
            100 * train_cfg["length_training_fraction"],
            max_text_len,
        )

    logger.info(
        f"Tokenizer: {tokenizer_name} | class={type(tokenizer).__name__} | vocab={len(tokenizer)} | "
        f"pad={tokenizer.pad_token_id} bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} | "
        f"add_bos={getattr(tokenizer, 'add_bos_token', None)} add_eos={getattr(tokenizer, 'add_eos_token', None)}"
    )
    logger.info(
        f"Latent buckets: {latent_buckets} | pad_value={latent_pad_value} | "
        f"upward_promotion=60/25/15,75/25,100 | silence=remaining bucket tail"
    )
    logger.info(
        "Objective mix: "
        f"full={objective_cfg['p_full']:.2f} | "
        f"suffix={objective_cfg['p_suffix']:.2f} | "
        f"infill={objective_cfg['p_infill']:.2f} | "
        f"partial_frac=[{objective_cfg['partial_frac_min']:.2f}, "
        f"{objective_cfg['partial_frac_max']:.2f}]"
    )
    logger.info(
        f"Silence latent: {silence_latent_path} | "
        f"frames={silence_latent.shape[0]} | dim={silence_latent.shape[1]}"
    )

    model = model_from_config(
        cfg,
        use_speaker_conditioning=args.use_speaker,
        speaker_emb_dim=192,
    )

    if args.length_training_stage and hasattr(model.text_encoder, "ensure_nonempty_attention_rows"):
        model.text_encoder.ensure_nonempty_attention_rows = True
        logger.info("Length training: text SDPA masks protect fully masked query rows")

    logger.info(f"Model parameters: {get_num_params(model)}")
    logger.info("Patch/refinement architecture: %s", model.architecture_signature())
    logger.info(
        "Sequence lengths: frames=%d | global_tokens=%d | refinement_tokens=%d",
        max_latent_len,
        (max_latent_len + model.patch_size - 1) // model.patch_size,
        max_latent_len if model.local_num_layers > 0 else 0,
    )
    ema = deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()

    if train_cfg.get("use_float8", True):
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            model,
            config=config,
            module_filter_fn=module_filter_fn,
        )
        logger.info("Linear precision: FP8 tensorwise (with Accelerate autocast)")
    else:
        logger.info("Linear precision: Accelerate %s; FP8 disabled", accelerator.mixed_precision)

    if train_cfg.get("compile", False):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer_name, optimizer, aux_optimizer, scheduler, aux_scheduler = build_training_optimizers(
        args, model, train_cfg
    )

    titanet_model = None

    if args.stage2 and args.use_speaker:
        titanet_model = load_titanet(accelerator.device)
        logger.info("Loaded frozen TitaNet-Large for on-the-fly speaker embedding extraction")

    dataset, dataloader = build_training_dataloader(
        args,
        tokenizer,
        data_cfg,
        train_cfg,
        max_latent_len,
        latent_buckets,
        latent_pad_value,
    )

    try:
        logger.info(f"Dataset size: {len(dataset):,} samples")
    except TypeError:
        logger.info("Dataset size: streaming")

    discriminator = None
    disc_optimizer = None

    if args.stage2 and args.use_adversarial:
        base_model = unwrap_model(model)
        disc_input_dim = base_model.num_layers * base_model.model_size
        discriminator = ConformerDiscirminator(
            input_dim=disc_input_dim,
            channels=args.disc_channels,
            num_layers=args.disc_num_layers,
            num_heads=args.disc_num_heads,
            depthwise_conv_kernel_size=args.disc_kernel,
            use_group_norm=True,
        )
        disc_optimizer = bnb.optim.AdamW8bit(
            discriminator.parameters(),
            lr=args.disc_lr,
            betas=(0.9, 0.95),
            weight_decay=train_cfg["weight_decay"],
        )

    if args.stage2 and args.use_adversarial:
        model, optimizer, dataloader, scheduler, discriminator, disc_optimizer = (
            accelerator.prepare(
                model, optimizer, dataloader, scheduler, discriminator, disc_optimizer
            )
        )
    elif aux_optimizer is not None:
        model, optimizer, aux_optimizer, dataloader, scheduler, aux_scheduler = accelerator.prepare(
            model, optimizer, aux_optimizer, dataloader, scheduler, aux_scheduler
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

    ema = ema.to(accelerator.device)
    silence_latent = silence_latent.to(accelerator.device)

    if args.resume is None:
        update_ema(ema, unwrap_model(model), decay=0)

    global_step = 0

    if args.resume is not None:
        tag = args.resume

        if tag == "latest":
            tag = find_latest_checkpoint(args.exp_dir)

            if tag is None:
                logger.warning("No checkpoint found, training from scratch")
            else:
                global_step = load_checkpoint(
                    accelerator,
                    model,
                    ema,
                    args.exp_dir,
                    tag,
                    drop_optimizer_state=args.drop_optimizer_state,
                )
        else:
            global_step = load_checkpoint(
                accelerator,
                model,
                ema,
                args.exp_dir,
                tag,
                drop_optimizer_state=args.drop_optimizer_state,
            )

    if args.reset_lr_scheduler:
        if train_cfg.get("lr_schedule") != "constant" or aux_optimizer is not None:
            raise ValueError("--reset_lr_scheduler requires constant LR and a single optimizer")

        reset_constant_lr(optimizer, scheduler, train_cfg["lr"])
        logger.info(
            "Reset scheduler only: constant LR %.2e; optimizer moments and step %d retained",
            train_cfg["lr"],
            global_step,
        )

    try:
        steps_per_epoch = len(dataloader) // train_cfg["grad_accum_steps"]
    except TypeError:
        steps_per_epoch = max(train_cfg["max_steps"] - global_step, 1)

    max_steps = train_cfg["max_steps"]
    log_every = train_cfg["log_every"]
    save_every = train_cfg["save_every"]
    keep_last_k = train_cfg["keep_last_k"]
    local_keep_last_k = int(train_cfg.get("local_keep_last_k", keep_last_k))

    if local_keep_last_k < 1:
        raise ValueError("local_keep_last_k must be >= 1")

    grad_clip = train_cfg["grad_clip"]
    ema_decay = train_cfg["ema_decay"]
    use_ckpt = train_cfg["use_checkpoint"]
    logger.info(f"Training from step {global_step} to {max_steps}")
    logger.info(f"Batch size per device: {train_cfg['batch_size']}")
    logger.info(f"Gradient accumulation: {train_cfg['grad_accum_steps']}")
    logger.info(
        f"Effective batch size: "
        f"{train_cfg['batch_size'] * train_cfg['grad_accum_steps'] * accelerator.num_processes}"
    )

    if args.stage2:
        logger.info(
            f"Stage 2 | use_speaker={args.use_speaker} | "
            f"adversarial={args.use_adversarial} | "
            f"adv_lambda={args.adv_lambda if args.use_adversarial else 0.0} | "
            f"speaker_aux_weight={args.speaker_aux_weight} | "
            f"speaker_drop_prob={args.speaker_drop_prob}"
        )

    model.train()

    if args.stage2 and args.use_adversarial:
        discriminator.train()

    running_loss = 0.0
    running_fm = 0.0
    running_speech = 0.0
    running_silence = 0.0
    running_full_frac = 0.0
    running_suffix_frac = 0.0
    running_infill_frac = 0.0
    running_zero_visible = 0.0
    running_adv = 0.0
    running_spk_aux = 0.0
    running_disc = 0.0
    log_steps = 0
    last_grad_norm = float("nan")
    pending_grad_stats = None
    start_time = time.time()
    epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0

    while global_step < max_steps:
        try:
            num_batches = len(dataloader)
        except TypeError:
            num_batches = None

        grad_accum = train_cfg["grad_accum_steps"]
        steps_per_epoch = (
            num_batches // grad_accum
            if num_batches is not None
            else max(max_steps - global_step, 1)
        )
        epoch_step = 0
        logger.info(f"Starting epoch {epoch}  ({steps_per_epoch} steps)")

        for batch_idx, batch in enumerate(dataloader):
            accumulate_ctx = (
                accelerator.accumulate(model, discriminator)
                if args.stage2 and args.use_adversarial
                else accelerator.accumulate(model)
            )

            with accumulate_ctx:
                if args.length_training_stage:
                    batch["_rng_before_loss_cpu"] = torch.get_rng_state()

                    if accelerator.device.type == "cuda":
                        batch["_rng_before_loss_cuda"] = torch.cuda.get_rng_state(
                            accelerator.device
                        )

                    batch["_microbatch_index"] = torch.tensor(batch_idx)

                if args.stage2 and args.use_speaker and titanet_model is not None:
                    batch["speaker_embs"] = extract_speaker_embs_from_batch(
                        titanet_model,
                        batch,
                        accelerator.device,
                    )

                if args.stage2 and args.use_adversarial:
                    set_requires_grad(discriminator, False)

                with accelerator.autocast():
                    loss, stats, disc_cache = compute_loss(
                        model=model,
                        batch=batch,
                        use_checkpoint=use_ckpt,
                        stage2=args.stage2,
                        use_speaker=args.use_speaker,
                        use_adversarial=(args.stage2 and args.use_adversarial),
                        discriminator=discriminator,
                        adv_lambda=(
                            args.adv_lambda if args.stage2 and args.use_adversarial else 0.0
                        ),
                        speaker_aux_weight=args.speaker_aux_weight,
                        silence_latent=silence_latent,
                        objective_mix=objective_cfg,
                    )

                if not torch.isfinite(loss.detach()).all():
                    save_nonfinite_diagnostic(
                        args.exp_dir, model, batch, global_step + 1, "forward loss"
                    )
                    raise FloatingPointError(
                        f"Non-finite forward loss before optimizer step {global_step + 1}; "
                        f'latent_shape={tuple(batch["latents"].shape)}, '
                        f'text_shape={tuple(batch["text_ids"].shape)}. '
                        "Stopped before backward, optimizer update, or checkpoint save."
                    )

                accelerator.backward(loss)

                if args.length_training_stage:
                    check_backward_finite(model, batch, args.exp_dir, global_step + 1)

                if args.stage2 and args.use_adversarial:
                    set_requires_grad(discriminator, True)
                    disc_loss = compute_discriminator_loss(
                        model=model,
                        discriminator=discriminator,
                        cache=disc_cache,
                    )
                    accelerator.backward(disc_loss)

                    if accelerator.sync_gradients:
                        track_norms = (
                            optimizer_name == "muon" and (global_step + 1) % log_every == 0
                        )

                        if track_norms:
                            pending_grad_stats = collect_muon_norm_stats(model, grads=True)

                        last_grad_norm = checked_grad_norm(
                            accelerator, model, batch, args.exp_dir, global_step + 1, grad_clip
                        )
                        disc_grad_norm = checked_grad_norm(
                            accelerator,
                            discriminator,
                            batch,
                            args.exp_dir,
                            global_step + 1,
                            grad_clip,
                        )

                        if not math.isfinite(last_grad_norm) or not math.isfinite(disc_grad_norm):
                            save_nonfinite_diagnostic(
                                args.exp_dir, model, batch, global_step + 1, "gradient norm"
                            )
                            raise FloatingPointError(
                                f"Non-finite gradient norm at optimizer step {global_step + 1}; no update or checkpoint saved"
                            )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if aux_optimizer is not None:
                        aux_optimizer.step()
                        aux_scheduler.step()
                        aux_optimizer.zero_grad()

                    disc_optimizer.step()
                    disc_optimizer.zero_grad()
                else:
                    if accelerator.sync_gradients:
                        track_norms = (
                            optimizer_name == "muon" and (global_step + 1) % log_every == 0
                        )

                        if track_norms:
                            pending_grad_stats = collect_muon_norm_stats(model, grads=True)

                        last_grad_norm = checked_grad_norm(
                            accelerator, model, batch, args.exp_dir, global_step + 1, grad_clip
                        )

                        if not math.isfinite(last_grad_norm):
                            save_nonfinite_diagnostic(
                                args.exp_dir, model, batch, global_step + 1, "gradient norm"
                            )
                            raise FloatingPointError(
                                f"Non-finite gradient norm at optimizer step {global_step + 1}; no update or checkpoint saved"
                            )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if aux_optimizer is not None:
                        aux_optimizer.step()
                        aux_scheduler.step()
                        aux_optimizer.zero_grad()

                    disc_loss = loss.new_zeros(())

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            epoch_step += 1
            update_ema(ema, unwrap_model(model), ema_decay)
            running_loss += loss.detach().item()
            running_fm += stats["fm_loss"].item()
            running_speech += stats["speech_loss"].item()
            running_silence += stats["silence_loss"].item()
            running_full_frac += stats["full_tts_frac"].item()
            running_suffix_frac += stats["suffix_frac"].item()
            running_infill_frac += stats["infill_frac"].item()
            running_zero_visible += stats["zero_visible_frac"].item()
            running_adv += stats["gen_adv_loss"].item()
            running_spk_aux += stats["speaker_aux_loss"].item()
            running_disc += disc_loss.detach().item()
            log_steps += 1

            if global_step % log_every == 0:
                n = log_steps

                if optimizer_name == "muon":
                    lr_text = f"muon_lr={optimizer.param_groups[0]['lr']:.2e} | adamw_lr={aux_optimizer.param_groups[0]['lr']:.2e}"
                else:
                    lr_text = f"lr={optimizer.param_groups[0]['lr']:.2e}"

                elapsed = time.time() - start_time
                sps = n / elapsed
                remaining = (max_steps - global_step) / sps
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                speed = f"{sps:.2f} it/s" if sps >= 1.0 else f"{elapsed / n:.2f} s/it"

                if optimizer_name == "muon" and norm_logger is not None:
                    weight_stats = collect_muon_norm_stats(model, grads=False)
                    grad_stats = pending_grad_stats or {}
                    norm_logger.info(
                        "step=%d | muon_lr=%.8g | adamw_lr=%.8g | grad_global_preclip=%.6g | "
                        "muon_grad_rms=%.6g muon_grad_max=%.6g muon_grad_l2=%.6g | "
                        "q_grad_rms=%.6g q_grad_max=%.6g q_grad_l2=%.6g | "
                        "k_grad_rms=%.6g k_grad_max=%.6g k_grad_l2=%.6g | "
                        "muon_w_rms=%.6g muon_w_max=%.6g muon_w_l2=%.6g | "
                        "q_w_rms=%.6g q_w_max=%.6g q_w_l2=%.6g | "
                        "k_w_rms=%.6g k_w_max=%.6g k_w_l2=%.6g | "
                        "peak_muon_grad=%.6g:%s peak_q_grad=%.6g:%s peak_k_grad=%.6g:%s | "
                        "peak_muon_w=%.6g:%s peak_q_w=%.6g:%s peak_k_w=%.6g:%s",
                        global_step,
                        optimizer.param_groups[0]["lr"],
                        aux_optimizer.param_groups[0]["lr"],
                        last_grad_norm,
                        grad_stats.get("muon_rms", 0.0),
                        grad_stats.get("muon_max", 0.0),
                        grad_stats.get("muon_l2", 0.0),
                        grad_stats.get("q_rms", 0.0),
                        grad_stats.get("q_max", 0.0),
                        grad_stats.get("q_l2", 0.0),
                        grad_stats.get("k_rms", 0.0),
                        grad_stats.get("k_max", 0.0),
                        grad_stats.get("k_l2", 0.0),
                        weight_stats["muon_rms"],
                        weight_stats["muon_max"],
                        weight_stats["muon_l2"],
                        weight_stats["q_rms"],
                        weight_stats["q_max"],
                        weight_stats["q_l2"],
                        weight_stats["k_rms"],
                        weight_stats["k_max"],
                        weight_stats["k_l2"],
                        grad_stats.get("muon_peak_l2", 0.0),
                        grad_stats.get("muon_peak_name", "none"),
                        grad_stats.get("q_peak_l2", 0.0),
                        grad_stats.get("q_peak_name", "none"),
                        grad_stats.get("k_peak_l2", 0.0),
                        grad_stats.get("k_peak_name", "none"),
                        weight_stats["muon_peak_l2"],
                        weight_stats["muon_peak_name"],
                        weight_stats["q_peak_l2"],
                        weight_stats["q_peak_name"],
                        weight_stats["k_peak_l2"],
                        weight_stats["k_peak_name"],
                    )

                    for handler in norm_logger.handlers:
                        handler.flush()

                    pending_grad_stats = None

                if args.stage2:
                    logger.info(
                        f"step={global_step:>7d} | "
                        f"epoch={epoch} [{epoch_step}/{steps_per_epoch}] | "
                        f"loss={running_loss / n:.4f} | "
                        f"fm={running_fm / n:.4f} | "
                        f"speech={running_speech / n:.4f} | silence={running_silence / n:.4f} | "
                        f"modes={running_full_frac / n:.2f}/{running_suffix_frac / n:.2f}/{running_infill_frac / n:.2f} | "
                        f"zero_ctx={running_zero_visible / n:.2f} | "
                        f"spk_aux={running_spk_aux / n:.4f} | "
                        f"g_adv={running_adv / n:.4f} | "
                        f"d={running_disc / n:.4f} | "
                        f"{lr_text} | {speed} | "
                        f"mem={mem_gb:.1f}GB | eta={format_eta(remaining)}"
                    )
                else:
                    logger.info(
                        f"step={global_step:>7d} | "
                        f"epoch={epoch} [{epoch_step}/{steps_per_epoch}] | "
                        f"loss={running_loss / n:.4f} | "
                        f"speech={running_speech / n:.4f} | silence={running_silence / n:.4f} | "
                        f"modes={running_full_frac / n:.2f}/{running_suffix_frac / n:.2f}/{running_infill_frac / n:.2f} | "
                        f"zero_ctx={running_zero_visible / n:.2f} | "
                        f"{lr_text} | {speed} | "
                        f"mem={mem_gb:.1f}GB | eta={format_eta(remaining)}"
                    )

                running_loss = running_fm = running_speech = running_silence = 0.0
                running_full_frac = running_suffix_frac = running_infill_frac = (
                    running_zero_visible
                ) = 0.0
                running_adv = running_spk_aux = running_disc = 0.0
                log_steps = 0
                start_time = time.time()

            if global_step % save_every == 0:
                tag = f"step_{global_step:07d}"
                save_checkpoint(
                    accelerator,
                    model,
                    ema,
                    global_step,
                    args.exp_dir,
                    tag,
                    checkpoint_bucket=train_cfg.get("checkpoint_bucket"),
                    keep_last_k=keep_last_k,
                    local_keep_last_k=local_keep_last_k,
                )

                if accelerator.is_main_process:
                    cleanup_checkpoints(args.exp_dir, local_keep_last_k)

            if global_step >= max_steps:
                break

        epoch += 1

    tag = f"step_{global_step:07d}"
    save_checkpoint(
        accelerator,
        model,
        ema,
        global_step,
        args.exp_dir,
        tag,
        checkpoint_bucket=train_cfg.get("checkpoint_bucket"),
        keep_last_k=keep_last_k,
        local_keep_last_k=local_keep_last_k,
    )

    if accelerator.is_main_process:
        cleanup_checkpoints(args.exp_dir, local_keep_last_k)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
