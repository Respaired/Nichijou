import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from transformers import AutoModel, ModernBertConfig, ModernBertModel
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)

_FA_VARLEN = None
FLASH_ATTN_VERSION = 0

# gotta try 'em all
try:
    from flash_attn_interface import flash_attn_varlen_func as _FA_VARLEN
    FLASH_ATTN_VERSION = 3
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func as _FA_VARLEN
        FLASH_ATTN_VERSION = 2
    except ImportError:
        _FA_VARLEN = None
        FLASH_ATTN_VERSION = 0


def flash_attn_available() -> bool:
    return _FA_VARLEN is not None


def _unpad(
    x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    x_unpad = x.reshape(-1, *x.shape[2:])[indices]

    return x_unpad, indices, cu_seqlens, int(seqlens.max().item())


def _pad(
    x_unpad: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
) -> torch.Tensor:
    out = torch.zeros(
        batch * seqlen,
        *x_unpad.shape[1:],
        dtype=x_unpad.dtype,
        device=x_unpad.device,
    )
    out[indices] = x_unpad

    return out.view(batch, seqlen, *x_unpad.shape[1:])


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict):
        raise ValueError("The config must contain a model object")
    model_cfg = cfg["model"]

    if not isinstance(model_cfg.get("decoder"), dict):
        raise ValueError("The config must contain model.decoder")

    if "local_detail" in model_cfg and not isinstance(model_cfg["local_detail"], dict):
        raise ValueError("model.local_detail must be an object when present")

    return cfg


def model_from_config(
    cfg: Dict[str, Any],
    use_speaker_conditioning: bool = False,
    speaker_emb_dim: int = 192,
) -> "NichijouDiT":
    model_config = cfg["model"]
    reference_config = model_config["reference_projector"]
    decoder_config = model_config["decoder"]
    conditioning_config = model_config["conditioning"]
    local_detail_config = model_config.get("local_detail", {})
    hourglass_config = decoder_config.get("hourglass", {})

    if not isinstance(hourglass_config, dict):
        raise ValueError("model.decoder.hourglass must be an object when present")

    return NichijouDiT(
        latent_size=model_config["latent_size"],
        model_size=decoder_config["model_size"],
        num_layers=decoder_config["num_layers"],
        num_heads=decoder_config["num_heads"],
        intermediate_size=decoder_config["intermediate_size"],
        norm_eps=model_config.get("norm_eps", 1e-6),
        text_encoder_config=model_config["text_encoder"],
        reference_latent_dim=reference_config["latent_dim"],
        timestep_embed_size=conditioning_config["timestep_embed_size"],
        adaln_rank=conditioning_config["adaln_rank"],
        use_speaker_conditioning=use_speaker_conditioning,
        speaker_emb_dim=speaker_emb_dim,
        use_flash_attn=decoder_config.get("use_flash_attn", True),
        checkpoint_every_n=decoder_config.get("checkpoint_every_n", 2),
        patch_size=decoder_config.get("patch_size", 1),
        local_num_layers=local_detail_config.get("num_layers", 0),
        local_model_size=local_detail_config.get("model_size", 384),
        local_num_heads=local_detail_config.get("num_heads", 6),
        local_intermediate_size=local_detail_config.get("intermediate_size", 1536),
        local_adaln_rank=local_detail_config.get("adaln_rank", 128),
        hourglass_enabled=hourglass_config.get("enabled", False),
        hourglass_pre_layers=hourglass_config.get("pre_layers", 0),
        hourglass_post_layers=hourglass_config.get("post_layers", 0),
    )


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    *,
    start: float = 0.0,
    step: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be positive and even, got dim={dim}")
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    positions = torch.arange(end, dtype=torch.float32, device=device) * step + start
    phases = torch.outer(positions, freqs)

    return torch.stack((torch.cos(phases), torch.sin(phases)), dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Head dim must be even for RoPE, got {x.shape[-1]}")

    if freqs_cis.shape[-1] != 2:
        raise ValueError(
            "Expected real RoPE table with shape [S, D // 2, 2] "
            "where the last dimension is [cos, sin]"
        )
    x_dtype = x.dtype
    x_pair = x.float().reshape(*x.shape[:3], -1, 2)
    x_real = x_pair[..., 0]
    x_imag = x_pair[..., 1]
    cos = freqs_cis[..., 0].to(device=x.device, dtype=torch.float32)[None, :, None, :]
    sin = freqs_cis[..., 1].to(device=x.device, dtype=torch.float32)[None, :, None, :]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    out = torch.stack((out_real, out_imag), dim=-1).reshape_as(x.float())

    return out.to(dtype=x_dtype)


def get_timestep_embedding(timestep: torch.Tensor, embed_size: int) -> torch.Tensor:
    if embed_size % 2 != 0:
        raise ValueError(f"timestep embedding dim must be even, got {embed_size}")
    half = embed_size // 2
    freqs = 1000.0 * torch.exp(
        -torch.log(torch.tensor(10000.0, device=timestep.device, dtype=torch.float32))
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timestep.device)
        / half
    )
    args = timestep[..., None] * freqs[None]

    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(timestep.dtype)


class RMSNorm(nn.Module):

    def __init__(self, model_size: int | Tuple[int, int], eps: float):
        super().__init__()
        self.eps = eps

        if isinstance(model_size, int):
            model_size = (model_size,)
        self.weight = nn.Parameter(torch.ones(model_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight

        return x.to(x_dtype)


class LowRankAdaLN(nn.Module):

    def __init__(self, model_size: int, rank: int, eps: float):
        super().__init__()
        self.eps = eps
        self.shift_down = nn.Linear(model_size, rank, bias=False)
        self.scale_down = nn.Linear(model_size, rank, bias=False)
        self.gate_down = nn.Linear(model_size, rank, bias=False)
        self.shift_up = nn.Linear(rank, model_size, bias=True)
        self.scale_up = nn.Linear(rank, model_size, bias=True)
        self.gate_up = nn.Linear(rank, model_size, bias=True)

    def forward(
        self, x: torch.Tensor, cond_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = cond_embed.chunk(3, dim=-1)
        shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(F.silu(gate))) + gate
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (scale + 1.0) + shift
        gate = torch.tanh(gate)

        return x.to(x_dtype), gate


class SpeakerFiLMAdapter(nn.Module):

    def __init__(
        self,
        speaker_emb_dim: int,
        model_size: int,
        num_layers: int,
        num_modulation_points: int = 2,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modulation_points = num_modulation_points
        self.model_size = model_size
        total_params = num_layers * num_modulation_points * 2 * model_size
        self.mlp = nn.Sequential(
            nn.Linear(speaker_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, total_params),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, speaker_emb: torch.Tensor) -> torch.Tensor:
        bsz = speaker_emb.shape[0]
        params = self.mlp(speaker_emb)

        return params.view(
            bsz,
            self.num_layers,
            self.num_modulation_points,
            2,
            self.model_size,
        )


class ReferenceProjector(nn.Module):

    def __init__(self, latent_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, text_dim, bias=False),
            nn.SiLU(),
            nn.Linear(text_dim, text_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PretrainedMmBertEncoder(nn.Module):

    def __init__(
        self,
        pretrained_name: str = "jhu-clsp/mmBERT-base",
        use_custom_tokenizer: bool = False,
        custom_vocab_size: Optional[int] = None,
        pad_token_id: int = 1,
        init_noise_std: float = 0.02,
        num_layers: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        load_kwargs: Dict[str, Any] = {}

        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
        self.bert = AutoModel.from_pretrained(pretrained_name, **load_kwargs)
        self.hidden_size: int = self.bert.config.hidden_size

        if num_layers is not None:
            self._truncate_layers(num_layers)

        if use_custom_tokenizer:
            if custom_vocab_size is None:
                raise ValueError(
                    "custom_vocab_size is required when use_custom_tokenizer=True"
                )
            self._swap_embeddings(custom_vocab_size, pad_token_id, init_noise_std)

    def _truncate_layers(self, num_keep: int) -> None:
        total = len(self.bert.layers)

        if num_keep < 1 or num_keep > total:
            raise ValueError(f"num_layers={num_keep} must be in [1, {total}]")
        self.bert.layers = nn.ModuleList(list(self.bert.layers)[:num_keep])
        self.bert.config.num_hidden_layers = num_keep

    @torch.no_grad()
    def _swap_embeddings(
        self,
        new_vocab_size: int,
        pad_token_id: int,
        noise_std: float,
    ) -> None:
        old_emb = self.bert.get_input_embeddings()
        emb_mean = old_emb.weight.data.mean(dim=0, keepdim=True)
        new_emb = nn.Embedding(
            new_vocab_size, self.hidden_size, padding_idx=pad_token_id
        )
        new_emb.weight.data.copy_(emb_mean.expand(new_vocab_size, -1))
        new_emb.weight.data.add_(torch.randn_like(new_emb.weight.data) * noise_std)
        new_emb.weight.data[pad_token_id].zero_()
        self.bert.set_input_embeddings(new_emb)
        self.bert.config.vocab_size = new_vocab_size
        self.bert.config.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = mask.to(torch.long) if mask is not None else None
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)

        return out.last_hidden_state


def safe_sdpa_attention_rows(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None or mask.ndim != 4:
        return mask
    first_key = torch.arange(mask.shape[-1], device=mask.device) == 0

    if mask.dtype == torch.bool:
        empty = ~mask.any(dim=-1, keepdim=True)

        return mask | (empty & first_key)
    blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
    empty = blocked.all(dim=-1, keepdim=True)

    return mask.masked_fill(empty & first_key, 0.0)


class Encoder_Reinit(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 8192,
        pad_token_id: int = 1,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=model_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
        )

        if attn_implementation is not None:
            self.bert = ModernBertModel._from_config(
                config, attn_implementation=attn_implementation
            )
        else:
            self.bert = ModernBertModel(config)
        self.hidden_size: int = model_size
        self.ensure_nonempty_attention_rows = False

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = mask.to(torch.long) if mask is not None else None

        if self.ensure_nonempty_attention_rows and self.bert.config._attn_implementation == 'sdpa':
            dummy = self.bert.embeddings.tok_embeddings.weight.new_empty((*input_ids.shape, 1))
            arguments = dict(config=self.bert.config, inputs_embeds=dummy,
                             attention_mask=attn_mask)
            attn_mask = {
                'full_attention': safe_sdpa_attention_rows(create_bidirectional_mask(**arguments)),
                'sliding_attention': safe_sdpa_attention_rows(
                    create_bidirectional_sliding_window_mask(**arguments)),
            }
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)

        return out.last_hidden_state


def build_text_encoder(cfg: Dict[str, Any]) -> nn.Module:
    kind = cfg.get("kind", "scratch")

    attn_implementation = cfg.get("attn_implementation")

    if kind == "using_pretrained":
        return PretrainedMmBertEncoder(
            pretrained_name=cfg.get("pretrained_name", "jhu-clsp/mmBERT-base"),
            use_custom_tokenizer=cfg.get("use_custom_tokenizer", False),
            custom_vocab_size=cfg.get("vocab_size"),
            pad_token_id=cfg.get("pad_token_id", 1),
            init_noise_std=cfg.get("init_noise_std", 0.02),
            num_layers=cfg.get("num_layers"),
            attn_implementation=attn_implementation,
        )
    elif kind == "scratch":
        return Encoder_Reinit(
            vocab_size=cfg["vocab_size"],
            model_size=cfg["model_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg.get("max_position_embeddings", 8192),
            pad_token_id=cfg.get("pad_token_id", 1),
            attn_implementation=attn_implementation,
        )
    else:
        raise ValueError(f"Unknown text encoder kind: {kind!r}")


class TextJointAttention(nn.Module):

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        text_model_size: int,
        norm_eps: float,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        if model_size % num_heads != 0:
            raise ValueError(f"model_size={model_size} not divisible by num_heads={num_heads}")

        if num_heads % 2 != 0:
            raise ValueError(f"num_heads must be even, got {num_heads}")
        self.num_heads = num_heads
        self.head_dim = model_size // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

        if use_flash_attn and self.head_dim > 256:
            raise ValueError(
                f"FlashAttention supports head_dim <= 256, got {self.head_dim}; "
                "set use_flash_attn=False"
            )
        self.use_flash_attn = use_flash_attn
        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.wk_text = nn.Linear(text_model_size, model_size, bias=False)
        self.wv_text = nn.Linear(text_model_size, model_size, bias=False)
        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.gate = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)

    def _apply_rotary_half(self, y: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=-2)
        y1 = apply_rotary_emb(y1, fc)

        return torch.cat([y1, y2], dim=-2)

    def _flash_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        latent_mask: Optional[torch.Tensor],
        context_key_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, S = xq.shape[:2]
        ctx_len = xk.shape[1] - S
        device = xq.device
        lat_key = (
            latent_mask.to(torch.bool)
            if latent_mask is not None
            else torch.ones(B, S, dtype=torch.bool, device=device)
        )
        ctx_key = (
            context_key_mask.to(torch.bool)
            if context_key_mask is not None
            else torch.ones(B, ctx_len, dtype=torch.bool, device=device)
        )
        key_mask = torch.cat([lat_key, ctx_key], dim=1)

        first_q = torch.arange(S, device=device)[None, :] == 0
        safe_queries = lat_key | (~lat_key.any(dim=1, keepdim=True) & first_q)
        first_k = torch.arange(S + ctx_len, device=device)[None, :] == 0
        safe_keys = key_mask | (~key_mask.any(dim=1, keepdim=True) & first_k)
        xq = xq.masked_fill(~lat_key[:, :, None, None], 0.0)
        xk = xk.masked_fill(~key_mask[:, :, None, None], 0.0)
        xv = xv.masked_fill(~key_mask[:, :, None, None], 0.0)
        q_unpad, q_indices, cu_q, max_q = _unpad(xq, safe_queries)
        k_unpad, k_indices, cu_k, max_k = _unpad(xk, safe_keys)
        v_unpad = xv.reshape(-1, *xv.shape[2:])[k_indices]
        out = _FA_VARLEN(
            q_unpad, k_unpad, v_unpad,
            cu_q, cu_k, max_q, max_k,
        )

        if isinstance(out, tuple):
            out = out[0]

        return _pad(out, q_indices, B, S).masked_fill(
            ~lat_key[:, :, None, None], 0.0
        )

    def _can_use_flash_attention(self, query, attn_mask, context_bias):
        return (
            self.use_flash_attn
            and _FA_VARLEN is not None
            and attn_mask is None
            and context_bias is None
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
        )

    def _sdpa_forward(
        self,
        query,
        key,
        value,
        latent_mask,
        context_key_mask,
        context_bias,
        attn_mask,
    ):
        batch_size, sequence_length = query.shape[:2]
        context_length = key.shape[1] - sequence_length

        if attn_mask is None:
            latent_keys = (
                latent_mask.to(torch.bool)
                if latent_mask is not None
                else torch.ones(
                    batch_size, sequence_length, dtype=torch.bool, device=query.device
                )
            )
            context_keys = (
                context_key_mask.to(torch.bool)
                if context_key_mask is not None
                else torch.ones(
                    batch_size, context_length, dtype=torch.bool, device=query.device
                )
            )

            key_mask = torch.cat([latent_keys, context_keys], dim=1)
            first_key = torch.arange(key.shape[1], device=query.device)[None, :] == 0
            safe_keys = key_mask | (~key_mask.any(dim=1, keepdim=True) & first_key)

            key = key.masked_fill(~key_mask[:, :, None, None], 0.0)
            value = value.masked_fill(~key_mask[:, :, None, None], 0.0)

            if context_bias is None:
                attn_mask = safe_keys[:, None, None, :]
            else:
                attn_mask = torch.zeros(
                    batch_size,
                    1,
                    sequence_length,
                    key.shape[1],
                    dtype=torch.float32,
                    device=query.device,
                )
                attn_mask[:, :, :, sequence_length:] += context_bias.to(attn_mask.dtype)
                attn_mask = attn_mask.masked_fill(
                    ~safe_keys[:, None, None, :], float("-inf")
                )

        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(query.dtype)

        return F.scaled_dot_product_attention(
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(1, 2)

    def forward(
        self,
        x,
        freqs_cis,
        kv_cache_text,
        start_pos,
        attn_mask=None,
        latent_mask=None,
        context_key_mask=None,
        context_bias=None,
    ) -> torch.Tensor:
        batch_size, sequence_length = x.shape[:2]

        query = self.q_norm(
            self.wq(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        self_key = self.k_norm(
            self.wk(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        self_value = self.wv(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        position = 0 if start_pos is None else start_pos
        freqs = freqs_cis[position:position + sequence_length]
        query = self._apply_rotary_half(query, freqs)
        self_key = self._apply_rotary_half(self_key, freqs)

        text_key, text_value = kv_cache_text
        key = torch.cat([self_key, text_key.to(self_key.dtype)], dim=1)
        value = torch.cat([self_value, text_value.to(self_value.dtype)], dim=1)

        if self._can_use_flash_attention(query, attn_mask, context_bias):
            output = self._flash_forward(
                query, key, value, latent_mask, context_key_mask
            )
        else:
            output = self._sdpa_forward(
                query, key, value, latent_mask, context_key_mask,
                context_bias, attn_mask,
            )

        output = output.reshape(batch_size, sequence_length, -1)
        output = self.wo(output * torch.sigmoid(self.gate(x)))

        if latent_mask is not None:
            output = output.masked_fill(~latent_mask[:, :, None].to(torch.bool), 0.0)

        return output

    def get_kv_cache_text(self, text_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = text_state.shape[:2]

        key = self.wk_text(text_state).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        value = self.wv_text(text_state).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        return self.k_norm(key), value


class MLP(nn.Module):

    def __init__(self, model_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(
        self, model_size, num_heads, intermediate_size,
        norm_eps, text_model_size, adaln_rank,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.attention = TextJointAttention(
            model_size, num_heads, text_model_size, norm_eps,
            use_flash_attn=use_flash_attn,
        )
        self.mlp = MLP(model_size, intermediate_size)
        self.attention_adaln = LowRankAdaLN(model_size, adaln_rank, norm_eps)
        self.mlp_adaln = LowRankAdaLN(model_size, adaln_rank, norm_eps)

    @staticmethod
    def _apply_speaker_film(
        x: torch.Tensor,
        film_params: Optional[torch.Tensor],
        point_idx: int,
    ) -> torch.Tensor:
        if film_params is None:
            return x
        scale = film_params[:, point_idx, 0, :]
        shift = film_params[:, point_idx, 1, :]

        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self, x, cond_embed, context_key_mask, freqs_cis,
        kv_cache_text, start_pos,
        latent_mask=None, context_bias=None,
        speaker_film: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm, attention_gate = self.attention_adaln(x, cond_embed)
        x_norm = self._apply_speaker_film(x_norm, speaker_film, point_idx=0)
        x = x + attention_gate * self.attention(
            x=x_norm, freqs_cis=freqs_cis, kv_cache_text=kv_cache_text,
            start_pos=start_pos, latent_mask=latent_mask,
            context_key_mask=context_key_mask, context_bias=context_bias,
        )

        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x_norm = self._apply_speaker_film(x_norm, speaker_film, point_idx=1)
        x = x + mlp_gate * self.mlp(x_norm)

        return x


class LocalSelfAttention(nn.Module):

    def __init__(self, model_size: int, num_heads: int, norm_eps: float):
        super().__init__()

        if model_size <= 0 or num_heads <= 0 or model_size % num_heads != 0:
            raise ValueError(
                f"local model_size={model_size} must be positive and divisible "
                f"by positive num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = model_size // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError(f"local head_dim must be even for RoPE, got {self.head_dim}")
        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.wo = nn.Linear(model_size, model_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, length, _ = x.shape

        if latent_mask is not None:
            latent_mask = latent_mask.to(torch.bool)
            x = x.masked_fill(~latent_mask[:, :, None], 0.0)
        q = self.q_norm(self.wq(x).reshape(b, length, self.num_heads, self.head_dim))
        k = self.k_norm(self.wk(x).reshape(b, length, self.num_heads, self.head_dim))
        v = self.wv(x).reshape(b, length, self.num_heads, self.head_dim)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        mask = None

        if latent_mask is not None:
            first_key = torch.arange(length, device=x.device)[None, :] == 0
            safe_keys = latent_mask | (~latent_mask.any(dim=1, keepdim=True) & first_key)
            mask = safe_keys[:, None, None, :]
        q, k, v = (y.transpose(1, 2).to(v.dtype) for y in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
        )
        out = self.wo(out.transpose(1, 2).reshape(b, length, -1))

        if latent_mask is not None:
            out = out.masked_fill(~latent_mask[:, :, None], 0.0)

        return out


class LocalBlock(nn.Module):

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        adaln_rank: int,
    ):
        super().__init__()
        self.attention = LocalSelfAttention(model_size, num_heads, norm_eps)
        self.mlp = MLP(model_size, intermediate_size)
        self.attention_adaln = LowRankAdaLN(model_size, adaln_rank, norm_eps)
        self.mlp_adaln = LowRankAdaLN(model_size, adaln_rank, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        freqs_cis: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent_mask is not None:
            x = x.masked_fill(~latent_mask[:, :, None], 0.0)
        x_norm, gate = self.attention_adaln(x, cond_embed)
        x = x + gate * self.attention(x_norm, freqs_cis, latent_mask)
        x_norm, gate = self.mlp_adaln(x, cond_embed)
        x = x + gate * self.mlp(x_norm)

        if latent_mask is not None:
            x = x.masked_fill(~latent_mask[:, :, None], 0.0)

        return x

@dataclass


class _DenoiseRunState:
    cond_embed: torch.Tensor
    context_mask: Optional[torch.Tensor]
    context_bias: Optional[torch.Tensor]
    speaker_film: Optional[torch.Tensor]
    use_checkpoint: bool
    hidden_states: Optional[list[torch.Tensor]]


class NichijouDiT(nn.Module):
    TITANET_EMB_DIM = 192

    def __init__(
        self,
        latent_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        text_encoder_config: Dict[str, Any],
        reference_latent_dim: int,
        timestep_embed_size: int,
        adaln_rank: int,
        use_speaker_conditioning: bool = False,
        speaker_emb_dim: int = 192,
        use_flash_attn: bool = True,
        checkpoint_every_n: int = 2,
        patch_size: int = 1,
        local_num_layers: int = 0,
        local_model_size: int = 384,
        local_num_heads: int = 6,
        local_intermediate_size: int = 1536,
        local_adaln_rank: int = 128,
        hourglass_enabled: bool = False,
        hourglass_pre_layers: int = 0,
        hourglass_post_layers: int = 0,
    ):
        super().__init__()
        self._validate_architecture(
            model_size=model_size,
            num_heads=num_heads,
            patch_size=patch_size,
            local_num_layers=local_num_layers,
            local_intermediate_size=local_intermediate_size,
            local_adaln_rank=local_adaln_rank,
            hourglass_enabled=hourglass_enabled,
            hourglass_pre_layers=hourglass_pre_layers,
            hourglass_post_layers=hourglass_post_layers,
            num_layers=num_layers,
            checkpoint_every_n=checkpoint_every_n,
        )

        hourglass_enabled = bool(hourglass_enabled)
        hourglass_pre_layers = int(hourglass_pre_layers)
        hourglass_post_layers = int(hourglass_post_layers)

        self.patch_size = int(patch_size)
        self.hourglass_enabled = hourglass_enabled
        self.hourglass_pre_layers = hourglass_pre_layers
        self.hourglass_post_layers = hourglass_post_layers
        self.middle_num_layers = (
            num_layers - hourglass_pre_layers - hourglass_post_layers
            if hourglass_enabled else num_layers
        )
        self.local_num_layers = int(local_num_layers)
        self.local_model_size = int(local_model_size) if local_num_layers > 0 else 0
        self.latent_size = latent_size
        self.model_size = model_size
        self.num_heads = num_heads
        self.head_dim = model_size // num_heads
        self.timestep_embed_size = timestep_embed_size
        self.num_layers = num_layers
        self.use_speaker_conditioning = use_speaker_conditioning
        self.speaker_emb_dim = speaker_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpoint_every_n = int(checkpoint_every_n)

        self.text_encoder = build_text_encoder(text_encoder_config)
        text_model_size = self.text_encoder.hidden_size
        self.text_model_size = text_model_size
        self.text_norm = RMSNorm(text_model_size, norm_eps)

        self.reference_proj = ReferenceProjector(reference_latent_dim, text_model_size)

        self.cond_module = nn.Sequential(
            nn.Linear(timestep_embed_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size * 3, bias=False),
        )

        self.in_proj = nn.Linear(latent_size, model_size, bias=True)
        self.span_mask_proj = nn.Linear(1, model_size, bias=True)
        nn.init.normal_(self.span_mask_proj.weight, std=0.02)
        nn.init.zeros_(self.span_mask_proj.bias)

        def make_blocks(count: int) -> nn.ModuleList:
            return nn.ModuleList([
                TransformerBlock(
                    model_size, num_heads, intermediate_size,
                    norm_eps, text_model_size, adaln_rank,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(count)
            ])
        self.pre_blocks = make_blocks(self.hourglass_pre_layers)
        self.blocks = make_blocks(self.middle_num_layers)
        self.post_blocks = make_blocks(self.hourglass_post_layers)

        if self.patch_size > 1:
            self.patch_merge = nn.Linear(model_size * self.patch_size, model_size, bias=True)
            self.patch_expand = nn.Linear(model_size, model_size * self.patch_size, bias=True)
        else:
            self.patch_merge = None
            self.patch_expand = None

        if self.local_num_layers > 0:
            self.local_in_proj = nn.Linear(model_size, self.local_model_size, bias=True)

            self.local_cond_module = nn.Sequential(
                nn.Linear(timestep_embed_size, self.local_model_size, bias=False),
                nn.SiLU(),
                nn.Linear(self.local_model_size, self.local_model_size * 3, bias=False),
            )
            self.local_blocks = nn.ModuleList([
                LocalBlock(
                    self.local_model_size, local_num_heads,
                    local_intermediate_size, norm_eps, local_adaln_rank,
                )
                for _ in range(self.local_num_layers)
            ])
            self.local_out_norm = RMSNorm(self.local_model_size, norm_eps)
            head_size = self.local_model_size
        else:
            self.local_in_proj = None
            self.local_cond_module = None
            self.local_blocks = None
            self.local_out_norm = None
            head_size = model_size

        self.out_norm = RMSNorm(model_size, norm_eps) if self.local_num_layers == 0 else None
        self.out_proj = nn.Linear(head_size, latent_size, bias=True)
        self.cond_proj = nn.Linear(latent_size, model_size, bias=True)

        if use_speaker_conditioning:
            self.speaker_film_adapter = SpeakerFiLMAdapter(
                speaker_emb_dim=speaker_emb_dim,
                model_size=model_size,
                num_layers=num_layers,
                num_modulation_points=2,
                hidden_dim=1024,
            )
            self.speaker_reg_head = nn.Sequential(
                nn.Linear(model_size, model_size * 2),
                nn.SiLU(),
                nn.Linear(model_size * 2, speaker_emb_dim),
            )
        else:
            self.speaker_film_adapter = None
            self.speaker_reg_head = None

    @staticmethod
    def _validate_architecture(
        model_size,
        num_heads,
        patch_size,
        local_num_layers,
        local_intermediate_size,
        local_adaln_rank,
        hourglass_enabled,
        hourglass_pre_layers,
        hourglass_post_layers,
        num_layers,
        checkpoint_every_n,
    ):
        if model_size % num_heads != 0:
            raise ValueError(
                f"model_size={model_size} not divisible by num_heads={num_heads}"
            )

        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")

        if local_num_layers < 0:
            raise ValueError(
                f"local_num_layers must be >= 0, got {local_num_layers}"
            )

        if local_num_layers > 0 and (
            local_intermediate_size <= 0 or local_adaln_rank <= 0
        ):
            raise ValueError(
                "local intermediate_size and adaln_rank must be positive"
            )

        if hourglass_enabled:
            if patch_size <= 1:
                raise ValueError("hourglass mode requires patch_size > 1")

            if local_num_layers > 0:
                raise ValueError(
                    "hourglass mode and local_detail are mutually exclusive"
                )

            if hourglass_pre_layers < 1 or hourglass_post_layers < 1:
                raise ValueError(
                    "hourglass pre_layers and post_layers must both be >= 1"
                )

            if hourglass_pre_layers + hourglass_post_layers >= num_layers:
                raise ValueError(
                    "hourglass pre_layers + post_layers must be smaller "
                    "than decoder.num_layers"
                )

        elif hourglass_pre_layers != 0 or hourglass_post_layers != 0:
            raise ValueError(
                "hourglass pre/post layers require hourglass.enabled=true"
            )

        if checkpoint_every_n < 1:
            raise ValueError(
                f"checkpoint_every_n must be >= 1, got {checkpoint_every_n}"
            )

        if (model_size // num_heads) % 2 != 0:
            raise ValueError(
                f"head_dim must be even, got {model_size // num_heads}"
            )

    def architecture_signature(self) -> Dict[str, Any]:
        local = self.local_blocks[0] if self.local_num_layers > 0 else None
        signature = {
            "latent_size": self.latent_size,
            "model_size": self.model_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "patch_size": self.patch_size,
            "global_positions": "codec_frame_centers",
            "refinement_attention": "full_sequence" if local is not None else "disabled",
            "refinement_positions": "codec_frame_rope" if local is not None else "disabled",
            "local_num_layers": self.local_num_layers,
            "local_model_size": self.local_model_size,
            "local_num_heads": local.attention.num_heads if local is not None else 0,
            "local_intermediate_size": local.mlp.w1.out_features if local is not None else 0,
            "use_speaker_conditioning": self.use_speaker_conditioning,
        }

        if self.hourglass_enabled:
            signature.update({
                "hourglass": "full_resolution_coarse_full_resolution",
                "hourglass_pre_layers": self.hourglass_pre_layers,
                "hourglass_middle_layers": self.middle_num_layers,
                "hourglass_post_layers": self.hourglass_post_layers,
                "hourglass_skip": "pre_downsample_residual",
            })

        return signature

    def encode_text(self, text_input_ids, text_mask):
        return self.text_norm(self.text_encoder(text_input_ids, text_mask))

    def project_reference(self, reference_latent):
        return self.reference_proj(reference_latent)

    def get_kv_cache(self, context):
        if self.hourglass_enabled:
            return {
                "pre": [block.attention.get_kv_cache_text(context) for block in self.pre_blocks],
                "middle": [block.attention.get_kv_cache_text(context) for block in self.blocks],
                "post": [block.attention.get_kv_cache_text(context) for block in self.post_blocks],
            }

        return [block.attention.get_kv_cache_text(context) for block in self.blocks]

    def _validate_denoise_inputs(
        self,
        x,
        start_pos,
        latent_mask,
        valid_audio_mask,
        span_mask,
        cond,
    ):
        if x.ndim != 3 or x.shape[-1] != self.latent_size or x.shape[1] < 1:
            raise ValueError(
                f"Expected nonempty [B, T, {self.latent_size}] latents, "
                f"got {tuple(x.shape)}"
            )

        if start_pos < 0:
            raise ValueError("start_pos must be a nonnegative codec-frame offset")

        for name, mask in (
            ("latent_mask", latent_mask),
            ("valid_audio_mask", valid_audio_mask),
            ("span_mask", span_mask),
        ):
            if mask is not None and mask.shape != x.shape[:2]:
                raise ValueError(
                    f"{name} must have shape {tuple(x.shape[:2])}, "
                    f"got {tuple(mask.shape)}"
                )

        if cond is not None and cond.shape != x.shape:
            raise ValueError("cond must have the same [B, T, latent_size] shape as x")

    @staticmethod
    def _frame_mask(x, latent_mask, valid_audio_mask):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        if latent_mask is not None:
            mask = mask & latent_mask.to(device=x.device, dtype=torch.bool)

        if valid_audio_mask is not None:
            mask = mask & valid_audio_mask.to(device=x.device, dtype=torch.bool)

        return mask

    def _project_denoiser_input(
        self, x, frame_mask, cond, span_mask, audio_cond_drop
    ):
        x = self.in_proj(x.masked_fill(~frame_mask[:, :, None], 0.0))

        if cond is not None:
            conditioned = cond.to(x.dtype).masked_fill(
                ~frame_mask[:, :, None], 0.0
            )

            if audio_cond_drop is not None:
                conditioned = conditioned.masked_fill(
                    audio_cond_drop[:, None, None], 0.0
                )

            x = x + self.cond_proj(conditioned)

        if span_mask is not None:
            x = x + self.span_mask_proj(span_mask[:, :, None].to(x.dtype))

        return x.masked_fill(~frame_mask[:, :, None], 0.0)

    def _speaker_film_parameters(
        self, x, speaker_emb, speaker_cond_drop, speaker_adaln_scale
    ):
        if (
            not self.use_speaker_conditioning
            or self.speaker_film_adapter is None
            or speaker_emb is None
        ):
            return None

        speaker = speaker_emb.to(x.dtype)

        if speaker_cond_drop is not None:
            speaker = speaker.masked_fill(
                speaker_cond_drop.to(torch.bool)[:, None], 0.0
            )

        film = self.speaker_film_adapter(speaker)

        if speaker_adaln_scale != 1.0:
            film = film * float(speaker_adaln_scale)

        return film

    def _run_transformer_blocks(
        self,
        x,
        *,
        blocks,
        cached_keys_and_values,
        freqs,
        latent_mask,
        film_offset,
        run_state: _DenoiseRunState,
    ):
        for index, block in enumerate(blocks):
            layer_film = (
                run_state.speaker_film[:, film_offset + index]
                if run_state.speaker_film is not None
                else None
            )

            block_kwargs = dict(
                cond_embed=run_state.cond_embed,
                context_key_mask=run_state.context_mask,
                freqs_cis=freqs,
                kv_cache_text=cached_keys_and_values[index],
                start_pos=0,
                latent_mask=latent_mask,
                context_bias=run_state.context_bias,
                speaker_film=layer_film,
            )

            if (
                run_state.use_checkpoint
                and self.training
                and index % self.checkpoint_every_n == 0
            ):
                x = cp.checkpoint(block, x, use_reentrant=False, **block_kwargs)
            else:
                x = block(x, **block_kwargs)

            x = x.masked_fill(~latent_mask[:, :, None], 0.0)

            if run_state.hidden_states is not None:
                run_state.hidden_states.append(x)

        return x

    def _merge_patches(self, x, frame_mask):
        if self.patch_size == 1:
            return x, frame_mask

        batch_size, frame_count, _ = x.shape
        patch_count = (frame_count + self.patch_size - 1) // self.patch_size
        padding = patch_count * self.patch_size - frame_count

        if padding:
            x = F.pad(x, (0, 0, 0, padding))

        x = self.patch_merge(
            x.reshape(batch_size, patch_count, self.patch_size * self.model_size)
        )

        padded_mask = (
            F.pad(frame_mask, (0, padding), value=False)
            if padding else frame_mask
        )
        patch_mask = padded_mask.reshape(
            batch_size, patch_count, self.patch_size
        ).any(dim=-1)

        return x, patch_mask

    def _expand_patches(self, x, original_length):
        if self.patch_size > 1:
            batch_size, patch_count, _ = x.shape
            x = self.patch_expand(x).reshape(
                batch_size, patch_count * self.patch_size, self.model_size
            )

        return x[:, :original_length]

    def _run_local_refinement(
        self, x, residual, timestep_embed, frame_mask, start_pos, use_checkpoint
    ):
        hidden = self.local_in_proj(x + residual)
        hidden = hidden.masked_fill(~frame_mask[:, :, None], 0.0)

        local_cond = self.local_cond_module(timestep_embed)[:, None]
        local_freqs = precompute_freqs_cis(
            self.local_blocks[0].attention.head_dim,
            hidden.shape[1],
            start=start_pos,
            device=hidden.device,
        )

        for index, block in enumerate(self.local_blocks):
            if use_checkpoint and self.training and index % self.checkpoint_every_n == 0:
                hidden = cp.checkpoint(
                    block,
                    hidden,
                    local_cond,
                    local_freqs,
                    latent_mask=frame_mask,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden, local_cond, local_freqs, latent_mask=frame_mask
                )

        return self.local_out_norm(hidden)

    def denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_mask: torch.Tensor | None,
        kv_cache: Any,
        start_pos: int | None = None,
        use_checkpoint: bool = False,
        latent_mask: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
        context_bias: torch.Tensor | None = None,
        valid_audio_mask: torch.Tensor | None = None,
        cond: Optional[torch.Tensor] = None,
        audio_cond_drop: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_cond_drop: Optional[torch.Tensor] = None,
        speaker_adaln_scale: float = 1.0,
    ):
        start_pos = 0 if start_pos is None else start_pos
        self._validate_denoise_inputs(
            x, start_pos, latent_mask, valid_audio_mask, span_mask, cond
        )

        original_length = x.shape[1]
        frame_mask = self._frame_mask(x, latent_mask, valid_audio_mask)

        if span_mask is not None:
            span_mask = span_mask.to(device=x.device, dtype=torch.bool)

        if audio_cond_drop is not None:
            audio_cond_drop = audio_cond_drop.to(device=x.device, dtype=torch.bool)

        patch_count = (original_length + self.patch_size - 1) // self.patch_size
        global_freqs = precompute_freqs_cis(
            self.head_dim,
            patch_count,
            start=start_pos + (self.patch_size - 1) / 2,
            step=self.patch_size,
            device=x.device,
        )
        frame_freqs = (
            precompute_freqs_cis(
                self.head_dim, original_length, start=start_pos, device=x.device
            )
            if self.hourglass_enabled
            else None
        )

        timestep_embed = get_timestep_embedding(t, self.timestep_embed_size)
        cond_embed = self.cond_module(timestep_embed)[:, None]

        x = self._project_denoiser_input(
            x, frame_mask, cond, span_mask, audio_cond_drop
        )
        speaker_film = self._speaker_film_parameters(
            x, speaker_emb, speaker_cond_drop, speaker_adaln_scale
        )
        hidden_states = [] if return_hidden_states else None
        run_state = _DenoiseRunState(
            cond_embed=cond_embed,
            context_mask=context_mask,
            context_bias=context_bias,
            speaker_film=speaker_film,
            use_checkpoint=use_checkpoint,
            hidden_states=hidden_states,
        )

        if self.hourglass_enabled:
            x = self._run_transformer_blocks(
                x,
                blocks=self.pre_blocks,
                cached_keys_and_values=kv_cache["pre"],
                freqs=frame_freqs,
                latent_mask=frame_mask,
                film_offset=0,
                run_state=run_state,
            )
            residual = x
        else:
            residual = x if self.local_num_layers > 0 else None

        x, patch_mask = self._merge_patches(x, frame_mask)
        x = x.masked_fill(~patch_mask[:, :, None], 0.0)

        middle_cache = kv_cache["middle"] if self.hourglass_enabled else kv_cache
        film_offset = self.hourglass_pre_layers if self.hourglass_enabled else 0

        x = self._run_transformer_blocks(
            x,
            blocks=self.blocks,
            cached_keys_and_values=middle_cache,
            freqs=global_freqs,
            latent_mask=patch_mask,
            film_offset=film_offset,
            run_state=run_state,
        )
        x = self._expand_patches(x, original_length)

        if self.hourglass_enabled:
            x = (x + residual).masked_fill(~frame_mask[:, :, None], 0.0)

            x = self._run_transformer_blocks(
                x,
                blocks=self.post_blocks,
                cached_keys_and_values=kv_cache["post"],
                freqs=frame_freqs,
                latent_mask=frame_mask,
                film_offset=self.hourglass_pre_layers + self.middle_num_layers,
                run_state=run_state,
            )
            x = self.out_norm(x)

        elif self.local_num_layers > 0:
            x = self._run_local_refinement(
                x, residual, timestep_embed, frame_mask, start_pos, use_checkpoint
            )

        else:
            x = self.out_norm(x)

        x = self.out_proj(x).masked_fill(~frame_mask[:, :, None], 0.0).float()

        if return_hidden_states:
            return x, hidden_states

        return x

    def _prepare_conditioning_context(
        self,
        text_ids,
        text_mask,
        reference_latent,
        reference_mask,
        text_cond_drop,
        speaker_cond_drop,
    ):
        text_state = self.encode_text(text_ids, text_mask)
        has_reference = reference_latent is not None and reference_mask is not None

        if has_reference:
            reference_mask = reference_mask.to(torch.bool)
            reference_state = self.project_reference(reference_latent)
            context = torch.cat([text_state, reference_state], dim=1)
            context_mask = torch.cat([text_mask, reference_mask], dim=1)
        else:
            context = text_state
            context_mask = text_mask

        conditioning_mask = context_mask.clone()
        text_length = text_mask.shape[1]

        if text_cond_drop is not None and text_cond_drop.any():
            conditioning_mask[text_cond_drop, :text_length] = False

        if has_reference and speaker_cond_drop is not None and speaker_cond_drop.any():
            conditioning_mask[speaker_cond_drop, text_length:] = False

        return context, conditioning_mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        reference_latent: Optional[torch.Tensor],
        reference_mask: Optional[torch.Tensor],
        text_cond_drop: Optional[torch.Tensor] = None,
        speaker_cond_drop: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        use_checkpoint: bool = False,
        latent_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        valid_audio_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        audio_cond_drop: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_adaln_scale: float = 1.0,
    ):
        if text_mask is None:
            text_mask = torch.ones(
                text_ids.shape[:2], dtype=torch.bool, device=text_ids.device
            )
        else:
            text_mask = text_mask.to(torch.bool)

        if latent_mask is not None:
            latent_mask = latent_mask.to(torch.bool)

        if span_mask is not None:
            span_mask = span_mask.to(torch.bool)

        if valid_audio_mask is not None:
            valid_audio_mask = valid_audio_mask.to(torch.bool)

        context, conditioning_mask = self._prepare_conditioning_context(
            text_ids,
            text_mask,
            reference_latent,
            reference_mask,
            text_cond_drop,
            speaker_cond_drop,
        )
        kv_cache = self.get_kv_cache(context)

        return self.denoise(
            x=x,
            t=t,
            context_mask=conditioning_mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
            use_checkpoint=use_checkpoint,
            latent_mask=latent_mask,
            span_mask=span_mask,
            valid_audio_mask=valid_audio_mask,
            cond=cond,
            audio_cond_drop=audio_cond_drop,
            return_hidden_states=return_hidden_states,
            speaker_emb=speaker_emb,
            speaker_cond_drop=speaker_cond_drop,
            speaker_adaln_scale=speaker_adaln_scale,
        )

    def predict_speaker(self, hidden: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        if self.speaker_reg_head is None:
            return None

        if mask.shape[1] != hidden.shape[1]:
            pad = hidden.shape[1] * self.patch_size - mask.shape[1]

            if pad > 0:
                mask = F.pad(mask, (0, pad), value=False)
            mask = mask.reshape(mask.shape[0], hidden.shape[1], self.patch_size).any(dim=-1)
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return self.speaker_reg_head(pooled)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
