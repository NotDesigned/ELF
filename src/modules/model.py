"""ELF transformer model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002, ZERO_INIT,
    _make_linear,
)


class ELFBlock(nn.Module):
    """ELF Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module] = None,
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                             deterministic=deterministic)
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class SlotDiTBlock(nn.Module):
    """Small time-conditioned bidirectional block over plan slots."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn_time = _make_linear(
            hidden_size, 2 * hidden_size, bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
        )
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp_time = _make_linear(
            hidden_size, 2 * hidden_size, bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
        )
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    @staticmethod
    def _modulate(x: torch.Tensor, time_emb: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        scale, shift = proj(F.silu(time_emb)).unsqueeze(1).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                deterministic: bool = True) -> torch.Tensor:
        h = self._modulate(self.norm1(x), time_emb, self.attn_time)
        x = x + self.attn(h, rope_fn=None, attention_mask=None, deterministic=deterministic)
        h = self._modulate(self.norm2(x), time_emb, self.mlp_time)
        x = x + self.mlp(h, deterministic=deterministic)
        return x


class SlotDiT(nn.Module):
    """Optional slot-level DiT refinement for plan adapter slots."""

    def __init__(self, hidden_size: int, num_heads: int, depth: int,
                 num_plan_tokens: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if depth <= 0:
            raise ValueError("plan_slot_dit_depth must be positive when plan_adapter_type='slot_dit'")
        self.pos_emb = nn.Parameter(torch.empty(1, num_plan_tokens, hidden_size))
        NORMAL_INIT_002(self.pos_emb)
        self.blocks = nn.ModuleList([
            SlotDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop, proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])
        self.out_norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                deterministic: bool = True) -> torch.Tensor:
        x = x + self.pos_emb.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, time_emb=time_emb, deterministic=deterministic)
        return self.out_norm(x)


class ELF(nn.Module):
    """Text ELF Transformer."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
        use_sentence_plan: bool = False,
        sentence_encoder_type: str = "sentence_t5",
        sentence_emb_dim: int = 768,
        num_plan_tokens: int = 8,
        plan_adapter_type: str = "slot_mlp",
        plan_slot_dit_depth: int = 2,
        plan_learned_encoder_norm: bool = True,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sentence_plan = use_sentence_plan
        self.sentence_encoder_type = sentence_encoder_type
        self.sentence_emb_dim = sentence_emb_dim
        self.num_plan_tokens = num_plan_tokens if use_sentence_plan else 0
        self.plan_adapter_type = plan_adapter_type
        self.plan_learned_encoder_norm = plan_learned_encoder_norm

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        if self.use_sentence_plan:
            if self.sentence_encoder_type not in {"sentence_t5", "learned"}:
                raise ValueError("sentence_encoder_type must be 'sentence_t5' or 'learned'")
            if self.plan_adapter_type not in {"slot_mlp", "slot_dit"}:
                raise ValueError("plan_adapter_type must be 'slot_mlp' or 'slot_dit'")
            if self.num_plan_tokens <= 0:
                raise ValueError("num_plan_tokens must be positive when use_sentence_plan=True")
            if self.sentence_emb_dim <= 0:
                raise ValueError("sentence_emb_dim must be positive when use_sentence_plan=True")
            self.plan_tokens = nn.Parameter(torch.empty(1, self.num_plan_tokens, hidden_size))
            self.plan_in = _make_linear(sentence_emb_dim, hidden_size * self.num_plan_tokens, bias=True)
            self.plan_time_embedder = TimestepEmbedder(hidden_size)
            self.plan_norm = RMSNorm(hidden_size, eps=1e-6)
            self.plan_out = _make_linear(hidden_size * self.num_plan_tokens, sentence_emb_dim, bias=True)
            NORMAL_INIT_002(self.plan_tokens)
            if self.plan_adapter_type == "slot_dit":
                self.plan_in_dit = SlotDiT(
                    hidden_size=hidden_size, num_heads=num_heads, depth=plan_slot_dit_depth,
                    num_plan_tokens=self.num_plan_tokens, mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop, proj_drop=proj_drop,
                )
                self.plan_out_input = _make_linear(hidden_size * 2, hidden_size, bias=True)
                self.plan_out_dit = SlotDiT(
                    hidden_size=hidden_size, num_heads=num_heads, depth=plan_slot_dit_depth,
                    num_plan_tokens=self.num_plan_tokens, mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop, proj_drop=proj_drop,
                )
            else:
                self.plan_in_dit = None
                self.plan_out_input = None
                self.plan_out_dit = None
            if self.sentence_encoder_type == "learned":
                self.plan_encoder_query = nn.Parameter(torch.empty(1, sentence_emb_dim))
                NORMAL_INIT_002(self.plan_encoder_query)
                self.plan_encoder_output_norm = (
                    nn.RMSNorm(sentence_emb_dim, elementwise_affine=False)
                    if self.plan_learned_encoder_norm else nn.Identity()
                )
            else:
                self.register_parameter("plan_encoder_query", None)
                self.plan_encoder_output_norm = nn.Identity()

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        if self.use_sentence_plan:
            prefix_total += self.num_plan_tokens
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total,
        )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        # Final flow-matching output head.
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def build_context(self, t: torch.Tensor,
                      self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix_tokens.append(
            self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        )

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return prefix_tokens

    def build_plan_tokens(self, plan_z: torch.Tensor, plan_t: torch.Tensor,
                          deterministic: bool = True) -> Tuple[torch.Tensor, dict]:
        """Project a sentence latent into learnable in-context plan slots."""
        if plan_z.shape[-1] != self.sentence_emb_dim:
            raise ValueError(
                f"plan_z dim {plan_z.shape[-1]} does not match sentence_emb_dim={self.sentence_emb_dim}"
            )
        B = plan_z.shape[0]
        plan_hidden = self.plan_in(plan_z.float()).reshape(B, self.num_plan_tokens, self.hidden_size)
        plan_time = self.plan_time_embedder(plan_t)
        plan_hidden = plan_hidden + self.plan_tokens.expand(B, -1, -1) + plan_time.unsqueeze(1)
        if self.plan_adapter_type == "slot_dit":
            plan_hidden = self.plan_in_dit(plan_hidden, time_emb=plan_time, deterministic=deterministic)
            return plan_hidden, {"raw_plan_slots": plan_hidden, "time_emb": plan_time}
        return plan_hidden, {}

    def predict_plan(self, plan_hidden: torch.Tensor, plan_context: dict,
                     deterministic: bool = True,
                     learned_plan_encode: bool = False) -> torch.Tensor:
        """Read processed plan slots back into sentence latent space."""
        if self.plan_adapter_type == "slot_dit":
            raw_plan_slots = plan_context["raw_plan_slots"]
            time_emb = plan_context["time_emb"]
            plan_hidden = torch.cat([raw_plan_slots.float(), plan_hidden.float()], dim=-1)
            plan_hidden = self.plan_out_input(plan_hidden)
            plan_hidden = self.plan_out_dit(plan_hidden, time_emb=time_emb.float(), deterministic=deterministic)
        plan_f32 = self.plan_norm(plan_hidden.float()).reshape(plan_hidden.shape[0], -1)
        plan_pred = self.plan_out(plan_f32)
        if learned_plan_encode and self.sentence_encoder_type == "learned":
            plan_pred = self.plan_encoder_output_norm(plan_pred)
        return plan_pred

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        plan_z: Optional[torch.Tensor] = None,
        plan_t: Optional[torch.Tensor] = None,
        return_plan: bool = False,
        learned_plan_encode: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid."""
        B = x.shape[0]

        if learned_plan_encode:
            if not self.use_sentence_plan or self.sentence_encoder_type != "learned":
                raise ValueError("learned_plan_encode requires use_sentence_plan=True and sentence_encoder_type='learned'")
            plan_z = self.plan_encoder_query.expand(B, -1)
            return_plan = True

        if self.use_sentence_plan:
            if plan_z is None:
                raise ValueError("plan_z is required when use_sentence_plan=True")
            if plan_t is None:
                plan_t = t
        elif plan_z is not None:
            raise ValueError("plan_z was provided but use_sentence_plan=False")

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast('cuda', enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            x = self.text_proj(x.float())
            context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
            if self.use_sentence_plan:
                plan_tokens, plan_context = self.build_plan_tokens(
                    plan_z, plan_t, deterministic=deterministic,
                )
            else:
                plan_tokens, plan_context = None, None

        # Prepend learnable model-mode tokens (gated by decoder_step_active),
        # optional sentence plan slots, and context prefix tokens.
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        plan_offset = 0
        sequence_parts = []
        attention_parts = []
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            sequence_parts.append(mode_tokens)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((B, self.num_model_mode_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_parts.append(mode_mask)

        if plan_tokens is not None:
            sequence_parts.append(plan_tokens)
            plan_offset = self.num_plan_tokens
            if attention_mask is not None:
                plan_mask = torch.ones((B, self.num_plan_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_parts.append(plan_mask)

        sequence_parts.append(x)
        if attention_mask is not None:
            attention_parts.append(attention_mask)
        x = torch.cat(sequence_parts, dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(attention_parts, dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((B, prefix_len),
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpoint:
                def _block_forward(hidden: torch.Tensor, block: ELFBlock = block) -> torch.Tensor:
                    return block(hidden, rope_fn=self.feat_rope, attention_mask=attention_mask,
                                 deterministic=deterministic)

                x = checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask,
                          deterministic=deterministic)

        plan_hidden = None
        field_start = prefix_len + model_mode_offset + plan_offset
        if self.use_sentence_plan and return_plan:
            plan_start = prefix_len + model_mode_offset
            plan_hidden = x[:, plan_start:field_start]
        x = x[:, field_start:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        with torch.amp.autocast('cuda', enabled=False):
            decoder_logits = None
            if decoder_step_active is not None:
                x_f32 = x.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            output = self.final_layer(x.float())
            plan_pred = None
            if plan_hidden is not None:
                plan_pred = self.predict_plan(
                    plan_hidden, plan_context,
                    deterministic=deterministic,
                    learned_plan_encode=learned_plan_encode,
                )
        if return_plan:
            return output, decoder_logits, plan_pred
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
