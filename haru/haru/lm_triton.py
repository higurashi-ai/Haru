"""Triton kernels for single-frame language model inference and audio sampling.

Prefill uses PyTorch. Each CUDA graph covers the layers between attention operations.
Attention runs outside the graphs so its key and value buffers can grow with the conversation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from .models.haru import HaruModel, LMStream
from .models.meanflow_head import _rms_norm

KV_CHUNK = 2048  # KV cache growth in positions


@triton.jit
def _rnd(x):
    """Round float32 to bfloat16 and back, where the PyTorch module would store a tensor."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


# ---------------------------------------------------------------- linear layers


@triton.jit
def _gemv_kernel(w_ptr, x_ptr, res_ptr, out_ptr, N, K: tl.constexpr, RES: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr):
    """out[n] = w[n] . x (+ res[n]) for w [N, K] bfloat16, accumulated in float32. BK divides K."""
    n = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN)
    m = n < N
    acc = tl.zeros([BN], tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * tl.load(x_ptr + k).to(tl.float32)[None, :], axis=1)
    y = _rnd(acc)
    if RES:
        y += tl.load(res_ptr + n, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + n, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gate_up_kernel(wg_ptr, wu_ptr, x_ptr, out_ptr, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN)
    m = n < N
    gate = tl.zeros([BN], tl.float32)
    up = tl.zeros([BN], tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        offs = n[:, None] * K + k[None, :]
        x = tl.load(x_ptr + k).to(tl.float32)[None, :]
        gate += tl.sum(tl.load(wg_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32) * x, axis=1)
        up += tl.sum(tl.load(wu_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32) * x, axis=1)
    tl.store(out_ptr + n, (_rnd(_silu(_rnd(gate))) * _rnd(up)).to(tl.bfloat16), mask=m)


def _tiles(weight: torch.Tensor, x: torch.Tensor) -> tuple[int, int, int, int]:
    """``N, K`` of ``weight`` and the tile sizes ``BN, BK`` of its GEMV (``BK``: a power of two dividing ``K``)."""
    N, K = weight.shape
    assert x.is_contiguous() and x.numel() == K and x.dtype == weight.dtype == torch.bfloat16
    return N, K, 8, min(512, K & -K)


def gemv(weight: torch.Tensor, x: torch.Tensor, res: torch.Tensor | None = None,
         dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``F.linear(x, weight)`` (``+ res``) for one vector ``x [K]`` -> ``[N]``, rounded to bfloat16."""
    N, K, bn, bk = _tiles(weight, x)
    out = torch.empty(N, device=weight.device, dtype=dtype)
    _gemv_kernel[(triton.cdiv(N, bn),)](weight, x, out if res is None else res, out, N, K=K, RES=res is not None,
                                        BN=bn, BK=bk, num_warps=4)
    return out


def gate_up(gate: torch.Tensor, up: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``F.silu(F.linear(x, gate)) * F.linear(x, up)`` for one vector ``x [K]`` -> ``[N]``."""
    N, K, bn, bk = _tiles(gate, x)
    out = torch.empty(N, device=x.device, dtype=x.dtype)
    _gate_up_kernel[(triton.cdiv(N, bn),)](gate, up, x, out, N, K=K, BN=bn, BK=bk, num_warps=4)
    return out


# ---------------------------------------------------------------- RMSNorm


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, C, BC: tl.constexpr):
    """Qwen3.5 RMSNorm of one vector: x * rsqrt(mean(x^2) + eps) * (1 + w), in float32."""
    c = tl.arange(0, BC)
    m = c < C
    x = tl.load(x_ptr + c, mask=m, other=0.0).to(tl.float32)
    y = x * tl.math.rsqrt(tl.sum(x * x, axis=0) / C + eps) * (1.0 + tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32))
    tl.store(out_ptr + c, y.to(tl.bfloat16), mask=m)


def rmsnorm(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _rmsnorm_kernel[(1,)](x, norm.weight, out, norm.eps, x.numel(), BC=triton.next_power_of_2(x.numel()), num_warps=4)
    return out


# ---------------------------------------------------------------- Gated DeltaNet


@triton.jit
def _conv_kernel(x_ptr, state_ptr, w_ptr, out_ptr, C, KS: tl.constexpr, BC: tl.constexpr):
    """Causal depthwise conv + SiLU of one new column x [C]. state [C, KS - 1] holds the columns before it
    and is shifted in place; w [C, KS]."""
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    m = c < C
    x = tl.load(x_ptr + c, mask=m, other=0.0)
    acc = tl.load(w_ptr + c * KS + (KS - 1), mask=m, other=0.0).to(tl.float32) * x.to(tl.float32)
    for j in tl.static_range(KS - 1):
        s = tl.load(state_ptr + c * (KS - 1) + j, mask=m, other=0.0)
        acc += tl.load(w_ptr + c * KS + j, mask=m, other=0.0).to(tl.float32) * s.to(tl.float32)
        if j > 0:
            tl.store(state_ptr + c * (KS - 1) + j - 1, s, mask=m)
    tl.store(state_ptr + c * (KS - 1) + KS - 2, x, mask=m)
    tl.store(out_ptr + c, _silu(_rnd(acc)).to(tl.bfloat16), mask=m)


@triton.jit
def _delta_rule_kernel(qkv_ptr, z_ptr, b_ptr, a_ptr, a_log_ptr, dt_bias_ptr, norm_w_ptr, state_ptr, out_ptr, eps,
                       HK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
    """One step of the gated delta rule and the gated output norm, one value head per program.

    qkv: queries [HK, DK], keys [HK, DK], values [HV, DV] after the convolution; each key head serves
    HV / HK value heads. state [HV, DK, DV] float32 is updated in place:
    S = decay * S;  S += k (beta * (v - S^T k))^T;  out = RMSNorm(S^T q) * w * silu(z)."""
    hv = tl.program_id(0)
    hk = hv // (HV // HK)
    i = tl.arange(0, DK)
    j = tl.arange(0, DV)
    q = tl.load(qkv_ptr + hk * DK + i).to(tl.float32)
    k = tl.load(qkv_ptr + (HK + hk) * DK + i).to(tl.float32)
    v = tl.load(qkv_ptr + 2 * HK * DK + hv * DV + j).to(tl.float32)
    q = q * tl.math.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (DK ** -0.5)
    k = k * tl.math.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
    beta = _rnd(tl.sigmoid(tl.load(b_ptr + hv).to(tl.float32)))
    a = tl.load(a_ptr + hv).to(tl.float32) + tl.load(dt_bias_ptr + hv).to(tl.float32)
    softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    decay = tl.exp(-tl.exp(tl.load(a_log_ptr + hv).to(tl.float32)) * softplus)
    s_ptr = state_ptr + hv * DK * DV + i[:, None] * DV + j[None, :]
    s = tl.load(s_ptr) * decay
    s += k[:, None] * ((v - tl.sum(s * k[:, None], axis=0)) * beta)[None, :]
    tl.store(s_ptr, s)
    o = _rnd(tl.sum(s * q[:, None], axis=0))
    o = _rnd(o * tl.math.rsqrt(tl.sum(o * o, axis=0) / DV + eps))
    o = _rnd(o * tl.load(norm_w_ptr + j).to(tl.float32))
    tl.store(out_ptr + hv * DV + j, (o * _silu(tl.load(z_ptr + hv * DV + j).to(tl.float32))).to(tl.bfloat16))


def gated_delta(net: torch.nn.Module, h: torch.Tensor, conv_state: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """A ``Qwen3_5GatedDeltaNet`` module up to its output projection, on one normalized vector ``h [H]``.

    ``conv_state [C, kernel - 1]`` and ``state [HV, DK, DV]`` are updated in place."""
    qkv = torch.empty(net.conv_dim, device=h.device, dtype=h.dtype)
    _conv_kernel[(triton.cdiv(net.conv_dim, 1024),)](gemv(net.in_proj_qkv.weight, h), conv_state, net.conv1d.weight,
                                                     qkv, net.conv_dim, KS=net.conv_kernel_size, BC=1024, num_warps=4)
    out = torch.empty(net.value_dim, device=h.device, dtype=h.dtype)
    _delta_rule_kernel[(net.num_v_heads,)](
        qkv, gemv(net.in_proj_z.weight, h), gemv(net.in_proj_b.weight, h), gemv(net.in_proj_a.weight, h), net.A_log,
        net.dt_bias, net.norm.weight, state, out, net.norm.variance_epsilon, HK=net.num_k_heads, HV=net.num_v_heads,
        DK=net.head_k_dim, DV=net.head_v_dim, num_warps=8)
    return out


# ---------------------------------------------------------------- stream


class TritonLMStream(LMStream):
    """Capture graphs on the first prefill. Calls on the same stream must be serialized."""

    def __init__(self, model: HaruModel, guidance_scale: float | None = None):
        super().__init__(model, guidance_scale)
        self._full = [i for i, layer in enumerate(model.lm.layers) if hasattr(layer, "self_attn")]
        self._graphs: list[torch.cuda.CUDAGraph] = []
        self.length = 0

    @torch.no_grad()
    def prefill(self, prompt) -> torch.Tensor:
        hidden = super().prefill(prompt)
        if not self._graphs:
            self._build()
        self._load()
        return hidden

    @torch.no_grad()
    def step(self, token: torch.Tensor, latent: torch.Tensor, user_latent: torch.Tensor) -> torch.Tensor:
        if self.length == self._capacity:
            self._reserve(self._capacity + KV_CHUNK)
        self._token.copy_(token.view(1, 1))
        self._model_latent.copy_(latent.view(1, 1, -1))
        self._user_latent.copy_(user_latent.view(1, 1, -1))
        self._position.fill_(self.length)
        for j, graph in enumerate(self._graphs[:-1]):
            graph.replay()
            if j < len(self._full):
                self._attend(j)
        self.length += 1
        return self._hidden.clone()

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return gemv(self.model.lm.embed_tokens.weight, hidden.contiguous(), dtype=torch.float32)[None]

    def latent(self, hidden: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        self._head_hidden.copy_(hidden)
        self._head_noise.copy_(noise)
        self._graphs[-1].replay()
        return self._head_latent.clone()

    # ------------------------------------------------------------ state

    def _load(self) -> None:
        """Take over the state the prompt left in the PyTorch cache."""
        cache, self.cache = self.cache, None
        self.length = cache.get_seq_length()
        self._keys, self._values, self._capacity = [], [], 0
        for i, layer in enumerate(cache.layers):
            if i in self._conv:
                self._conv[i].copy_(layer.conv_states[0][0, :, -self._conv[i].shape[1]:])
                self._recurrent[i].copy_(layer.recurrent_states[0][0])
            else:
                self._keys.append(layer.keys)
                self._values.append(layer.values)
        self._reserve(self.length + KV_CHUNK)

    def _reserve(self, capacity: int) -> None:
        """Move the keys and values ``[1, heads, n, dim]`` to buffers with room for ``capacity`` positions."""
        for buffers in (self._keys, self._values):
            for j, old in enumerate(buffers):
                buffers[j] = old.new_empty(*old.shape[:2], capacity, old.shape[3])
                buffers[j][:, :, : self.length] = old[:, :, : self.length]
        self._capacity = capacity

    def _attend(self, j: int) -> None:
        n = self.length
        query, key, value = self._qkv[j]
        keys, values = self._keys[j], self._values[j]
        keys[:, :, n], values[:, :, n] = key[:, :, 0], value[:, :, 0]
        out = F.scaled_dot_product_attention(query, keys[:, :, : n + 1], values[:, :, : n + 1])
        self._attended[j].copy_(out.reshape(-1))

    # ------------------------------------------------------------ graphs

    def _build(self) -> None:
        """Allocate the static buffers, compile the kernels with one eager frame, then capture the graphs."""
        model, device = self.model, self.model.device
        cfg, dim = model.lm.config, model.config.latent_dim
        self._token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._model_latent = torch.zeros(1, 1, dim, device=device)
        self._user_latent = torch.zeros(1, 1, dim, device=device)
        self._position = torch.zeros(3, 1, 1, dtype=torch.long, device=device)  # temporal, height, width: all equal
        self._conv, self._recurrent = {}, {}
        for i, layer in enumerate(model.lm.layers):
            if i not in self._full:
                net = layer.linear_attn
                self._conv[i] = torch.zeros(net.conv_dim, net.conv_kernel_size - 1, device=device, dtype=model.dtype)
                self._recurrent[i] = torch.zeros(net.num_v_heads, net.head_k_dim, net.head_v_dim, device=device)
        width = cfg.num_attention_heads * cfg.head_dim
        self._attended = [torch.zeros(width, device=device, dtype=model.dtype) for _ in self._full]
        self._qkv: list = [None] * len(self._full)
        self._gate: list = [None] * len(self._full)
        self._head_hidden = torch.zeros(1, cfg.hidden_size, device=device, dtype=model.dtype)
        self._head_noise = torch.zeros(1, dim, device=device, dtype=model.dtype)
        self._head_bias = self._head_biases()

        stages = [lambda j=j: self._segment(j) for j in range(len(self._full) + 1)] + [self._head]
        for stage in stages:  # the state they overwrite is loaded afterwards
            stage()
        pool = torch.cuda.graph_pool_handle()
        for stage in stages:
            graph = torch.cuda.CUDAGraph()
            # thread_local: other threads of the process may use CUDA during the capture
            with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                stage()
            self._graphs.append(graph)

    def _segment(self, j: int) -> None:
        """The layers from the ``j - 1``-th attention operation (or the embedding) to the ``j``-th (or the end).

        Tensors read after the graph or by a later one are kept on ``self``, so that the pool does not reuse them."""
        model = self.model
        layers = model.lm.layers
        start = self._full[j - 1] if j else 0
        stop = self._full[j] if j < len(self._full) else len(layers)
        if j == 0:
            x = model.embed(self._token, self._model_latent, self._user_latent).view(-1)
            self._rope = model.lm.rotary_emb(x, self._position)
        else:
            attn = layers[start].self_attn
            gated = self._attended[j - 1] * torch.sigmoid(self._gate[j - 1])
            x = self._mlp(layers[start], gemv(attn.o_proj.weight, gated, res=self._x))
            start += 1
        for i in range(start, stop):
            net = layers[i].linear_attn
            out = gated_delta(net, rmsnorm(layers[i].input_layernorm, x), self._conv[i], self._recurrent[i])
            x = self._mlp(layers[i], gemv(net.out_proj.weight, out, res=x))
        self._x = x
        if stop == len(layers):
            self._hidden = rmsnorm(model.lm.norm, x)[None]
            return
        attn = layers[stop].self_attn
        h = rmsnorm(layers[stop].input_layernorm, x)
        query, gate = gemv(attn.q_proj.weight, h).view(1, 1, -1, 2 * attn.head_dim).chunk(2, dim=-1)
        query = attn.q_norm(query).transpose(1, 2)
        key = attn.k_norm(gemv(attn.k_proj.weight, h).view(1, 1, -1, attn.head_dim)).transpose(1, 2)
        value = gemv(attn.v_proj.weight, h).view(1, -1, 1, attn.head_dim)
        query, key = apply_rotary_pos_emb(query, key, *self._rope)
        # The query heads of one key / value head attend as the query positions of that head: no repeated keys.
        self._qkv[j] = (query.view(1, key.shape[1], -1, attn.head_dim), key, value)
        self._gate[j] = gate.reshape(-1)

    @staticmethod
    def _mlp(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        mlp = layer.mlp
        hidden = gate_up(mlp.gate_proj.weight, mlp.up_proj.weight, rmsnorm(layer.post_attention_layernorm, x))
        return gemv(mlp.down_proj.weight, hidden, res=x)

    def _head_biases(self) -> list[torch.Tensor]:
        """Per sampling step, the head's embeddings of the scalars that are fixed for the conversation."""
        head, cfg = self.model.head, self.model.head.config
        guidance = cfg.guidance_scale if self.guidance_scale is None else self.guidance_scale

        def embed(module: torch.nn.Module, value: float) -> torch.Tensor:
            return module(torch.tensor([value], device=self.model.device).to(self.model.dtype))

        interval = 1.0 / cfg.num_steps
        fixed = embed(head.dt_embed, cfg.interval_embed) + embed(head.h_embed, interval)
        fixed = fixed + embed(head.omega_embed, 1.0 - 1.0 / guidance)
        return [embed(head.t_embed, i * interval) + fixed for i in range(cfg.num_steps)]

    def _head(self) -> None:
        """``MeanFlowHead.sample`` and the unscaling, on the static hidden state and noise."""
        head, eps = self.model.head, self.model.head.config.norm_eps
        z, interval = self._head_noise, 1.0 / len(self._head_bias)
        cond = gemv(head.cond_proj.weight, self._head_hidden)
        for bias in self._head_bias:
            c = F.silu(cond + bias[0])
            x = F.linear(z[0], head.latent_proj.weight)
            for layer in head.layers:
                shift, scale, gate = gemv(layer.ada.weight, c).chunk(3)
                h = _rms_norm(x, eps) * layer.norm.weight * (1 + scale) + shift
                ffn = layer.ffn
                x = x + gate * gemv(ffn.down_proj.weight, gate_up(ffn.gate_proj.weight, ffn.up_proj.weight, h))
            shift, scale = gemv(head.final.ada.weight, c).chunk(2)
            z = z + interval * F.linear(_rms_norm(x, eps) * (1 + scale) + shift, head.final.proj.weight)
        self._head_latent = self.model.unscale_latents(z)
