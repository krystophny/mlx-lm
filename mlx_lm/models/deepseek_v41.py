# Copyright © 2026 Apple Inc.

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs
from .pipeline import PipelineMixin
from .rope_utils import YarnRoPE
from .switch_layers import SwitchGLU, SwitchLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20

    sliding_window: int = 128
    compress_ratios: tuple[int, ...] = ()
    kv_source_layer_ids: tuple[int, ...] = ()
    index_source_layer_ids: tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    rope_theta: float = 10000.0
    rope_scaling: Optional[dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 1048576
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # Engram: hashed n-gram embeddings added to the residual streams before
    # the layers in engram_layer_ids.
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelArgs":
        if "text_config" in params:
            params = {**params["text_config"], "model_type": params["model_type"]}
        return super().from_dict(params)

    def __post_init__(self) -> None:
        if not self.engram_layer_ids:
            return
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError(
                f"engram_num_embeddings has {len(self.engram_num_embeddings)} "
                f"entries for {len(self.engram_layer_ids)} engram layers."
            )
        if (
            self.engram_max_ngram_size < 2
            or self.engram_n_heads < 1
            or self.engram_head_dim < 1
            or self.engram_vocab_size < 1
            or self.engram_compressed_vocab_size < 1
        ):
            raise ValueError(
                "Engram layers need engram_max_ngram_size >= 2 and positive "
                "engram_n_heads, engram_head_dim, engram_vocab_size and "
                "engram_compressed_vocab_size."
            )


def _apply_rope(
    x: mx.array,
    rope: YarnRoPE,
    *,
    rope_dim: int,
    offset: int = 0,
    scale: float = 1.0,
    inverse: bool = False,
) -> mx.array:
    """Apply interleaved RoPE to the trailing feature slice."""
    if rope_dim == 0:
        return x
    head = x[..., :-rope_dim]
    tail = x[..., -rope_dim:].astype(mx.float32)
    if x.ndim == 4:
        tail = tail.transpose(0, 2, 1, 3)
        tail = rope(tail, offset=offset, scale=scale, inverse=inverse)
        tail = tail.transpose(0, 2, 1, 3)
    else:
        tail = rope(tail, offset=offset, scale=scale, inverse=inverse)
    return mx.concatenate([head, tail.astype(x.dtype)], axis=-1)


# The reference also rounds fp8/fp4 matmul inputs to fp8; skipping it is more precise.
def _fake_quant_fp8(x: mx.array, block_size: int = 32) -> mx.array:
    """Round-trip FP8 e4m3 with power-of-two block scales."""
    if x.shape[-1] % block_size:
        return x
    dtype = x.dtype
    shape = x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), 1e-4)
    exponent = mx.ceil(mx.log2(amax / 448.0))
    scale = mx.power(2.0, exponent)
    quantized = mx.clip(blocks / scale, -448.0, 448.0)
    return (
        (mx.from_fp8(mx.to_fp8(quantized), mx.float32) * scale)
        .reshape(shape)
        .astype(dtype)
    )


_FP4_LUT = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)


def _round_fp4(x: mx.array) -> mx.array:
    mag = mx.abs(x)
    idx = mx.zeros(mag.shape, dtype=mx.int32)
    for threshold in (0.25, 1.25, 2.5, 5.0):
        idx = idx + (mag > threshold).astype(mx.int32)
    for threshold in (0.75, 1.75, 3.5):
        idx = idx + (mag >= threshold).astype(mx.int32)
    return mx.sign(x) * _FP4_LUT[idx]


def _fake_quant_fp4(
    x: mx.array, *, block_size: int, e4m3_scale: bool = False
) -> mx.array:
    if x.shape[-1] % block_size:
        return x
    dtype = x.dtype
    shape = x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    floor = 6.0 * 2.0**-9 if e4m3_scale else 6.0 * 2.0**-126
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), floor)
    if e4m3_scale:
        scale = mx.from_fp8(mx.to_fp8(amax / 6.0), mx.float32)
    else:
        scale = mx.power(2.0, mx.ceil(mx.log2(amax / 6.0)))
    q = _round_fp4(mx.clip(blocks / scale, -6.0, 6.0)) * scale
    return q.reshape(shape).astype(dtype)


def _make_sinkhorn_kernel() -> Optional[Callable]:
    if not mx.metal.is_available():
        return None
    # The loop of _sinkhorn in one launch, one thread per matrix, in the same order.
    source = """
        uint n = thread_position_in_grid.x;
        if (n >= comb_shape[0]) {
            return;
        }
        float m[HC * HC];
        for (int k = 0; k < HC * HC; k++) {
            m[k] = comb[n * HC * HC + k];
        }
        float e = eps[0];
        for (int t = 0; t < 2 * ITERS - 1; t++) {
            bool rows = t % 2 == 1;
            for (int a = 0; a < HC; a++) {
                float s = 0.0f;
                for (int b = 0; b < HC; b++) {
                    s += rows ? m[a * HC + b] : m[b * HC + a];
                }
                float d = s + e;
                for (int b = 0; b < HC; b++) {
                    int k = rows ? a * HC + b : b * HC + a;
                    m[k] = m[k] / d;
                }
            }
        }
        for (int k = 0; k < HC * HC; k++) {
            out[n * HC * HC + k] = m[k];
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v41_sinkhorn",
        input_names=["comb", "eps"],
        output_names=["out"],
        source=source,
    )


_sinkhorn_kernel = _make_sinkhorn_kernel()


def _sinkhorn(comb: mx.array, *, iters: int, eps: mx.array) -> mx.array:
    """Normalize the columns, then the rows and columns iters - 1 more times."""
    if (
        _sinkhorn_kernel is None
        or mx.default_device() != mx.gpu
        or iters < 1
        or comb.size == 0
    ):
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        for _ in range(iters - 1):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        return comb
    hc = comb.shape[-1]
    flat = comb.reshape(-1, hc, hc)
    (out,) = _sinkhorn_kernel(
        inputs=[flat, eps],
        template=[("HC", hc), ("ITERS", iters)],
        grid=(flat.shape[0], 1, 1),
        threadgroup=(min(flat.shape[0], 256), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[mx.float32],
    )
    return out.reshape(comb.shape)


class HyperConnection(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_mult
        self.iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.rms_norm_eps
        # The leading underscores keep these out of the parameters.
        self._eps = mx.array([args.hc_eps], dtype=mx.float32)
        # The scale of each mix column: pre, post, then comb.
        self._scale_ids = mx.array([0] * self.hc + [1] * self.hc + [2] * self.hc**2)
        mix = (2 + self.hc) * self.hc
        self.fn = mx.zeros((mix, self.hc * args.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, streams: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # The reference normalizes in float32.
        flat = streams.reshape(*streams.shape[:2], -1).astype(mx.float32)
        flat = mx.fast.rms_norm(flat, weight=None, eps=self.norm_eps)
        mix = (flat @ self.fn.T) * self.scale[self._scale_ids] + self.base
        pre, post, comb = mx.split(mix, [self.hc, 2 * self.hc], axis=-1)
        pre = mx.sigmoid(pre) + self.hc_eps
        post = 2.0 * mx.sigmoid(post)
        comb = comb.reshape(*comb.shape[:-1], self.hc, self.hc)
        comb = mx.softmax(comb, axis=-1) + self.hc_eps
        return pre, post, _sinkhorn(comb, iters=self.iters, eps=self._eps)


def _collapse(streams: mx.array, mix: mx.array) -> mx.array:
    out = mx.sum(mix[..., None] * streams.astype(mx.float32), axis=2)
    return out.astype(streams.dtype)


def _expand(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    mixed = mx.sum(comb[..., None] * residual[..., :, None, :], axis=2)
    out = post[..., None] * x[..., None, :] + mixed
    return out.astype(residual.dtype)


def _append(cached: Optional[mx.array], new: mx.array) -> mx.array:
    if cached is None:
        return new
    return mx.concatenate([cached, new], axis=1)


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for every n below 3.3e24."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_primes(args: ModelArgs) -> np.ndarray:
    """The bucket modulus of every (engram layer, n-gram size, head).

    Each head of each n-gram size owns its own range of the layer's table. The
    primes are drawn in order, each the smallest above engram_vocab_size - 1
    that no earlier pair took, so the ranges never share a modulus.
    """
    primes, seen = [], set()
    for _ in args.engram_layer_ids:
        per_ngram = []
        for _ in range(args.engram_max_ngram_size - 1):
            sizes, current = [], args.engram_vocab_size - 1
            for _ in range(args.engram_n_heads):
                current += 1
                while current in seen or not _is_prime(current):
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return np.array(primes, dtype=np.int64)


def engram_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, vocab_size: int
) -> np.ndarray:
    """One odd multiplier per (engram layer, lookback), bounded so that a
    compressed token id times its multiplier cannot overflow int64."""
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


def build_compressed_token_map(tokenizer) -> tuple[np.ndarray, int]:
    """Map every token id to a compressed id; tokens that normalize alike
    (case, accents, runs of whitespace) share one. Returns the map and the
    compressed vocabulary size, from which the hash multipliers derive.

    Accepts a Hugging Face fast tokenizer or an mlx_lm TokenizerWrapper.
    """
    from tokenizers import Regex, normalizers

    if not hasattr(tokenizer, "backend_tokenizer"):
        # An mlx_lm TokenizerWrapper keeps the Hugging Face tokenizer here.
        tokenizer = tokenizer._tokenizer
    # A private-use character keeps a token that is exactly one space from
    # stripping to the empty string.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = np.zeros(len(tokenizer), dtype=np.int64)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # A partial UTF-8 byte token is keyed by its raw form.
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup[token_id] = keys.setdefault(key, len(keys))
    return lookup, len(keys)


class EngramHasher:
    """The hash ids of the n-grams that end at each position, in int64 NumPy.

    A position hashes itself with the engram_max_ngram_size - 1 compressed ids
    before it; lookback stops at the start of the request. The caller keeps
    the last engram_max_ngram_size - 1 compressed ids as request history, so
    chunked prefill and decode give the same ids as one full pass.
    """

    DEAD = -1

    def __init__(self, args: ModelArgs, token_map: np.ndarray, vocab_size: int):
        if vocab_size != args.engram_compressed_vocab_size:
            raise ValueError(
                f"The tokenizer compresses to {vocab_size} engram ids, but the "
                f"config declares {args.engram_compressed_vocab_size}; this "
                "tokenizer does not belong to the checkpoint."
            )
        self.depth = args.engram_max_ngram_size - 1
        self.token_map = np.asarray(token_map, dtype=np.int64)
        self.pad_id = int(self.token_map[args.engram_pad_token_id])
        self.primes = engram_primes(args)
        flat = self.primes.reshape(len(args.engram_layer_ids), -1)
        rows = [int(r) for r in flat.sum(-1)]
        if rows != [int(r) for r in args.engram_num_embeddings]:
            raise ValueError(
                f"The engram prime layout needs {rows} table rows, but the "
                f"config declares {list(args.engram_num_embeddings)}."
            )
        self.offsets = np.cumsum(flat, axis=-1) - flat
        self.multipliers = engram_multipliers(
            tuple(args.engram_layer_ids), args.engram_max_ngram_size, vocab_size
        )

    @classmethod
    def from_tokenizer(cls, args: ModelArgs, tokenizer) -> "EngramHasher":
        return cls(args, *build_compressed_token_map(tokenizer))

    def __call__(
        self, ids: np.ndarray, history: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the hash ids [B, L, n_engram_layers, n_hash_cols] and the
        new history [B, engram_max_ngram_size - 1]."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= len(self.token_map)):
            raise ValueError("A token id lies outside the engram token map.")
        batch, length = ids.shape
        if history is None:
            history = np.full((batch, self.depth), self.DEAD, dtype=np.int64)
        elif history.shape != (batch, self.depth):
            raise ValueError(
                f"The engram history has shape {history.shape}, expected "
                f"{(batch, self.depth)}."
            )
        joined = np.concatenate([history, self.token_map[ids]], axis=1)
        blocked = np.zeros((batch, length), dtype=bool)
        lookback = []
        for shift in range(self.depth + 1):
            source = joined[:, self.depth - shift : self.depth - shift + length]
            blocked |= source == self.DEAD
            lookback.append(np.where(blocked, self.pad_id, source))
        # [B, L, n_engram_layers, max_ngram_size]
        products = np.stack(lookback, axis=-1)[:, :, None, :] * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.depth + 1):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1])
        hashes = np.concatenate(hashes, axis=-1) + self.offsets
        return hashes, joined[:, -self.depth :]


class Engram(nn.Module):
    """Adds a gated n-gram lookup to the hyper-connection streams.

    The hash ids fetch n_hash_cols table rows; wkv maps them to one key per
    stream and a value shared by all streams. The gate is a normalized dot
    product of each stream with its key.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.table_index = list(args.engram_layer_ids).index(layer_id)
        self.hc = args.hc_mult
        self.dim = args.hidden_size
        self.eps = args.rms_norm_eps
        rows = args.engram_num_embeddings[self.table_index]
        cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.embed = nn.Embedding(rows, args.engram_head_dim)
        self.wkv = nn.Linear(
            cols * args.engram_head_dim, self.dim * (self.hc + 1), bias=False
        )
        self.q_weight = mx.ones((self.hc, self.dim), dtype=mx.float32)
        self.k_weight = mx.ones((self.hc, self.dim), dtype=mx.float32)
        # Tensor parallel row sharding: the first local row and the group.
        self._row_start = None
        self._group = None

    def _shard_rows(self, rank: int, size: int) -> None:
        """Keep rows [rank * part, (rank + 1) * part) of the table, with
        part = ceil(rows / size)."""
        rows = self.embed.weight.shape[0]
        part = -(-rows // size)
        start = rank * part
        embed = self.embed
        for name in ("weight", "scales", "biases"):
            value = embed.get(name)
            if value is not None:
                setattr(embed, name, mx.contiguous(value[start : start + part]))
        if hasattr(embed, "num_embeddings"):
            embed.num_embeddings = embed.weight.shape[0]
        self._row_start = start

    def shard(self, group: mx.distributed.Group) -> None:
        self._shard_rows(group.rank(), group.size())
        self._group = group

    def lookup(self, ids: mx.array) -> mx.array:
        if self._row_start is None:
            return self.embed(ids)
        local = ids - self._row_start
        outside = (local < 0) | (local >= self.embed.weight.shape[0])
        values = self.embed(mx.where(outside, 0, local))
        values = mx.where(outside[..., None], mx.zeros_like(values), values)
        if self._group is not None and self._group.size() > 1:
            values = mx.distributed.all_sum(values, group=self._group)
        return values

    def __call__(self, streams: mx.array, ids: mx.array) -> mx.array:
        """streams: [B, L, hc, dim]; ids: [B, L, n_hash_cols]."""
        batch, length = ids.shape[:2]
        kv = self.wkv(self.lookup(ids).reshape(batch, length, -1))
        key, value = mx.split(kv, [self.hc * self.dim], axis=-1)
        key = key.astype(mx.float32).reshape(batch, length, self.hc, self.dim)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = streams.astype(mx.float32)
        # Normalized per (token, stream) over dim, not jointly over the streams.
        rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(key * key, axis=-1) + self.eps
        )
        dot = mx.sum(h * weight * key, axis=-1) * rstd * self.dim**-0.5
        # copysign(sqrt(max(|dot|, 1e-6)), dot), keeping the sign of -0.0.
        magnitude = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        negative = (dot < 0) | ((dot == 0) & (mx.reciprocal(dot) < 0))
        gate = mx.sigmoid(mx.where(negative, -magnitude, magnitude))
        value = value.astype(mx.float32)[:, :, None, :]
        return (h + gate[..., None] * value).astype(streams.dtype)


class LayerCache:
    """Cache for one decoder layer: the sliding window, the closed compressed
    groups and their index keys, and the rows of the group that is still open.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.offset = 0
        self.window = None
        self.compress_kv = None
        self.index_k = None
        self.pending_kv = None
        self.pending_gate = None
        # The last engram_max_ngram_size - 1 compressed token ids of the
        # request; only the first cache of a pipeline stage keeps it.
        self.engram_history = None

    def update_window(self, kv: mx.array) -> mx.array:
        """Return the window with kv appended, and keep its last rows."""
        window = _append(self.window, kv)
        self.window = window[:, -self.window_size :]
        self.offset += kv.shape[1]
        return window

    @property
    def state(self) -> tuple[Optional[mx.array], ...]:
        return (
            self.window,
            self.compress_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
            mx.array(self.offset),
        )

    def is_trimmable(self) -> bool:
        # Evicted window rows and closed compressed groups cannot be restored.
        return False

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        arrays = (
            self.window,
            self.compress_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
        )
        return sum(a.nbytes for a in arrays if a is not None)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        if self.ratio > 1:
            self.gate_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self, x: mx.array, start_pos: int, cache: LayerCache
    ) -> tuple[Optional[mx.array], int]:
        """Pool each full group of ratio tokens into one latent. Return the
        latents, or None if no group is full, and the index of the first one."""
        if self.ratio == 1:
            return self.kv_norm(self.kv_proj(x)), start_pos

        h = x.astype(mx.float32)
        pending = 0 if cache.pending_kv is None else cache.pending_kv.shape[1]
        kv = _append(cache.pending_kv, self.kv_proj(h))
        gate = _append(cache.pending_gate, self.gate_proj(h))
        full = kv.shape[1] - kv.shape[1] % self.ratio
        cache.pending_kv = kv[:, full:]
        cache.pending_gate = gate[:, full:]
        # A group takes the position of its first token, and the RoPE scale
        # multiplies the offset by the ratio, so return the group index.
        first = (start_pos - pending) // self.ratio
        if full == 0:
            return None, first
        batch, _, dim = kv.shape
        kv = kv[:, :full].reshape(batch, -1, self.ratio, dim)
        gate = gate[:, :full].reshape(batch, -1, self.ratio, dim)
        latent = mx.sum(kv * mx.softmax(gate, axis=2), axis=2)
        return self.kv_norm(latent.astype(x.dtype)), first


def _window_indices(*, previous: int, length: int, window: int) -> mx.array:
    """The window rows each query attends to, -1 for none. The window holds
    the previous rows, then the new ones."""
    width = min(window, previous + length)
    end = previous + mx.arange(length)
    start = mx.maximum(end - window + 1, 0)
    indices = start[:, None] + mx.arange(width)[None, :]
    return mx.where(indices > end[:, None], -1, indices)


def _gather_rows(values: mx.array, indices: mx.array) -> mx.array:
    batch, length, dim = values.shape
    safe = mx.maximum(indices, 0)
    base = (mx.arange(batch, dtype=mx.int32) * length).reshape(batch, 1, 1)
    flat = values.reshape(batch * length, dim)
    return flat[(safe + base).reshape(-1)].reshape(*indices.shape, dim)


def _sparse_attention(
    q: mx.array,
    kvs: list[mx.array],
    indices: list[mx.array],
    *,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Attend each query to the rows its indices pick from each KV array.

    q is [B, S, H, D], each KV array [B, K, D] and its indices [B, S, k].
    An index of -1 picks nothing.
    """
    batch, length, heads, dim = q.shape
    rows = [_gather_rows(kv, idx) for kv, idx in zip(kvs, indices, strict=True)]
    # Each query has its own keys, so each query is one batch entry.
    keys = mx.concatenate(rows, axis=2).reshape(batch * length, 1, -1, dim)
    keys = keys.astype(mx.float32)
    mask = mx.concatenate(indices, axis=-1) >= 0
    # SDPA rounds the scores to the input dtype at this head size, so use fp32.
    out = mx.fast.scaled_dot_product_attention(
        q.reshape(batch * length, heads, 1, dim).astype(mx.float32),
        keys,
        keys,
        scale=scale,
        mask=mask.reshape(batch * length, 1, 1, -1),
        sinks=sinks.astype(mx.float32),
    )
    return out.reshape(batch, length, heads, dim).astype(q.dtype)


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.num_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.rope_dim = args.qk_rope_head_dim
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        # Only a layer that compresses its own KV makes index keys.
        if layer_id in args.kv_source_layer_ids:
            self.k_proj = nn.Linear(args.head_dim, args.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(args.index_head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        *,
        index_k: Optional[mx.array],
        candidates: Optional[mx.array],
        rope: YarnRoPE,
        start_pos: int,
    ) -> tuple[mx.array, Optional[mx.array]]:
        """Return the top-k compressed entries of each query, -1 for none, and
        the candidate mask: the one this layer selects, or the given one."""
        batch, length, _ = x.shape
        if index_k is None or index_k.shape[1] == 0:
            return mx.zeros((batch, length, 0), dtype=mx.int32), candidates

        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, rope, rope_dim=self.rope_dim, offset=start_pos)
        q = _fake_quant_fp4(q, block_size=32)
        keys = index_k.astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32) * (self.num_heads**-0.5)
        scores = mx.matmul(q.astype(mx.float32), keys[:, None].swapaxes(-1, -2))
        scores = mx.maximum(scores, 0.0) * (self.head_dim**-0.5)
        scores = mx.sum(scores * weights[..., None], axis=2)
        # A query sees the groups that end at or before it.
        lens = ((start_pos + 1 + mx.arange(length)) // self.ratio)[:, None]
        visible = mx.arange(index_k.shape[1])[None, :] < lens
        scores = mx.where(visible[None], scores, -mx.inf)

        if self.is_candidate_source and self.candidate_topk_blocks > 0:
            block = self.candidate_block_size
            pad = (-scores.shape[-1]) % block
            padded = mx.pad(scores, [(0, 0), (0, 0), (0, pad)], constant_values=-mx.inf)
            block_scores = padded.reshape(batch, length, -1, block).max(axis=-1)
            # Always keep the newest block: it is only partly full, so an older
            # full block can have a higher score.
            last = mx.where(lens > 0, (lens - 1) // block, -1)
            num_blocks = block_scores.shape[-1]
            block_scores = mx.where(mx.arange(num_blocks) == last, mx.inf, block_scores)
            k_blocks = min(self.candidate_topk_blocks, num_blocks)
            chosen = mx.argpartition(-block_scores, k_blocks - 1, axis=-1)[
                ..., :k_blocks
            ]
            # With fewer reachable blocks than k_blocks, the extra picks are -inf.
            reachable = mx.take_along_axis(block_scores, chosen, axis=-1) > -mx.inf
            candidates = mx.zeros(block_scores.shape, dtype=mx.bool_)
            candidates = mx.put_along_axis(candidates, chosen, reachable, axis=-1)
            candidates = mx.repeat(candidates, block, axis=-1)[..., : scores.shape[-1]]
        elif candidates is not None:
            scores = mx.where(candidates, scores, -mx.inf)

        k = min(self.index_topk, index_k.shape[1])
        chosen = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        return mx.where(chosen < lens[None], chosen, -1), candidates


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.o_groups = args.o_groups
        ratios = args.compress_ratios
        self.ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, args.num_attention_heads * args.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        # wo_a is block diagonal: one projection per output group.
        self.o_a_proj = SwitchLinear(
            args.num_attention_heads * args.head_dim // args.o_groups,
            args.o_lora_rank,
            args.o_groups,
            bias=False,
        )
        self.o_b_proj = nn.Linear(
            args.o_groups * args.o_lora_rank, args.hidden_size, bias=False
        )
        self.sinks = mx.zeros((args.num_attention_heads,), dtype=mx.float32)
        self._group_ids = mx.arange(args.o_groups)[None]
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)
        if self.ratio:
            scaling = args.rope_scaling or {}
            original = scaling.get(
                "original_max_position_embeddings", args.max_position_embeddings
            )
            self.rope = YarnRoPE(
                dims=args.qk_rope_head_dim,
                traditional=True,
                base=args.compress_rope_theta,
                scaling_factor=float(scaling.get("factor", 1.0)),
                original_max_position_embeddings=int(original),
                beta_fast=int(scaling.get("beta_fast", 32)),
                beta_slow=int(scaling.get("beta_slow", 1)),
                # V4.1 uses attention_factor=1 for its YaRN RoPE.
                mscale=0.0,
                mscale_all_dim=0.0,
            )
        else:
            # A layer with only the window uses the base theta and no YaRN.
            self.rope = YarnRoPE(
                dims=args.qk_rope_head_dim, traditional=True, base=args.rope_theta
            )

    def __call__(
        self,
        x: mx.array,
        *,
        start_pos: int,
        window_idx: mx.array,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> mx.array:
        batch, length, _ = x.shape
        q_residual = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, self.rope, rope_dim=self.rope_dim, offset=start_pos)

        kv = self.kv_norm(self.kv_proj(x))
        kv = _apply_rope(kv, self.rope, rope_dim=self.rope_dim, offset=start_pos)
        kv = _fake_quant_fp8(kv, 32)
        kvs = [cache.update_window(kv)]
        indices = [window_idx]
        if self.ratio:
            if self.is_kv_source:
                self._compress(x, start_pos=start_pos, cache=cache, shared=shared)
            if self.is_index_source:
                topk, candidates = self.indexer(
                    x,
                    q_residual,
                    index_k=shared.get("index_k"),
                    candidates=shared.get("candidates"),
                    rope=self.rope,
                    start_pos=start_pos,
                )
                shared["topk_idx"] = topk
                if self.indexer.is_candidate_source:
                    shared["candidates"] = candidates
            compressed = shared.get("compress_kv")
            topk = shared.get("topk_idx")
            if compressed is not None and topk is not None and topk.shape[-1] > 0:
                kvs.append(compressed)
                indices.append(topk)

        output = _sparse_attention(
            q, kvs, indices, sinks=self.sinks, scale=self.head_dim**-0.5
        )
        output = _apply_rope(
            output, self.rope, rope_dim=self.rope_dim, offset=start_pos, inverse=True
        )
        output = output.reshape(batch * length, self.o_groups, 1, -1)
        output = self.o_a_proj(output, self._group_ids)
        return self.o_b_proj(output.reshape(batch, length, -1))

    def _compress(
        self,
        x: mx.array,
        *,
        start_pos: int,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> None:
        """Add the new compressed KV and index keys to the cache, and share them."""
        latent, first = self.compressor(x, start_pos, cache)
        if latent is not None:
            kv = _apply_rope(
                latent,
                self.rope,
                rope_dim=self.rope_dim,
                offset=first,
                scale=self.ratio,
            )
            kv = _fake_quant_fp4(kv, block_size=16, e4m3_scale=True)
            cache.compress_kv = _append(cache.compress_kv, kv)
            if self.is_index_source:
                k = self.indexer.k_norm(self.indexer.k_proj(latent))
                k = _apply_rope(
                    k,
                    self.rope,
                    rope_dim=self.rope_dim,
                    offset=first,
                    scale=self.ratio,
                )
                k = _fake_quant_fp4(k, block_size=32)
                cache.index_k = _append(cache.index_k, k)
        shared["compress_kv"] = cache.compress_kv
        if self.is_index_source:
            shared["index_k"] = cache.index_k


class DeepseekV41SwiGLU(nn.Module):
    """DeepSeek-V4.1's clamped SwiGLU activation."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        dtype = up.dtype
        gate = gate.astype(mx.float32)
        up = up.astype(mx.float32)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return swiglu(gate, up).astype(dtype)


class SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.moe_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            args.moe_intermediate_size, args.hidden_size, bias=False
        )
        self.activation = DeepseekV41SwiGLU(args.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation(up, gate).astype(x.dtype))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )
        self.top_k = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.routed_scaling_factor

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        logits = logits / self.gate_temp
        if self.score_func == "sigmoid":
            scores = mx.sigmoid(logits)
        elif self.score_func == "softmax":
            scores = mx.softmax(logits, axis=-1)
        else:
            scores = mx.sqrt(nn.softplus(logits))
        biased = scores + self.e_score_correction_bias
        chosen = mx.argpartition(-biased, self.top_k - 1, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.route_scale


class Experts(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=DeepseekV41SwiGLU(args.swiglu_limit),
        )

    def __call__(self, x: mx.array, indices: mx.array, weights: mx.array) -> mx.array:
        routed = self.switch_mlp(x, indices)
        return mx.sum(
            routed.astype(mx.float32) * weights[..., None].astype(mx.float32), axis=1
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = Experts(args)
        self.shared_experts = SharedMLP(args)
        self.dim = args.hidden_size
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        indices, weights = self.gate(flat)
        routed = self.experts(flat, indices, weights)
        out = routed + self.shared_experts(flat).astype(mx.float32)
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out.reshape(shape).astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_idx = layer_id
        self.self_attn = Attention(args, layer_id)
        self.mlp = MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)
        if layer_id in args.engram_layer_ids:
            self.engram = Engram(args, layer_id)

    def __call__(
        self,
        streams: mx.array,
        pre_mix: mx.array,
        *,
        start_pos: int,
        window_idx: mx.array,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> tuple[mx.array, mx.array]:
        attn_pre, attn_post, attn_comb = self.attn_hc(streams)
        h = self.input_layernorm(_collapse(streams, pre_mix))
        h = self.self_attn(
            h, start_pos=start_pos, window_idx=window_idx, cache=cache, shared=shared
        )
        streams = _expand(h, streams, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(streams)
        h = self.mlp(self.post_attention_layernorm(_collapse(streams, attn_pre)))
        streams = _expand(h, streams, ffn_post, ffn_comb)
        return streams, ffn_pre


def _shared_reads(args: ModelArgs, layer: int) -> dict[str, int]:
    """The shared state a layer reads, as {key: layer that writes it}."""
    ratios = args.compress_ratios
    if layer >= len(ratios) or not ratios[layer]:
        return {}
    kv, index = set(args.kv_source_layer_ids), set(args.index_source_layer_ids)

    def last(match: Callable[[int], bool]) -> Optional[int]:
        return max((j for j in range(layer + 1) if match(j)), default=None)

    reads = {"compress_kv": last(lambda j: j in kv and ratios[j])}
    if layer in index:
        reads["index_k"] = last(lambda j: j in kv and j in index)
        source = args.candidate_source_layer_id
        if 0 <= source < layer and args.candidate_topk_blocks > 0:
            reads["candidates"] = source
    else:
        reads["topk_idx"] = last(lambda j: j in index)
    return {key: source for key, source in reads.items() if source is not None}


def _crossing(args: ModelArgs, boundary: int) -> dict[str, int]:
    """The shared state that layers from `boundary` on read from layers before
    it, as {key: layer that writes it}."""
    crossing = {}
    for layer in range(boundary, args.num_hidden_layers):
        for key, source in _shared_reads(args, layer).items():
            if source < boundary:
                crossing[key] = source
    return crossing


def _entries(args: ModelArgs, source: Optional[int], position: int) -> int:
    """The compressed entries of kv source `source` after `position` tokens."""
    return position // args.compress_ratios[source] if source is not None else 0


def _index_keys(args: ModelArgs, layer: int, position: int) -> int:
    """The index keys that index layer `layer` scores after `position` tokens."""
    return _entries(args, _shared_reads(args, layer).get("index_k"), position)


# The shared state that grows each step, kept in the LayerCache field of that name.
_CACHE_FIELDS = ("compress_kv", "index_k")


def _layout(
    args: ModelArgs,
    crossing: dict[str, int],
    batch: int,
    length: int,
    start_pos: int,
    dtype: mx.Dtype,
) -> list[tuple[str, tuple[int, ...], mx.Dtype]]:
    """The parts of the message at a stage boundary, as (key, shape, dtype).

    The compressed KV and the index keys have the dtype of the streams, because
    the next stage cannot know the dtype the stage before makes them in.
    """
    end = start_pos + length
    layout = [
        ("streams", (batch, length, args.hc_mult, args.hidden_size), dtype),
        ("pre_mix", (batch, length, args.hc_mult), mx.float32),
    ]
    for key, source in crossing.items():
        if key in _CACHE_FIELDS:
            # Only the new entries: the next stage keeps the earlier ones.
            new = _entries(args, source, end) - _entries(args, source, start_pos)
            width = args.head_dim if key == "compress_kv" else args.index_head_dim
            layout.append((key, (batch, new, width), dtype))
        elif key == "candidates":
            # One flag for each candidate block.
            block = args.candidate_block_size
            blocks = (_index_keys(args, source, end) + block - 1) // block
            layout.append((key, (batch, length, blocks), mx.bool_))
        else:
            topk = min(args.index_topk, _index_keys(args, source, end))
            layout.append((key, (batch, length, topk), mx.int32))
    return layout


# A view at a byte offset that is not a multiple of its item size reads the
# wrong bytes, so each part of a message starts at a multiple of 8 bytes.
_ALIGN = 8


def _nbytes(shape: tuple[int, ...], dtype: mx.Dtype) -> int:
    """The bytes of one part in a message, with its padding."""
    size = math.prod(shape) * dtype.size
    return (size + _ALIGN - 1) // _ALIGN * _ALIGN


def _pack(parts: list[mx.array]) -> mx.array:
    """The bytes of all parts in one array: two sends can run in a different
    order than the two receives on the other rank."""
    data = []
    for part in parts:
        flat = part.view(mx.uint8).reshape(-1)
        pad = _nbytes(part.shape, part.dtype) - flat.size
        data.append(mx.pad(flat, (0, pad)) if pad else flat)
    return mx.concatenate(data)


def _unpack(
    packed: mx.array, layout: list[tuple[str, tuple[int, ...], mx.Dtype]]
) -> dict[str, mx.array]:
    parts = {}
    start = 0
    for key, shape, dtype in layout:
        data = packed[start : start + math.prod(shape) * dtype.size]
        parts[key] = data.reshape(*shape[:-1], shape[-1] * dtype.size).view(dtype)
        start += _nbytes(shape, dtype)
    return parts


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args
        self._engram_hasher = None

    def _engram_ids(self, inputs: mx.array, cache: LayerCache) -> Optional[mx.array]:
        """The engram hash ids of the inputs, if a local layer needs them."""
        if not any("engram" in layer for layer in self.pipeline_layers):
            return None
        if self._engram_hasher is None:
            raise ValueError(
                "This DeepSeek V4.1 checkpoint has engram layers, which hash the "
                "token ids: call model.attach_tokenizer(tokenizer) before running "
                "the model."
            )
        hashes, cache.engram_history = self._engram_hasher(
            np.array(inputs), cache.engram_history
        )
        # The tables have fewer than 2**31 rows.
        return mx.array(hashes.astype(np.int32))

    def pipeline(
        self, group: mx.distributed.Group, split: Optional[list[int]] = None
    ) -> None:
        super().pipeline(group, split)
        self._shared_in = _crossing(self.args, self.start_idx)
        self._shared_out = _crossing(self.args, self.end_idx)

    def make_cache(self) -> list[LayerCache]:
        return [LayerCache(self.args.sliding_window) for _ in self.pipeline_layers]

    def _receive(
        self,
        like: mx.array,
        start_pos: int,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> tuple[mx.array, mx.array]:
        """Receive the streams and pre_mix, and put the shared state in `shared`."""
        batch, length = like.shape[:2]
        layout = _layout(
            self.args, self._shared_in, batch, length, start_pos, like.dtype
        )
        size = sum(_nbytes(shape, dtype) for _, shape, dtype in layout)
        packed = mx.distributed.recv((size,), mx.uint8, self.pipeline_rank + 1)
        parts = _unpack(packed, layout)
        streams = parts.pop("streams")
        pre_mix = parts.pop("pre_mix")
        for key, part in parts.items():
            if key in _CACHE_FIELDS:
                # The first layer never writes this field, so it keeps the
                # entries of the source in the stage before.
                if part.shape[1]:
                    setattr(cache, key, _append(getattr(cache, key), part))
                shared[key] = getattr(cache, key)
            elif key == "candidates":
                keys = _index_keys(self.args, self._shared_in[key], start_pos + length)
                mask = mx.repeat(part, self.args.candidate_block_size, axis=-1)
                shared[key] = mask[..., :keys] if keys else None
            else:
                shared[key] = part
        return streams, pre_mix

    def _send(
        self,
        streams: mx.array,
        pre_mix: mx.array,
        start_pos: int,
        shared: dict[str, Optional[mx.array]],
    ) -> mx.array:
        """Send the streams, pre_mix and the shared state the next stages read."""
        batch, length = streams.shape[:2]
        layout = _layout(
            self.args, self._shared_out, batch, length, start_pos, streams.dtype
        )
        state = {**shared, "streams": streams, "pre_mix": pre_mix}
        parts = []
        for key, shape, dtype in layout:
            part = state.get(key)
            if part is None:
                # Not written yet, so the layout has no entries either.
                part = mx.zeros(shape, dtype)
            elif key in _CACHE_FIELDS:
                # It holds all entries so far; send only the new ones.
                part = part[:, part.shape[1] - shape[1] :]
            elif key == "candidates":
                part = part[..., :: self.args.candidate_block_size]
            if part.shape != shape or part.dtype != dtype:
                raise ValueError(
                    f"The next pipeline stage expects {key} as {dtype} {shape}, "
                    f"not {part.dtype} {part.shape}."
                )
            parts.append(part)
        return mx.distributed.send(_pack(parts), self.pipeline_rank - 1)

    def __call__(
        self, inputs: mx.array, cache: Optional[list[LayerCache]] = None
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        batch, length, _ = h.shape
        layers = self.pipeline_layers
        cache = cache or self.make_cache()
        start_pos = cache[0].offset
        # Every layer's window holds the same rows, so all use the same indices.
        previous = 0 if cache[0].window is None else cache[0].window.shape[1]
        window_idx = _window_indices(
            previous=previous, length=length, window=self.args.sliding_window
        )
        window_idx = mx.broadcast_to(window_idx, (batch, *window_idx.shape))

        streams = mx.broadcast_to(
            h[:, :, None, :], (batch, length, self.args.hc_mult, self.args.hidden_size)
        )
        pre_mix = mx.zeros((batch, length, self.args.hc_mult), dtype=mx.float32)
        pre_mix[..., 0] = 1.0
        rank, size = self.pipeline_rank, self.pipeline_size
        engram_ids = self._engram_ids(inputs, cache[0])
        shared = {}
        if rank < size - 1:
            streams, pre_mix = self._receive(streams, start_pos, cache[0], shared)

        for layer, layer_cache in zip(layers, cache, strict=True):
            if "engram" in layer:
                streams = layer.engram(
                    streams, engram_ids[:, :, layer.engram.table_index]
                )
            streams, pre_mix = layer(
                streams,
                pre_mix,
                start_pos=start_pos,
                window_idx=window_idx,
                cache=layer_cache,
                shared=shared,
            )

        if rank > 0:
            sent = self._send(streams, pre_mix, start_pos, shared)
            streams, pre_mix = mx.depends([streams, pre_mix], sent)
            # Prefill only evaluates the cache, which must still send.
            cache[-1].window = mx.depends(cache[-1].window, sent)
        out = _collapse(streams, pre_mix)
        if size > 1:
            # Rank 0 runs the last layers, and every rank needs its output.
            out = mx.distributed.all_gather(out)[:batch]
        return self.norm(out)


_DROP_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")
_EXPERT_KEY = r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)"
_EXPERT_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _rename(key: str) -> str:
    """Map one DeepSeek checkpoint name to its MLX name."""
    if key == "head.weight":
        return "lm_head.weight"
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    key = f"model.{key}"
    if ".engram." in key:
        # Engram keeps the checkpoint names, wkv included.
        return key
    key = key.replace(".attn.", ".self_attn.")
    key = key.replace(".ffn.", ".mlp.")
    key = key.replace(".attn_norm.", ".input_layernorm.")
    key = key.replace(".ffn_norm.", ".post_attention_layernorm.")
    key = key.replace(".hc_attn_", ".attn_hc.")
    key = key.replace(".hc_ffn_", ".ffn_hc.")
    key = key.replace(".attn_sink", ".sinks")
    key = key.replace(".wq_a.", ".q_a_proj.")
    key = key.replace(".wq_b.", ".q_b_proj.")
    key = key.replace(".wkv.", ".kv_proj.")
    key = key.replace(".wgate.", ".gate_proj.")
    key = key.replace(".wo_a.", ".o_a_proj.")
    key = key.replace(".wo_b.", ".o_b_proj.")
    key = key.replace(".q_norm.", ".q_a_norm.")
    key = key.replace(".compressor.norm.", ".compressor.kv_norm.")
    key = key.replace(".indexer.wk.", ".indexer.k_proj.")
    key = key.replace(".gate.bias", ".gate.e_score_correction_bias")
    key = key.replace(".shared_experts.w1.", ".shared_experts.gate_proj.")
    key = key.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
    key = key.replace(".shared_experts.w3.", ".shared_experts.up_proj.")
    return key


def _repack(weight: mx.array, scale: mx.array) -> tuple[mx.array, mx.array]:
    """Put an fp8 or fp4 weight and its ue8m0 scale in MLX's quantized layout."""
    if scale.dtype != mx.uint8:
        scale = (mx.log2(scale.astype(mx.float32)) + 127).astype(mx.uint8)
    if scale.shape[-2] != weight.shape[-2]:
        # fp8 keeps one scale per 32x32 tile, MLX one per output row.
        scale = mx.repeat(scale, 32, axis=-2)[..., : weight.shape[-2], :]
    return weight.view(mx.uint32), scale


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[list[LayerCache]] = None
    ) -> mx.array:
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.pipeline_layers

    def make_cache(self) -> list[LayerCache]:
        return self.model.make_cache()

    def attach_tokenizer(self, tokenizer) -> None:
        """Build the engram n-gram hasher from the checkpoint's tokenizer (a
        Hugging Face fast tokenizer or an mlx_lm TokenizerWrapper). Models
        without engram layers ignore it."""
        if self.args.engram_layer_ids:
            self.model._engram_hasher = EngramHasher.from_tokenizer(
                self.args, tokenizer
            )

    def shard(self, group: Optional[mx.distributed.Group] = None) -> None:
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        heads, groups = self.args.num_attention_heads, self.args.o_groups
        if heads % size or groups % size:
            raise ValueError(
                f"Tensor parallelism over {size} ranks needs {size} to divide both "
                f"the {heads} attention heads and the {groups} output groups "
                f"(heads % {size} = {heads % size}, groups % {size} = "
                f"{groups % size}); use a rank count that divides {math.gcd(heads, groups)}."
            )
        for layer in self.layers:
            attn = layer.self_attn
            attn.q_b_proj = shard_linear(attn.q_b_proj, "all-to-sharded", group=group)
            # An output group reads only its own heads, so a rank keeps whole groups.
            shard_inplace(attn.o_a_proj, lambda path, weight: 0, group=group)
            attn.o_b_proj = shard_linear(attn.o_b_proj, "sharded-to-all", group=group)
            attn.num_heads //= size
            attn.o_groups //= size
            attn._group_ids = mx.arange(attn.o_groups)[None]
            attn.sinks = attn.sinks[rank * attn.num_heads : (rank + 1) * attn.num_heads]

            layer.mlp.sharding_group = group
            for mlp in (layer.mlp.experts.switch_mlp, layer.mlp.shared_experts):
                shard_inplace(mlp.gate_proj, "all-to-sharded", group=group)
                shard_inplace(mlp.up_proj, "all-to-sharded", group=group)
                shard_inplace(mlp.down_proj, "sharded-to-all", group=group)

            if "engram" in layer:
                # Rows of the table are split over the ranks; wkv, q_weight
                # and k_weight stay replicated.
                layer.engram.shard(group)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Rename the checkpoint keys and keep the fp8/fp4 weights packed."""
        # A converted checkpoint already has MLX names; the published one never
        # starts a key with "model.".
        if any(key.startswith("model.") for key in weights):
            return weights
        text = {
            key: value
            for key, value in weights.items()
            if not key.startswith(_DROP_PREFIXES)
        }
        clean = {}
        experts = {}
        for key, value in text.items():
            expert = re.fullmatch(_EXPERT_KEY, key)
            if expert is not None:
                layer, index, proj, kind = expert.groups()
                layer, index = int(layer), int(index)
                rows = experts.setdefault((layer, proj), {})
                rows.setdefault(index, {})[kind] = value
                continue
            # bias_vl routes image tokens, which the text model never sees.
            if key.endswith(".gate.bias_vl"):
                continue
            if (
                key.endswith(".scale")
                and f"{key.removesuffix('.scale')}.weight" in text
            ):
                continue
            scale = None
            if key.endswith(".weight"):
                scale = text.get(f"{key.removesuffix('.weight')}.scale")
            name = _rename(key)
            if scale is not None:
                value, scale = _repack(value, scale)
            if name.endswith(".o_a_proj.weight"):
                # wo_a is block diagonal, so split it back into its groups.
                value = value.reshape(self.args.o_groups, -1, value.shape[-1])
                if scale is not None:
                    scale = scale.reshape(self.args.o_groups, -1, scale.shape[-1])
            clean[name] = value
            if scale is not None:
                clean[f"{name.removesuffix('weight')}scales"] = scale

        n_experts = self.args.n_routed_experts
        for (layer, proj), rows in experts.items():
            if len(rows) != n_experts:
                raise ValueError(
                    f"Layer {layer} has {len(rows)} {proj} expert weights, "
                    f"but the config declares {n_experts} experts."
                )
            parts = [rows[index] for index in sorted(rows)]
            name = f"model.layers.{layer}.mlp.experts.switch_mlp.{_EXPERT_PROJ[proj]}"
            value = mx.stack([part["weight"] for part in parts])
            if "scale" in parts[0]:
                scales = mx.stack([part["scale"] for part in parts])
                value, scales = _repack(value, scales)
                clean[f"{name}.scales"] = scales
            clean[f"{name}.weight"] = value
        return clean

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        # The reference runtime keeps these in float32.
        keep = (
            "attn_hc.",
            "ffn_hc.",
            "sinks",
            "e_score_correction_bias",
            "compressor.",
            "engram.q_weight",
            "engram.k_weight",
        )

        def predicate(k: str) -> bool:
            return not any(s in k for s in keep)

        return predicate
