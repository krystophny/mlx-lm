# Copyright © 2026 Apple Inc.

import copy
import unittest
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm import utils
from mlx_lm.models import deepseek_v41
from mlx_lm.models.rope_utils import YarnRoPE

# Values from the DeepSeek-V4.1-Flash-1017 config.json.
_FLASH_ENGRAM = dict(
    engram_layer_ids=(1, 14),
    engram_num_embeddings=(384006168, 384016682),
    engram_max_ngram_size=4,
    engram_vocab_size=16000000,
    engram_n_heads=8,
    engram_head_dim=256,
    engram_pad_token_id=2,
    engram_compressed_vocab_size=99092,
)


def _tiny_args(**kwargs):
    params = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=32,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        sliding_window=4,
        compress_ratios=(0, 2, 2, 2, 1, 1),
        kv_source_layer_ids=(1, 4),
        index_source_layer_ids=(1, 2, 4, 5),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=3,
        candidate_source_layer_id=4,
        candidate_topk_blocks=2,
        candidate_block_size=2,
    )
    params.update(kwargs)
    return deepseek_v41.ModelArgs(**params)


def _slow_primes(args):
    """Trial-division transcription of the official EngramLayout."""

    def isprime(n):
        return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))

    primes, seen = [], set()
    for _ in args.engram_layer_ids:
        per_ngram = []
        for _ in range(args.engram_max_ngram_size - 1):
            sizes, current = [], args.engram_vocab_size - 1
            for _ in range(args.engram_n_heads):
                current += 1
                while not isprime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return np.array(primes, dtype=np.int64)


def _engram_args(compressed_vocab, **kwargs):
    """Tiny engram args whose table rows follow the prime layout."""
    engram = dict(
        engram_layer_ids=(1, 4),
        engram_max_ngram_size=3,
        engram_vocab_size=40,
        engram_n_heads=2,
        engram_head_dim=8,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=compressed_vocab,
    )
    engram.update(kwargs)
    n = len(engram["engram_layer_ids"])
    args = _tiny_args(**engram, engram_num_embeddings=(1,) * n)
    rows = tuple(int(r) for r in _slow_primes(args).reshape(n, -1).sum(-1))
    return replace(args, engram_num_embeddings=rows)


def _official_hashes(args, token_map, vocab, ids):
    """NumPy transcription of the official NgramHashState over a full
    position cache (start_pos 0, no image mask)."""
    primes = _slow_primes(args)
    flat = primes.reshape(len(args.engram_layer_ids), -1)
    offsets = np.stack([np.cumsum([0, *f[:-1]]) for f in flat])
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab) // 2)
    multipliers = np.stack(
        [
            np.random.default_rng(10007 * lid).integers(
                0, bound, size=(args.engram_max_ngram_size,), dtype=np.int64
            )
            * 2
            + 1
            for lid in args.engram_layer_ids
        ]
    )
    pad_id = token_map[args.engram_pad_token_id]
    batch, seqlen = ids.shape
    cache = token_map[ids]
    positions = np.broadcast_to(np.arange(seqlen), (batch, seqlen))
    tokens, blocked = [], np.zeros((batch, seqlen), dtype=bool)
    for shift in range(args.engram_max_ngram_size):
        source = np.take_along_axis(cache, np.maximum(positions - shift, 0), 1)
        blocked = blocked | (positions < shift) | (source == -1)
        tokens.append(np.where(blocked, pad_id, source))
    tokens = np.stack(tokens, -1)
    products = tokens[:, :, None, :] * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, args.engram_max_ngram_size):
        rolling = np.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling[..., None] % primes[:, i - 1])
    return np.concatenate(hashes, -1) + offsets


def _fake_tokenizer(extra=128):
    """A byte-level fast tokenizer with case, accent, whitespace and
    partial-UTF-8 variants, padded with filler tokens to `extra` entries."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

    def encode(text):
        return "".join(piece for piece, _ in byte_level.pre_tokenize_str(text))

    texts = ["<pad>", "<s>", "</s>", "The", " the", "THE", "café", "cafe"]
    texts += [" ", "  ", "\n", "e", "é", "x"]
    vocab = {}
    for text in texts:
        vocab.setdefault(encode(text), len(vocab))
    e_acute = encode("é")
    # The two bytes of "é" alone are partial UTF-8 tokens.
    vocab[e_acute[0]] = len(vocab)
    vocab[e_acute[1]] = len(vocab)
    i = 0
    while len(vocab) < extra:
        vocab.setdefault(encode(f"w{i}"), len(vocab))
        i += 1
    tok = Tokenizer(models.WordLevel(vocab, unk_token=encode("<pad>")))
    tok.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tok), vocab, encode


def _e4m3_to_float(bits):
    bits = bits.astype(np.int64)
    sign = np.where(bits & 0x80, -1.0, 1.0)
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    normal = (1 + man / 8.0) * 2.0 ** (exp - 7)
    subnormal = (man / 8.0) * 2.0**-6
    return sign * np.where(exp == 0, subnormal, normal)


def _random_e4m3(rng, shape):
    bits = rng.integers(0, 256, size=shape).astype(np.uint8)
    # 0x7f and 0xff are NaN in e4m3.
    return np.where((bits & 0x7F) == 0x7F, 0x38, bits).astype(np.uint8)


class TestDeepseekV41(unittest.TestCase):
    def test_rope_inverse(self):
        rope = YarnRoPE(
            dims=4,
            traditional=True,
            base=160000.0,
            scaling_factor=2.0,
            original_max_position_embeddings=16,
            mscale=0.0,
            mscale_all_dim=0.0,
        )
        for shape in ((1, 5, 8), (1, 5, 2, 8)):
            x = mx.random.normal(shape)
            y = deepseek_v41._apply_rope(x, rope, rope_dim=4, offset=4, scale=2.0)
            y = deepseek_v41._apply_rope(
                y, rope, rope_dim=4, offset=4, scale=2.0, inverse=True
            )
            self.assertTrue(mx.allclose(y, x, atol=1e-5))

    def test_cached_matches_forward(self):
        args = _tiny_args()
        mx.random.seed(0)
        model = deepseek_v41.Model(args)
        tokens = mx.random.randint(0, args.vocab_size, (2, 16))
        full = model(tokens)

        cache = model.make_cache()
        steps = []
        for start, end in ((0, 7), (7, 15), (15, 16)):
            steps.append(model(tokens[:, start:end], cache=cache))
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )

    def test_engram_config(self):
        config = {
            "model_type": "deepseek_v41",
            "text_config": {
                **_FLASH_ENGRAM,
                "num_nextn_predict_layers": 1,
                "dspark_block_size": 16,
                "dspark_target_layer_ids": [1, 2],
            },
            "vision_config": {"hidden_size": 1024},
        }
        args = deepseek_v41.ModelArgs.from_dict(config)
        self.assertEqual(list(args.engram_layer_ids), [1, 14])
        self.assertEqual(args.engram_pad_token_id, 2)
        self.assertFalse(hasattr(args, "dspark_block_size"))
        with self.assertRaises(ValueError):
            replace(args, engram_num_embeddings=(1,))

    def test_engram_flash_prime_layout(self):
        args = _tiny_args(**_FLASH_ENGRAM)
        primes = deepseek_v41.engram_primes(args)
        self.assertEqual(primes.shape, (2, 3, 8))
        self.assertEqual(
            [int(r) for r in primes.reshape(2, -1).sum(-1)], [384006168, 384016682]
        )
        self.assertEqual(len(set(primes.flatten().tolist())), 48)
        self.assertGreater(primes.min(), 16000000 - 1)

    def test_engram_compressed_token_map(self):
        hf, vocab, encode = _fake_tokenizer()
        token_map, size = deepseek_v41.build_compressed_token_map(hf)
        ids = {text: vocab[encode(text)] for text in vocab_texts()}

        def same(a, b):
            return token_map[ids[a]] == token_map[ids[b]]

        self.assertTrue(same("The", " the") and same("The", "THE"))
        self.assertTrue(same("café", "cafe") and same("é", "e"))
        self.assertTrue(same(" ", "  ") and same(" ", "\n"))
        self.assertFalse(same("The", "x") or same(" ", "e"))
        e_acute = encode("é")
        partial = [token_map[vocab[e_acute[0]]], token_map[vocab[e_acute[1]]]]
        self.assertNotEqual(partial[0], partial[1])
        self.assertNotIn(partial[0], token_map[: vocab[e_acute[0]]])
        self.assertEqual(size, len(set(token_map.tolist())))
        self.assertEqual(token_map.max() + 1, size)

        wrapped = utils.TokenizerWrapper(hf)
        wrapped_map, wrapped_size = deepseek_v41.build_compressed_token_map(wrapped)
        self.assertEqual(wrapped_size, size)
        self.assertTrue(np.array_equal(wrapped_map, token_map))

    def test_engram_hasher_chunked_matches_official(self):
        rng = np.random.default_rng(0)
        vocab = 37
        token_map = rng.integers(0, vocab, size=100)
        token_map[:vocab] = np.arange(vocab)
        for ngram in (2, 3, 4):
            args = _engram_args(vocab, engram_max_ngram_size=ngram)
            hasher = deepseek_v41.EngramHasher(args, token_map, vocab)
            ids = rng.integers(0, len(token_map), size=(3, 13))
            official = _official_hashes(args, token_map, vocab, ids)
            full, _ = hasher(ids)
            self.assertTrue(np.array_equal(full, official))
            history, chunks = None, []
            for start, end in [(0, 1), (1, 6)] + [(i, i + 1) for i in range(6, 13)]:
                part, history = hasher(ids[:, start:end], history)
                chunks.append(part)
            self.assertTrue(np.array_equal(np.concatenate(chunks, 1), official))
            rows = np.array(args.engram_num_embeddings)[None, None, :, None]
            self.assertTrue((official >= 0).all() and (official < rows).all())

        args = _engram_args(vocab)
        with self.assertRaises(ValueError):
            deepseek_v41.EngramHasher(args, token_map, vocab + 1)
        with self.assertRaises(ValueError):
            bad = replace(args, engram_num_embeddings=(5, 5))
            deepseek_v41.EngramHasher(bad, token_map, vocab)

    def test_engram_forward_matches_reference(self):
        args = _engram_args(37, hidden_size=16, hc_mult=3, rms_norm_eps=1e-6)
        engram = deepseek_v41.Engram(args, layer_id=4)
        mx.random.seed(1)
        engram.q_weight = mx.random.normal(engram.q_weight.shape)
        engram.k_weight = mx.random.normal(engram.k_weight.shape)
        rows = args.engram_num_embeddings[1]
        streams = mx.random.normal((2, 5, 3, 16))
        # A zero stream gives a dot product of exactly zero.
        streams[0, 2, 1] = 0.0
        ids = mx.random.randint(0, rows, (2, 5, 4))
        out = engram(streams, ids)

        h = np.array(streams, dtype=np.float64)
        table = np.array(engram.embed.weight, dtype=np.float64)
        wkv = np.array(engram.wkv.weight, dtype=np.float64)
        kv = table[np.array(ids)].reshape(2, 5, -1) @ wkv.T
        key = kv[..., : 3 * 16].reshape(2, 5, 3, 16)
        value = kv[..., 3 * 16 :]
        weight = np.array(engram.q_weight) * np.array(engram.k_weight)
        eps = args.rms_norm_eps
        rstd = 1 / np.sqrt((h**2).mean(-1) + eps) / np.sqrt((key**2).mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * 16**-0.5
        self.assertEqual(dot[0, 2, 1], 0.0)
        signed = np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)
        gate = 1 / (1 + np.exp(-signed))
        expected = h + gate[..., None] * value[..., None, :]
        self.assertTrue(np.allclose(np.array(out), expected, rtol=1e-4, atol=1e-4))
        self.assertAlmostEqual(
            float(out[0, 2, 1, 0]), 1 / (1 + np.exp(-1e-3)) * value[0, 2, 0], 5
        )

    def _engram_model(self):
        hf, _, _ = _fake_tokenizer()
        _, size = deepseek_v41.build_compressed_token_map(hf)
        mx.random.seed(0)
        model = deepseek_v41.Model(_engram_args(size))
        for layer in (model.layers[1], model.layers[4]):
            # Strong engram signals so a hashing mistake changes the logits.
            layer.engram.embed.weight = 3 * layer.engram.embed.weight
            layer.engram.q_weight = mx.random.normal(layer.engram.q_weight.shape)
        return model, hf

    def test_engram_model_cached_matches_forward(self):
        model, hf = self._engram_model()
        tokens = mx.random.randint(0, model.args.vocab_size, (2, 16))
        with self.assertRaises(ValueError):
            model(tokens)
        model.attach_tokenizer(utils.TokenizerWrapper(hf))
        full = model(tokens)

        cache = model.make_cache()
        steps = []
        for start, end in ((0, 7), (7, 15), (15, 16)):
            steps.append(model(tokens[:, start:end], cache=cache))
        self.assertEqual(cache[0].engram_history.shape, (2, 2))
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )

        # Without the engram layers the logits change.
        plain = copy.deepcopy(model)
        for layer in plain.layers:
            if "engram" in layer:
                del layer["engram"]
        self.assertFalse(mx.allclose(plain(tokens), full, atol=1e-3))

    def test_engram_sanitize_and_mxfp8_embedding(self):
        args = _engram_args(37, engram_head_dim=32)
        model = deepseek_v41.Model(args)
        rows = args.engram_num_embeddings[0]
        cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        out_dim = args.hidden_size * (args.hc_mult + 1)
        rng = np.random.default_rng(0)
        table = _random_e4m3(rng, (rows, 32))
        table_scale = rng.integers(120, 132, size=(rows, 1)).astype(np.uint8)
        wkv = _random_e4m3(rng, (out_dim, cols * 32))
        wkv_scale = rng.integers(120, 132, size=(out_dim // 32, cols)).astype(np.uint8)
        ones = mx.ones((args.hc_mult, args.hidden_size), dtype=mx.bfloat16)
        weights = {
            "layers.1.engram.embed.weight": mx.array(table),
            "layers.1.engram.embed.scale": mx.array(table_scale),
            "layers.1.engram.wkv.weight": mx.array(wkv),
            "layers.1.engram.wkv.scale": mx.array(wkv_scale),
            "layers.1.engram.q_weight": ones,
            "layers.1.engram.k_weight": ones,
            "mtp.0.engram.q_weight": ones,
            "vision.blocks.0.weight": ones,
            "layers.1.ffn.gate.bias_vl": mx.zeros((8,)),
        }
        clean = model.sanitize(weights)
        prefix = "model.layers.1.engram"
        self.assertEqual(
            sorted(clean),
            sorted(
                f"{prefix}.{name}"
                for name in (
                    "embed.weight",
                    "embed.scales",
                    "wkv.weight",
                    "wkv.scales",
                    "q_weight",
                    "k_weight",
                )
            ),
        )
        self.assertEqual(clean[f"{prefix}.wkv.scales"].shape, (out_dim, cols))

        engram = model.layers[1].engram
        for name in ("embed", "wkv"):
            params = utils.infer_quant_config(
                f"{prefix}.{name}", getattr(engram, name), clean
            )
            self.assertEqual(params["mode"], "mxfp8")
        engram.embed = engram.embed.to_quantized(group_size=32, bits=8, mode="mxfp8")
        self.assertIsInstance(engram.embed, nn.QuantizedEmbedding)
        engram.wkv = engram.wkv.to_quantized(group_size=32, bits=8, mode="mxfp8")
        model.load_weights(list(clean.items()), strict=False)

        ids = rng.integers(0, rows, size=(2, 3, cols))
        reference = _e4m3_to_float(table) * 2.0 ** (table_scale.astype(np.int64) - 127)
        looked_up = np.array(engram.lookup(mx.array(ids)).astype(mx.float32))
        self.assertTrue(np.allclose(looked_up, reference[ids], rtol=1e-6, atol=0))

        x = rng.normal(size=(4, cols * 32)).astype(np.float32)
        dense = _e4m3_to_float(wkv) * 2.0 ** (
            np.repeat(wkv_scale, 32, axis=0).repeat(32, axis=1).astype(np.int64) - 127
        )
        out = np.array(engram.wkv(mx.array(x)).astype(mx.float32))
        self.assertTrue(np.allclose(out, x @ dense.T, rtol=1e-3, atol=1e-2))

    def test_engram_row_shard(self):
        model, hf = self._engram_model()
        model.attach_tokenizer(hf)
        tokens = mx.random.randint(0, model.args.vocab_size, (1, 9))
        before = model(tokens)
        model.shard(mx.distributed.init())
        self.assertTrue(mx.allclose(model(tokens), before, rtol=1e-5, atol=1e-5))

        # An odd row count leaves the last rank one row short.
        rows = 2 * 37 + 1
        args = replace(
            model.args, engram_num_embeddings=(rows, rows), engram_head_dim=32
        )
        ids = mx.random.randint(0, rows, (2, 5, 4))
        for quantize in (False, True):
            engram = deepseek_v41.Engram(args, layer_id=4)
            engram.embed.weight = mx.random.normal(engram.embed.weight.shape)
            if quantize:
                engram.embed = engram.embed.to_quantized(mode="mxfp8")
            full = engram.lookup(ids)
            parts = []
            for rank in range(2):
                part = copy.deepcopy(engram)
                part._shard_rows(rank, 2)
                self.assertEqual(part.embed.weight.shape[0], 38 - rank)
                parts.append(part.lookup(ids))
            self.assertTrue(mx.array_equal(parts[0] + parts[1], full))


def vocab_texts():
    return ["The", " the", "THE", "café", "cafe", " ", "  ", "\n", "e", "é", "x"]


if __name__ == "__main__":
    unittest.main()
