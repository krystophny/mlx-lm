# Copyright © 2026 Apple Inc.
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from unittest.mock import patch
from mlx.utils import tree_flatten
from mlx_lm.generate import (
    generate_step,
    mtp_speculative_generate_step,
    stream_generate,
)
from mlx_lm.models import glm_moe_dsa
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.models.deepseek_v32 import ModelArgs, MtpModule
from mlx_lm.utils import load_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


def tiny_model(all_accept=False):
    config = dataclasses.asdict(ModelArgs())
    config.update(
        model_type="glm_moe_dsa",
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        n_shared_experts=1,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        index_head_dim=64,
        index_n_heads=16,
        index_topk=8,
        indexer_types=["full", "full", "shared", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=1,
    )
    mx.random.seed(4)
    model = glm_moe_dsa.Model(glm_moe_dsa.ModelArgs.from_dict(config))
    model.mtp = MtpModule(model.args)
    for layer in [*model.layers, model.mtp.layer]:
        layer.mlp.gate.weight = mx.random.normal(layer.mlp.gate.weight.shape) * 0.03
    if all_accept:
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    return model, config


def cache_for(model):
    return [*model.make_cache(), model.make_mtp_cache()]


class TestMTP(unittest.TestCase):
    def test_matches_target_and_cleans_up_at_length_limit(self):
        for all_accept in (False, True):
            model, _ = tiny_model(all_accept)
            prompt = mx.arange(20) % 64
            for bits in (None, 8):
                for limit in (1, 2, 3, 12):
                    with self.subTest(accepted=all_accept, bits=bits, limit=limit):
                        cache = cache_for(model)
                        expected = [
                            int(t)
                            for t, _ in generate_step(
                                prompt,
                                model,
                                max_tokens=limit,
                                prefill_step_size=8,
                                kv_bits=bits,
                                quantized_kv_start=0,
                                kv_group_size=64,
                            )
                        ]
                        output = list(
                            mtp_speculative_generate_step(
                                prompt,
                                model,
                                max_tokens=limit,
                                prefill_step_size=8,
                                prompt_cache=cache,
                                kv_bits=bits,
                                quantized_kv_start=0,
                            )
                        )
                        self.assertEqual([int(t) for t, _, _ in output], expected)
                        self.assertEqual(
                            [c.offset for c in cache], [20 + limit - 1] * 5
                        )
                        if all_accept and limit > 1:
                            self.assertTrue(any(drafted for _, _, drafted in output))
                        if bits:
                            self.assertIsInstance(cache[-1][0], QuantizedKVCache)
                            self.assertIsInstance(cache[-1][1], KVCache)

    def test_cancel_and_reuse_changed_prefix(self):
        for all_accept in (False, True):
            model, _ = tiny_model(all_accept)
            prompt = mx.arange(20) % 64
            cache = cache_for(model)
            generator = mtp_speculative_generate_step(
                prompt,
                model,
                max_tokens=-1,
                prompt_cache=cache,
                prefill_step_size=8,
                kv_bits=8,
                quantized_kv_start=0,
            )
            generated = [int(next(generator)[0]) for _ in range(3)]
            generator.close()
            self.assertEqual([c.offset for c in cache], [22] * 5)
            # Change a suffix while retaining a prefix, then compare cold output.
            query = prompt.tolist() + generated + [11, 17]
            retained = 17
            for entry in cache:
                entry.trim(entry.offset - retained)
            expected = [
                int(t)
                for t, _ in generate_step(
                    mx.array(query),
                    model,
                    max_tokens=7,
                    kv_bits=8,
                    quantized_kv_start=0,
                    prefill_step_size=8,
                )
            ]
            actual = [
                int(t)
                for t, _, _ in mtp_speculative_generate_step(
                    mx.array(query[retained:]),
                    model,
                    max_tokens=7,
                    kv_bits=8,
                    quantized_kv_start=0,
                    prompt_cache=cache,
                    prefill_step_size=8,
                )
            ]
            self.assertEqual(actual, expected)

    def test_processors_see_each_target_position(self):
        model, _ = tiny_model()
        prompt = mx.array([4, 11, 3, 5])

        def next_integer(tokens, logits):
            forced = (int(tokens[-1].item()) + 1) % 64
            return mx.where(mx.arange(64) == forced, 0.0, -mx.inf)[None]

        actual = list(
            mtp_speculative_generate_step(
                prompt[-2:],
                model,
                max_tokens=7,
                prefill_step_size=1,
                logits_processors=[next_integer],
                logits_processor_tokens=prompt,
            )
        )
        self.assertEqual([int(t) for t, _, _ in actual], list(range(6, 13)))

    def test_stream_eos_and_consumer_close_release_speculative_entries(self):
        model, _ = tiny_model(True)
        backend = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="63"))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="0")
        cache = cache_for(model)
        output = list(
            stream_generate(
                model,
                tokenizer,
                [1, 2, 3],
                mtp=True,
                prompt_cache=cache,
                kv_bits=8,
                quantized_kv_start=0,
            )
        )
        self.assertEqual(output[-1].finish_reason, "stop")
        self.assertEqual([c.offset for c in cache], [3] * 5)
        tokenizer.eos_token = "63"
        cache = cache_for(model)
        generator = stream_generate(
            model,
            tokenizer,
            [1, 2, 3],
            mtp=True,
            prompt_cache=cache,
            kv_bits=8,
            quantized_kv_start=0,
        )
        next(generator)
        generator.close()
        self.assertEqual([c.offset for c in cache], [3] * 5)

    def test_sparse_verify_matches_sequential_attention(self):
        from mlx_lm.models import deepseek_v32

        for dtype in (mx.float32, mx.bfloat16):
            for bits in (None, 8):
                with self.subTest(dtype=dtype, bits=bits):
                    model, _ = tiny_model()
                    model.set_dtype(dtype)
                    layer = model.layers[0].self_attn
                    prompt = mx.random.normal((1, 20, 128)).astype(dtype)
                    tail = mx.random.normal((1, 2, 128)).astype(dtype)
                    first, second = model.make_cache()[0], model.make_cache()[0]
                    if bits:
                        first = first.to_quantized(bits=8, group_size=64)
                        second = second.to_quantized(bits=8, group_size=64)
                    layer(prompt, cache=first)
                    layer(prompt, cache=second)
                    expected = mx.concatenate(
                        [layer(tail[:, i : i + 1], cache=first)[0] for i in range(2)],
                        axis=1,
                    )
                    mask = mx.arange(22)[None, :] <= mx.arange(20, 22)[:, None]
                    with patch.object(deepseek_v32, "_SMALL_L_GATHER", True):
                        actual = layer(tail, mask=mask, cache=second)[0]
                    tolerance = 2e-5 if dtype == mx.float32 else 0.025
                    self.assertTrue(
                        mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)
                    )

    def test_mixed_quantized_checkpoint_roundtrip_and_generation(self):
        model, config = tiny_model()

        def quantization(path, module):
            if not hasattr(module, "to_quantized") or path.endswith(".gate"):
                return False
            return {"bits": 4 if ".switch_mlp." in path else 8, "group_size": 64}

        nn.quantize(model, class_predicate=quantization)
        config["quantization"] = {"group_size": 64, "bits": 8}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config))
            mx.save_safetensors(
                str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
            )
            restored, _ = load_model(path)
            prompt = mx.arange(20) % 64
            expected = [
                int(t)
                for t, _ in generate_step(
                    prompt,
                    restored,
                    max_tokens=12,
                    kv_bits=8,
                    quantized_kv_start=0,
                    prefill_step_size=8,
                )
            ]
            actual = [
                int(t)
                for t, _, _ in mtp_speculative_generate_step(
                    prompt,
                    restored,
                    max_tokens=12,
                    kv_bits=8,
                    quantized_kv_start=0,
                    prefill_step_size=8,
                )
            ]
            self.assertEqual(actual, expected)
            self.assertTrue(
                mx.allclose(model(prompt[None]), restored(prompt[None]), atol=1e-5)
            )

    def test_separate_head_load_matches_original_forward(self):
        model, config = tiny_model()
        weights = dict(tree_flatten(model.parameters()))
        main = {
            key: value for key, value in weights.items() if not key.startswith("mtp.")
        }
        raw = {}
        prefix = "model.layers.4."
        for key, value in weights.items():
            if key.startswith("mtp.layer."):
                key = prefix + key.removeprefix("mtp.layer.")
                if ".switch_mlp." in key:
                    for expert in range(4):
                        raw[key.replace(".switch_mlp.", f".experts.{expert}.")] = value[
                            expert
                        ]
                    continue
            elif key.startswith("mtp.shared_head."):
                key = (
                    prefix + "shared_head.norm." + key.removeprefix("mtp.shared_head.")
                )
            elif key.startswith("mtp."):
                key = prefix + key.removeprefix("mtp.")
            else:
                continue
            raw[key] = value
        attn = prefix + "self_attn."
        raw[attn + "kv_b_proj.weight"] = mx.concatenate(
            [
                raw.pop(attn + "embed_q.weight").swapaxes(-1, -2),
                raw.pop(attn + "unembed_out.weight"),
            ],
            axis=1,
        ).reshape(-1, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            head = path / "head"
            head.mkdir()
            (path / "config.json").write_text(json.dumps(config))
            (head / "mtp-config.json").write_text(
                json.dumps({"model_args": {"hidden_size": 128}})
            )
            mx.save_safetensors(str(path / "model.safetensors"), main)
            mx.save_safetensors(str(head / "mtp.safetensors"), raw)
            restored, _ = load_model(path, mtp_path=head)
            hidden = mx.random.normal((1, 4, 128))
            tokens = mx.array([[4, 5, 6, 7]])
            self.assertTrue(
                mx.allclose(
                    model.mtp_forward(hidden, tokens),
                    restored.mtp_forward(hidden, tokens),
                    atol=1e-5,
                )
            )
            (head / "mtp-config.json").write_text(
                json.dumps({"model_args": {"hidden_size": 256}})
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_model(path, mtp_path=head)


if __name__ == "__main__":
    unittest.main()
