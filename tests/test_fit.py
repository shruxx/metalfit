"""The arithmetic against what was measured on a 48 GB M5 Pro, macOS 27.0.1, llama.cpp build 782.

Every number here is a fact from a run of serve/server.py or llama-server, not a guess.  If a change to fit.py
breaks one of these, the change is wrong until the Mac says otherwise.
"""
import unittest

from metalfit.fit import GIB, COMPUTE_RESERVE, Model, kv_bytes, kv_heads, kv_layout, plan

STOCK_WS = 38338 * 1024 * 1024          # what llama-server reports at the stock Metal limit: 37.44 GiB
RAISED_WS = 41984 * 1024 * 1024         # after sudo sysctl iogpu.wired_limit_mb=41984: 41.0 GiB


def kolibri(must_hold_gib: float) -> Model:
    """Kolibri-1: 50 blocks, 4:1 sliding to full attention (40 sliding of 513, 10 full), 4 KV heads, 128/128."""
    return Model(path=None, shards=(), arch="kolibri1", n_layers=50,
                 file_bytes=int(must_hold_gib * GIB), lazy_bytes=0, must_hold=int(must_hold_gib * GIB),
                 kv_layers=10, swa_layers=40, swa_window=513, n_head_kv=4, k_len=128, v_len=128, ssm_bytes=0)


def qwen_q2_0() -> Model:
    """Qwen3.8-Flash-Next Q2_0: 48 blocks, every 4th full attention (12 with a KV cache, 36 recurrent),
    2 KV heads, 256/256.  66.4 GB on disk of which 28.8 GB is the lazy PLE table, so 35.0 GiB to hold."""
    return Model(path=None, shards=(), arch="qwen4exp", n_layers=48,
                 file_bytes=int(66.41e9), lazy_bytes=int(28.80e9), must_hold=int(37.61e9),
                 kv_layers=12, swa_layers=0, swa_window=0, n_head_kv=2, k_len=256, v_len=256,
                 ssm_bytes=36 * (6144 * 128 + 6144 * 3) * 4)


class QwenQ2_0(unittest.TestCase):
    """Measured: -c 65536 runs at 34.0 tok/s on the stock limit, -c 131072 dies with
    kIOGPUCommandBufferCallbackErrorOutOfMemory there and runs at 33.8 once the limit is raised."""

    def test_fits_at_64k_on_the_stock_limit(self):
        p = plan(qwen_q2_0(), 65536, STOCK_WS)
        self.assertTrue(p.fits_whole, p.advice)
        self.assertEqual(p.n_gpu_layers, 49)

    def test_does_not_fit_at_128k_on_the_stock_limit(self):
        p = plan(qwen_q2_0(), 131072, STOCK_WS)
        self.assertFalse(p.fits_whole, p.advice)
        self.assertEqual(p.max_ctx, 65536)          # and it says what would fit

    def test_fits_at_128k_once_the_limit_is_raised(self):
        self.assertTrue(plan(qwen_q2_0(), 131072, RAISED_WS).fits_whole)

    def test_the_lazy_table_is_not_counted(self):
        m = qwen_q2_0()
        self.assertAlmostEqual(m.must_hold / GIB, 35.0, delta=0.1)
        self.assertGreater(m.file_bytes - m.must_hold, 26 * GIB)


class KolibriQuants(unittest.TestCase):
    """Measured at -c 65536 on the stock limit: Q3_K_M (34.9 GiB) runs whole at 56.7-60.2 tok/s, Q4_K_M
    (44.2 GiB) cannot and runs as a split at ~32."""

    def test_q3_k_m_fits_whole_on_the_stock_limit(self):
        p = plan(kolibri(34.9), 65536, STOCK_WS)
        self.assertTrue(p.fits_whole, p.advice)
        self.assertEqual(p.n_gpu_layers, 51)

    def test_q4_k_m_never_fits(self):
        p = plan(kolibri(44.2), 65536, RAISED_WS)
        self.assertFalse(p.fits_whole)
        self.assertEqual(p.max_ctx, 0)              # not at any context, even with the limit raised
        self.assertIn("does not fit whole at any context", p.advice)

    def test_sliding_window_layers_barely_cost_anything(self):
        """40 of Kolibri's 50 layers cache only 513 tokens, so 128K context costs little: that is why its KV
        measured ~2.5 GiB of footprint rather than the ~7 GB a full-attention model of this shape would need."""
        m = kolibri(34.9)
        self.assertLess(kv_bytes(m, 131072), 1.6 * GIB)
        full = Model(**{**m.__dict__, "kv_layers": 50, "swa_layers": 0, "swa_window": 0})
        self.assertGreater(kv_bytes(full, 131072), 4 * kv_bytes(m, 131072))


class HybridInterval(unittest.TestCase):
    """qwen35 / qwen35moe give only full_attention_interval; llama.cpp makes every 4th layer attention and the
    MTP layers too.  Counted as every layer, Qwen3.8-27B (65 blocks, 1 MTP) needed 17.27 GiB of KV at 128K."""

    def test_every_fourth_layer_and_the_mtp_layer(self):
        self.assertEqual(kv_layout(65, interval=4, nextn=1), (17, 0))   # Qwen3.8-27B: 16 + 1 MTP
        self.assertEqual(kv_layout(40, interval=4), (10, 0))            # Qwen3.6-35B-A3B without MTP

    def test_a_pattern_or_ratios_still_win(self):
        self.assertEqual(kv_layout(50, pattern=[True] * 40 + [False] * 10, interval=4), (10, 40))
        self.assertEqual(kv_layout(48, ratios=[0, 0, 0, 4] * 12, interval=4), (12, 0))


class PerLayerHeads(unittest.TestCase):
    """gemma-4-26B-A4B gives head_count_kv per layer (8 sliding, 2 full) and its own key/value length for the
    sliding layers (256 against 512).  Read as one number the list was 0 heads and 0.00 GiB of KV."""

    def test_heads_split_by_the_pattern(self):
        pattern = ([True] * 5 + [False]) * 5
        heads = ([8] * 5 + [2]) * 5
        self.assertEqual(kv_heads(heads, pattern), (2, 8))
        self.assertEqual(kv_heads(4), (4, 0))

    def test_gemma4_kv_at_128k(self):
        m = Model(path=None, shards=(), arch="gemma4", n_layers=30, file_bytes=0, lazy_bytes=0, must_hold=0,
                  kv_layers=5, swa_layers=25, swa_window=1024, n_head_kv=2, k_len=512, v_len=512, ssm_bytes=0,
                  n_head_kv_swa=8, k_len_swa=256, v_len_swa=256)
        # 5 full layers * 2 heads * 1024 * 34/32 bytes * 131072 = 1.33 GiB, 25 sliding ones add 0.10
        self.assertAlmostEqual(kv_bytes(m, 131072) / GIB, 1.43, delta=0.01)


class Reserve(unittest.TestCase):
    def test_the_reserve_is_what_the_two_measurements_bracket(self):
        """Qwen Q2_0 at the stock limit leaves 2.44 GiB beside its weights; -c 65536 (0.80 GiB of KV) runs and
        -c 131072 (1.60 GiB) does not, so everything else is above 0.84 and at most 1.64 GiB."""
        m = qwen_q2_0()
        spare = STOCK_WS - m.must_hold
        self.assertGreater(COMPUTE_RESERVE, spare - kv_bytes(m, 131072))
        self.assertLessEqual(COMPUTE_RESERVE, spare - kv_bytes(m, 65536))


if __name__ == "__main__":
    unittest.main(verbosity=2)
