"""Words past the tokenizer's 512-token cut-off must not be silently dropped.

Chunks are built by word count but encoded with ``truncation=True``; words the
tokenizer cut get no ``word_id`` and so no keep decision. They are kept verbatim.
"""

from __future__ import annotations

import pytest

from headroom.transforms import kompress_compressor as kc

N_WORDS = 40
SCORED = 25  # the fake tokenizer "truncates" after this many words


class _Enc(dict):
    def __init__(self, word_ids_per_row, **kw):
        super().__init__(**kw)
        self._wids = word_ids_per_row

    def word_ids(self, batch_index=0):
        return self._wids[batch_index]


class _Tok:
    def __call__(self, chunks, **kw):
        batched = bool(chunks) and isinstance(chunks[0], list)
        rows = chunks if batched else [chunks]
        wids = [[i if i < SCORED else None for i in range(len(r))] for r in rows]
        n = max(len(r) for r in rows)
        return _Enc(
            wids,
            input_ids=[[0] * n for _ in rows],
            attention_mask=[[1] * n for _ in rows],
        )


class _Model:
    # The model drops every scored word, so only unscored words can survive.
    def get_keep_mask(self, input_ids, attention_mask):
        return [[False] * len(input_ids[0]) for _ in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[0.0] * len(input_ids[0]) for _ in input_ids]


@pytest.fixture
def comp(monkeypatch):
    monkeypatch.setenv("HEADROOM_KOMPRESS_MUST_KEEP", "0")
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_Model(), _Tok(), "onnx"))
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
    c = kc.KompressCompressor(kc.KompressConfig(min_input_words=10, enable_ccr=False))
    c.config.chunk_words = 1000
    return c


def test_keep_unscored_words_helper():
    kept: set[int] = set()
    kc._keep_unscored_words(kept, [None, 0, 1, 1, None], 4, 10)
    assert kept == {12, 13}


def test_single_path_keeps_truncated_tail(comp, monkeypatch):
    monkeypatch.setattr(comp, "_should_batch_single_content", lambda *a, **k: False)
    words = [f"w{i}" for i in range(N_WORDS)]
    out = comp.compress(" ".join(words)).compressed.split()
    for i in range(SCORED, N_WORDS):
        assert f"w{i}" in out
    assert "w0" not in out


def test_batch_path_keeps_truncated_tail(comp):
    words = [f"w{i}" for i in range(N_WORDS)]
    out = comp.compress_batch([" ".join(words)])[0].compressed.split()
    for i in range(SCORED, N_WORDS):
        assert f"w{i}" in out
    assert "w0" not in out
