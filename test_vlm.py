import numpy as np
import pytest
from PIL import Image

import vlm


def frame(tmp_path, name, size=(80, 80)):
    path = tmp_path / name
    Image.new("RGB", size, (100, 110, 120)).save(path)
    return path


def test_tile_outlines_a_synthetic_mask(tmp_path):
    path = frame(tmp_path, "frame.jpg")
    mask = np.zeros((80, 80), bool)
    mask[30:50, 30:50] = True
    out = vlm.tile(path, mask, "#0 A f0")
    pixels = np.asarray(out)
    assert out.size == (vlm.TILE, vlm.TILE)
    assert np.any(np.all(pixels[22:] == (255, 0, 255), axis=2))
    assert np.any(np.all(pixels[22:] == (100, 110, 120), axis=2))


def test_verify_parses_verdicts_converts_confidence_and_caches(tmp_path, monkeypatch):
    frames = {"A": [frame(tmp_path, "a.jpg")], "B": [frame(tmp_path, "b.jpg")]}
    mask = np.zeros((80, 80), bool)
    mask[30:50, 30:50] = True
    cands = [{"A": {0: mask}, "B": {}}, {"A": {}, "B": {0: mask}}]
    calls = []

    def fake_ask(prompt, images, schema, name):
        calls.append((prompt, images, schema, name))
        return {"candidates": [
            {"id": 1, "real": False, "name": "lamp", "description": "Beside the sofa.", "confidence": 0.8},
            {"id": 0, "real": True, "name": "chair", "description": "In front of the desk.", "confidence": 0.9}]}

    monkeypatch.setattr(vlm, "ask", fake_ask)
    cache = tmp_path / "nested" / "verify.json"
    expected = {0: {"real": True, "name": "chair", "description": "In front of the desk.", "p_real": 0.9},
                1: {"real": False, "name": "lamp", "description": "Beside the sofa.", "p_real": pytest.approx(0.2)}}
    assert vlm.verify(cands, frames, cache) == expected
    assert len(calls) == 1
    assert len(calls[0][1]) == 3
    assert "0, 1" in calls[0][0]
    assert vlm.verify(cands, frames, cache) == expected
    assert len(calls) == 1


def test_verify_keeps_a_skipped_candidate_undecided(tmp_path, monkeypatch):
    frames = {"A": [frame(tmp_path, "a.jpg")], "B": [frame(tmp_path, "b.jpg")]}
    mask = np.ones((80, 80), bool)
    monkeypatch.setattr(vlm, "ask", lambda *args: {"candidates": []})
    a = vlm.verify([{"A": {0: mask}, "B": {}}], frames, tmp_path / "cache.json")
    assert a[0]["real"] and a[0]["p_real"] == 0.5
