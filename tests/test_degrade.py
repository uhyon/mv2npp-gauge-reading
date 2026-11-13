import cv2
import numpy as np
import pytest

from gauge_reader.eval.degrade import (
    BLUR_PX,
    CONDITIONS,
    SPECKLE_FRACTION,
    degrade,
    rng_for,
)


def _scene(h=240, w=320):
    img = np.zeros((h, w, 3), np.uint8)
    img[(np.arange(h)[:, None] // 16 + np.arange(w)[None, :] // 16) % 2 == 1] = 200
    cv2.circle(img, (w // 2, h // 2), 60, (90, 160, 230), -1)
    return img


@pytest.mark.parametrize("cond", CONDITIONS)
def test_fixed_seed_same_image(cond):
    img = _scene()
    a = degrade(img, cond, 2, rng_for("voc_1", cond, 2))
    b = degrade(img, cond, 2, rng_for("voc_1", cond, 2))
    c = degrade(img, cond, 2, rng_for("voc_2", cond, 2))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert cond == "motion_blur" or not np.array_equal(a, c)  # blur differs only in direction
    assert not np.array_equal(a, img)


def test_low_light_darker_with_severity():
    img = np.full((100, 100, 3), 128, np.uint8)
    means = [degrade(img, "low_light", s, rng_for("x", "low_light", s)).mean() for s in (1, 2, 3)]
    assert 128 > means[0] > means[1] > means[2] > 0
    # gamma 1.8 on 0.5 grey is about 0.287 -> 73
    assert means[0] == pytest.approx(255 * 0.5 ** 1.8, abs=2)


@pytest.mark.parametrize("sev", [1, 2, 3])
def test_speckle_hit_count(sev):
    img = np.zeros((200, 300, 3), np.uint8)
    out = degrade(img, "radiation_speckle", sev, rng_for("x", "radiation_speckle", sev))
    lit = int((out.max(axis=2) == 255).sum())
    n = 200 * 300 * SPECKLE_FRACTION[sev]
    assert 0.8 * 0.9 * n <= lit <= 0.9 * n + 0.1 * n * 9 + 2


def test_blur_stronger_with_severity():
    # Averaging L pixels of white noise divides its variance by about L, in any direction.
    noise = np.random.default_rng(0).integers(0, 256, (200, 200, 3)).astype(np.uint8)
    std = [degrade(noise, "motion_blur", s, rng_for("x", "motion_blur", s)).astype(float).std()
           for s in (1, 2, 3)]
    assert noise.std() > std[0] > std[1] > std[2]
    assert list(BLUR_PX.values()) == [7, 15, 25]


def test_bad_arguments():
    with pytest.raises(ValueError):
        degrade(_scene(), "fog", 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        degrade(_scene(), "low_light", 4, np.random.default_rng(0))
