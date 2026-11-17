import numpy as np
import pytest
from test_predict import FakeModel

from gauge_reader.reader import GaugeReader, crop_box
from gauge_reader.reading.geometry import AlarmBand


def _img():
    return np.full((100, 120, 3), 200, np.uint8)


def test_crop_box_pads_outside_the_image():
    img = np.zeros((10, 10, 3), np.uint8)
    crop, (x, y) = crop_box(img, (0, 0, 10, 10), margin=0.2)
    assert (x, y) == (-2, -2) and crop.shape == (14, 14, 3)
    assert crop[0, 0, 0] == 114 and crop[5, 5, 0] == 0


def test_read_with_given_scale_and_alarm():
    # detection pass, then upright / +25 / -25 on the crop
    model = FakeModel([(270, 0.5), (270, 0.50), (270, 0.52), (270, 0.49)])
    r = GaugeReader(model=model, ocr=False).read(_img(), scale=(0, 10, "bar"),
                                                 band=AlarmBand(2, 4))
    assert r.status == "ok" and r.value == pytest.approx(5.0, abs=1e-3)
    assert r.spread == pytest.approx(3.0, abs=0.01) and r.alarm == "HIGH"
    assert r.unit == "bar" and r.scale_source == "given"
    assert set(r.ms) == {"detect", "keypoints"}


def test_read_retake_when_turned_readings_disagree():
    model = FakeModel([(270, 0.5), (270, 0.5), (270, 0.7), (270, 0.5)])
    r = GaugeReader(model=model, ocr=False).read(_img(), scale=(0, 10, "bar"))
    assert r.status == "retake" and r.value is None and r.alarm == "RETAKE"


def test_read_without_range_or_ocr_is_no_scale():
    model = FakeModel([(270, 0.5)] * 4)
    r = GaugeReader(model=model, ocr=lambda img: []).read(_img())
    assert r.status == "no_scale" and r.alarm == "RETAKE" and "ocr" in r.ms


def test_bad_geometry():
    model = FakeModel([(40, 0.5)] * 4)
    r = GaugeReader(model=model, ocr=False).read(_img(), scale=(0, 10, "bar"))
    assert r.status == "bad_geometry"
