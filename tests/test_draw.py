import numpy as np
from test_predict import FakeModel
from test_reader import _img

from gauge_reader.draw import draw_reading, label_text
from gauge_reader.reader import GaugeReader, Reading
from gauge_reader.reading.geometry import AlarmBand


def test_draw_reading_and_label():
    model = FakeModel([(270, 0.5), (270, 0.5), (270, 0.51), (270, 0.49)])
    r = GaugeReader(model=model, ocr=False).read(_img(), scale=(0, 10, "bar"), band=AlarmBand(2, 4))
    assert label_text(r) == "5 bar  HIGH"
    img = _img()
    out = draw_reading(img, r)
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(img, _img())          # the input is not changed


def test_draw_retake_without_gauge():
    r = Reading("no_gauge")
    assert label_text(r) == "RETAKE (no_gauge)"
    assert draw_reading(_img(), r).shape == _img().shape
