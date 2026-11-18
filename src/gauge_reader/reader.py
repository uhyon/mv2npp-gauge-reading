"""One call from a photo to a reading: ``GaugeReader(weights).read(image)``.

Stages (docs/REQUIREMENTS.md):

1. detection: the pose model on the whole photo; the most confident gauge is kept;
2. keypoints on a close-up of that gauge with a grey margin of ``CROP_PAD`` (the framing of the
   MeterFL crops read with --pad 0.15): the whole photo when the box covers at least
   ``WHOLE_IMAGE_SHARE`` of it, else the box plus ``CROP_MARGIN`` per side. (On MeterFL,
   cropping close-ups to the box instead let 2–5x more photos fall to RETAKE.) Read
   upright and turned by +/- ``tta_angle``; the needle fraction is the median of the three and
   is read on the dial face when the model has the scale-arc keypoints (``reading.perspective``);
3. RETAKE checks: no gauge, a scale arc under ``MIN_SWEEP_DEG`` (keypoints collapsed), or the
   three readings more than ``retake_spread`` % of scale apart (``predict.RETAKE_SPREAD``);
4. scale range and unit: given by the caller (a known instrument) or read from the dial numbers
   with OCR (``reading.scale_ocr``); no range is a RETAKE too;
5. value = min + fraction x span, and, with an alarm band, LOW / HIGH / NORMAL.

``Reading.alarm`` is RETAKE whenever no value is given, so a round can ask for another photo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gauge_reader.keypoints.yolo import GaugeDetection, detections_from_result, pad_image
from gauge_reader.predict import CLIP, MIN_SWEEP_DEG, RETAKE_SPREAD, TTA_ANGLE, rotate
from gauge_reader.reading.geometry import AlarmBand, GaugeKeypoints, check_alarm
from gauge_reader.reading.scale_ocr import Label, dial_crop, read_scale

CROP_MARGIN = 0.05   # real context around the box, share of its size per side
WHOLE_IMAGE_SHARE = 0.4  # a box this large (share of the photo) is read on the whole photo
CROP_PAD = 0.15      # then a grey margin, as for the MeterFL crops (predict_readings --pad)
OCR_CROP = 1.6       # OCR window, in scale radii around the centre


@dataclass
class Reading:
    status: str                      # ok, no_gauge, bad_geometry, retake, no_scale
    alarm: str = "RETAKE"            # NORMAL, LOW, HIGH ("" without a band) or RETAKE
    value: float | None = None
    unit: str = ""
    scale_min: float | None = None
    scale_max: float | None = None
    scale_source: str = ""           # "given" or "ocr"
    fraction: float | None = None    # median needle fraction (0 = min tick, 1 = max tick)
    spread: float | None = None      # max - min of the three fractions, % of scale
    box: tuple[float, float, float, float] | None = None
    box_conf: float | None = None
    box_share: float | None = None   # box area / photo area
    keypoints: GaugeKeypoints | None = None
    arc: tuple = ()
    labels: list[Label] = field(default_factory=list)   # OCR labels on the fitted scale line
    ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        k = self.keypoints
        d = {"status": self.status, "alarm": self.alarm, "value": self.value, "unit": self.unit,
             "scale_min": self.scale_min, "scale_max": self.scale_max,
             "scale_source": self.scale_source, "fraction": self.fraction, "spread": self.spread,
             "box": self.box, "box_conf": self.box_conf,
             "ms": {s: round(v, 1) for s, v in self.ms.items()}}
        if k is not None:
            d["keypoints"] = {"center": k.center, "tip": k.tip, "scale_min": k.scale_min,
                              "scale_max": k.scale_max}
        return d


def crop_box(img: np.ndarray, box, margin: float = CROP_MARGIN, value: int = 114,
             square: bool = False):
    """The box plus ``margin`` of its size per side; grey where that leaves the image.
    ``square``: a square window around the box centre, side = longer box side x (1 + 2 margin).
    Returns the crop and the (x, y) of its top-left corner in the image."""
    x0, y0, x1, y1 = box
    if square:
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    X0, Y0 = int(np.floor(x0 - mx)), int(np.floor(y0 - my))
    X1, Y1 = int(np.ceil(x1 + mx)), int(np.ceil(y1 + my))
    h, w = img.shape[:2]
    out = np.full((Y1 - Y0, X1 - X0) + img.shape[2:], value, dtype=img.dtype)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X1, w), min(Y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = img[sy0:sy1, sx0:sx1]
    return out, (X0, Y0)


def _move(det: GaugeDetection, dx: float, dy: float) -> GaugeDetection:
    mv = lambda p: (p[0] + dx, p[1] + dy)
    k = det.keypoints
    x0, y0, x1, y1 = det.box
    return GaugeDetection((x0 + dx, y0 + dy, x1 + dx, y1 + dy), det.box_conf,
                          GaugeKeypoints(mv(k.center), mv(k.tip), mv(k.scale_min), mv(k.scale_max)),
                          det.kpt_conf, tuple(mv(p) for p in det.arc))


class GaugeReader:
    def __init__(self, weights: str | Path | None = None, *, model=None, ocr=True,
                 tta_angle: float | None = TTA_ANGLE, retake_spread: float | None = RETAKE_SPREAD,
                 conf: float = 0.1, imgsz: int = 640, device=None, min_sweep: float = MIN_SWEEP_DEG,
                 clip: float = CLIP, crop_margin: float = CROP_MARGIN, crop_pad: float = CROP_PAD,
                 whole_share: float = WHOLE_IMAGE_SHARE, square: bool = False):
        """``model``: a loaded Ultralytics pose model (else loaded from ``weights``).
        ``ocr``: True (rapidocr), False (the range must be given) or an OCR callable.
        ``tta_angle`` None reads upright only (no spread check).
        Framing of a gauge smaller than ``whole_share`` of the photo: the box plus ``crop_margin``
        of real context per side (``square``: a square window), then ``crop_pad`` grey margin;
        ``whole_share`` > 1 reads every photo whole."""
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(weights))
        self.model = model
        if ocr is True:
            from gauge_reader.reading.scale_ocr import RapidOcr
            ocr = RapidOcr()
        self.ocr = ocr or None
        self.tta_angle, self.retake_spread = tta_angle, retake_spread
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.min_sweep, self.clip = min_sweep, clip
        self.crop_margin, self.crop_pad = crop_margin, crop_pad
        self.whole_share, self.square = whole_share, square

    def _detect(self, img: np.ndarray) -> list[GaugeDetection]:
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                 verbose=False)[0]
        return detections_from_result(res)

    def read(self, image, scale: tuple[float, float, str] | None = None,
             band: AlarmBand | None = None) -> Reading:
        """``image``: a path or a BGR array. ``scale``: (min, max, unit) when known; else OCR."""
        img = cv2.imread(str(image), cv2.IMREAD_COLOR) if isinstance(image, (str, Path)) else image
        if img is None:
            raise ValueError(f"cannot read image {image}")
        ms: dict[str, float] = {}
        t = time.perf_counter()
        padded, px, py = pad_image(img, CROP_PAD)   # also the upright pass of a close-up
        dets0 = self._detect(padded)
        dets = [_move(d, -px, -py) for d in dets0]
        ms["detect"] = (time.perf_counter() - t) * 1000
        if not dets:
            return Reading("no_gauge", ms=ms)
        top = dets[0]
        out = Reading("ok", box=top.box, box_conf=top.box_conf, ms=ms)

        # keypoints on the crop: upright and turned
        x0, y0, x1, y1 = top.box
        h, w = img.shape[:2]
        out.box_share = (x1 - x0) * (y1 - y0) / (w * h)
        closeup = out.box_share >= self.whole_share
        if closeup:
            crop, (ox, oy) = img, (0, 0)          # the photo is already a close-up
            pad = CROP_PAD                        # as the detection pass it reuses
        else:
            crop, (ox, oy) = crop_box(img, top.box, self.crop_margin, square=self.square)
            pad = self.crop_pad
        _, px, py = pad_image(crop, pad)
        ox, oy = ox - px, oy - py              # turned first, then padded (as predict_readings)
        angles = [0.0] if not self.tta_angle else [0.0, self.tta_angle, -self.tta_angle]
        fracs, upright = [], None
        t = time.perf_counter()
        for ang in angles:
            if ang == 0.0 and closeup:
                ds = dets0                          # same input as the detection pass
            else:
                ds = self._detect(pad_image(rotate(ang)(crop, {}) if ang else crop, pad)[0])
            if not ds:
                continue
            d = ds[0]
            if ang == 0.0:
                upright = _move(d, ox, oy)
            try:
                if d.sweep() >= self.min_sweep:
                    fracs.append(min(max(d.fraction(), -self.clip), 1.0 + self.clip))
            except ValueError:
                pass
        ms["keypoints"] = (time.perf_counter() - t) * 1000
        if upright is None:
            upright = top
        out.keypoints, out.arc = upright.keypoints, upright.arc
        try:
            sweep_ok = upright.sweep() >= self.min_sweep
        except ValueError:
            sweep_ok = False
        if not sweep_ok or not fracs:
            out.status = "bad_geometry"
            return out
        out.fraction = float(np.median(fracs))
        if len(angles) > 1:
            out.spread = (max(fracs) - min(fracs)) * 100 if len(fracs) >= 2 else 99.0
            if self.retake_spread is not None and out.spread > self.retake_spread:
                out.status = "retake"
                return out

        # scale range and unit
        if scale is not None:
            out.scale_min, out.scale_max, out.unit = float(scale[0]), float(scale[1]), scale[2]
            out.scale_source = "given"
        elif self.ocr is not None:
            t = time.perf_counter()
            win, (dx, dy) = dial_crop(img, upright.keypoints, OCR_CROP)
            boxes = [(s, [(x + dx, y + dy) for x, y in poly], c) for s, poly, c in self.ocr(win)]
            sr = read_scale(boxes, upright.keypoints, upright.dial_fit())
            ms["ocr"] = (time.perf_counter() - t) * 1000
            out.unit, out.labels = sr.unit, sr.inliers
            if sr.ok:
                out.scale_min, out.scale_max, out.scale_source = sr.scale_min, sr.scale_max, "ocr"
        if out.scale_min is None or out.scale_max is None:
            out.status = "no_scale"
            return out

        out.value = out.scale_min + out.fraction * (out.scale_max - out.scale_min)
        out.alarm = check_alarm(out.value, band, 1.0) if band is not None else ""
        return out
