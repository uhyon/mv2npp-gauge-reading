"""The pose model on ONNX Runtime alone, without PyTorch or Ultralytics.

``OnnxPose(path).detect(bgr)`` returns the same ``GaugeDetection`` list as Ultralytics'
``YOLO(path).predict`` turned by ``detections_from_result``: the same letterbox (grey 114,
centred, linear resize), class-agnostic NMS at IoU 0.7, boxes and keypoints scaled back to the
image and clipped to it. Used by the desktop app so its Windows build stays small.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gauge_reader.keypoints.yolo import GaugeDetection, detections_from_arrays

IOU = 0.7
MAX_DET = 300


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping the aspect ratio and pad to ``size`` square (Ultralytics LetterBox)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = IOU) -> list[int]:
    order, keep = np.argsort(-scores), []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        ov = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou]
    return keep


def decode(out: np.ndarray, conf: float, nkpt: int) -> tuple[np.ndarray, ...]:
    """Raw output (4 + 1 + 3 nkpt, anchors) -> boxes xyxy, scores, kpts (N, nkpt, 3), in the
    letterboxed image."""
    p = out.T
    keep = p[:, 4] > conf
    p = p[keep]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    scores = p[:, 4]
    kpts = p[:, 5:5 + 3 * nkpt].reshape(-1, nkpt, 3)
    idx = nms(boxes, scores)[:MAX_DET]
    return boxes[idx], scores[idx], kpts[idx]


class OnnxPose:
    def __init__(self, path: str | Path, imgsz: int = 640, threads: int | None = None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.nkpt = int(meta.get("kpt_shape", "[7, 3]").strip("[]").split(",")[0])
        self.imgsz = imgsz

    def detect(self, img: np.ndarray, conf: float = 0.1) -> list[GaugeDetection]:
        """Gauges in a BGR image, most confident first."""
        lb, r, left, top = letterbox(img, self.imgsz)
        x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255
        out = self.session.run(None, {self.input: x})[0][0]
        boxes, scores, kpts = decode(out, conf, self.nkpt)
        if not len(boxes):
            return []
        h, w = img.shape[:2]
        boxes = (boxes - [left, top, left, top]) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        xy = (kpts[:, :, :2] - [left, top]) / r
        xy[..., 0] = xy[..., 0].clip(0, w)
        xy[..., 1] = xy[..., 1].clip(0, h)
        return detections_from_arrays(boxes, scores, xy, kpts[:, :, 2])
