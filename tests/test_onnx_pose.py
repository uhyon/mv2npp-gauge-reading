import numpy as np

from gauge_reader.keypoints.onnx_pose import decode, letterbox, nms


def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((100, 200, 3), np.uint8)
    out, r, left, top = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 3.2 and left == 0 and top == 160
    assert out[0, 0, 0] == 114 and out[320, 320, 0] == 0


def test_nms_drops_overlaps_and_keeps_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    assert nms(boxes, np.array([0.5, 0.9, 0.7])) == [1, 2]


def test_decode_reads_boxes_scores_and_keypoints():
    nkpt, n = 7, 3
    out = np.zeros((5 + 3 * nkpt, n), np.float32)
    out[:4, 0] = [50, 60, 20, 40]          # cx, cy, w, h
    out[4] = [0.9, 0.05, 0.3]               # second anchor below conf
    out[:4, 2] = [200, 200, 10, 10]
    out[5:, 0] = np.tile([1, 2, 0.8], nkpt)
    boxes, scores, kpts = decode(out, 0.1, nkpt)
    assert np.allclose(boxes[0], [40, 40, 60, 80]) and list(scores) == [np.float32(0.9), np.float32(0.3)]
    assert kpts.shape == (2, nkpt, 3) and np.allclose(kpts[0, 0], [1, 2, 0.8])
