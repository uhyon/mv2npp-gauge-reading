"""Desktop app for a gauge round: open photos or a folder, read every gauge, save a CSV report.

    uv pip install -e ".[desktop,ocr]"
    uv run python desktop/app.py                 # weights/best.onnx, else GAUGE_WEIGHTS
    uv run python desktop/app.py photos/         # open a folder (or photos) at start

Runs on ONNX Runtime alone (``gauge_reader.keypoints.onnx_pose``): no PyTorch, so the Windows
build (``desktop/build_windows.md``) stays small. A photo named ``TAG__anything.jpg`` gets the
registered gauge ``TAG`` (configs/gauges.csv): range, unit and alarm band from the registry;
any other photo reads its range with OCR. The tag can be changed in the right panel.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FROZEN = getattr(sys, "frozen", False)
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
if not FROZEN:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QByteArray, QObject, Qt, QThread, Signal, Slot  # noqa: E402
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QImage, QPixmap  # noqa: E402
from PySide6.QtSvgWidgets import QSvgWidget  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHeaderView, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QSizePolicy, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar,
    QVBoxLayout, QWidget)

from gauge_reader.draw import draw_reading  # noqa: E402
from gauge_reader.reading.perspective import frontal_view  # noqa: E402
from gauge_reader.registry import load_registry  # noqa: E402
from meter import STATUS_COLORS, meter_svg, status_of  # noqa: E402

APP_NAME = "mv2npp 계기 순시 점검"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXAMPLES = ROOT / "demo" / "examples"
REGISTRY = load_registry(ROOT / "configs" / "gauges.csv")
NO_TAG = "미등록 (OCR로 범위 판독)"
SOURCE = {"given": "등록부", "ocr": "OCR"}
STATUS_KO = {"ok": "판독", "retake": "재촬영: 회전 판독 차이 5% 초과",
             "bad_geometry": "재촬영: 눈금 키포인트 불량", "no_gauge": "재촬영: 계기 없음",
             "no_scale": "재촬영: 눈금 범위를 못 읽음"}
COLUMNS = ["사진", "태그", "판독값", "단위", "눈금", "범위 출처", "상태", "회전 차이 %", "시간 ms"]
ROW_BG = {"HIGH": "#7f1d1d", "LOW": "#78350f", "RETAKE": "#713f12", "NORMAL": "#14532d"}


def find_weights() -> Path | None:
    cands = [os.environ.get("GAUGE_WEIGHTS"), ROOT / "weights" / "best.onnx",
             ROOT / "runs" / "pose" / "arc_v5b" / "weights" / "best.onnx"]
    return next((Path(c) for c in cands if c and Path(c).is_file()), None)


def tag_from_name(path: Path) -> str | None:
    tag = path.stem.split("__")[0] if "__" in path.stem else None
    return tag if tag in REGISTRY else None


@dataclass
class Item:
    path: Path
    tag: str | None
    reading: object = None
    images: dict = field(default_factory=dict)   # drawn, crop, front, photo (RGB)
    error: str = ""

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "" if self.reading is None else (status_of(self.reading) or "판독")


def analyse(reader, path: Path, tag: str | None, fast: bool, check: bool):
    data = np.fromfile(str(path), np.uint8)         # works with non-ASCII Windows paths
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("이미지를 열 수 없습니다")
    inst = REGISTRY.get(tag) if tag else None
    reader.tta_angle = None if fast else 25.0
    reader.check_scale = bool(check and inst is not None)
    r = reader.read(bgr, scale=inst.scale if inst else None, band=inst.band if inst else None)
    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    images = {"drawn": rgb(draw_reading(bgr, r)), "photo": rgb(bgr), "crop": None, "front": None}
    if r.box is not None:
        x0, y0, x1, y1 = map(int, r.box)
        h, w = bgr.shape[:2]
        m = int(0.1 * max(x1 - x0, y1 - y0))
        images["crop"] = rgb(bgr[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)])
    if r.fit is not None:
        images["front"] = rgb(frontal_view(bgr, r.fit, size=480))
    return r, images


class Worker(QObject):
    """Reads photos one by one off the UI thread; the model loads on the first job."""
    loaded = Signal(str)
    done = Signal(int, object, object, str)

    def __init__(self, weights: Path):
        super().__init__()
        self.weights, self.reader = weights, None

    @Slot(int, str, object, bool, bool)
    def read(self, index, path, tag, fast, check):
        try:
            if self.reader is None:
                from gauge_reader.reader import GaugeReader
                try:
                    import rapidocr_onnxruntime  # noqa: F401
                    ocr = True
                except ImportError:
                    ocr = False
                self.reader = GaugeReader(self.weights, ocr=ocr)
                self.loaded.emit("OCR 사용" if ocr else "OCR 없음: 등록된 계기만 판독")
            r, images = analyse(self.reader, Path(path), tag, fast, check)
            self.done.emit(index, r, images, "")
        except Exception as e:  # shown in the list, the round goes on
            self.done.emit(index, None, {}, str(e))


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class ImageView(QLabel):
    """A picture that keeps its aspect ratio when the window is resized."""

    def __init__(self, empty: str):
        super().__init__(empty, alignment=Qt.AlignCenter)
        self.empty, self._pix = empty, None
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_image(self, rgb):
        self._pix = None if rgb is None else to_pixmap(rgb)
        self._fit()

    def _fit(self):
        if self._pix is None:
            self.setPixmap(QPixmap())
            self.setText(self.empty)
        else:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


QSS = """
QMainWindow, QWidget { background: #0b1220; color: #e2e8f0; font-size: 13px; }
QToolBar { background: #0f2a3d; border: none; padding: 6px; spacing: 6px; }
QToolButton { background: #1e3a5f; border: 1px solid #2c5282; border-radius: 6px;
              padding: 6px 12px; color: #e2f2ff; }
QToolButton:hover { background: #24507f; }
QToolButton#primary { background: #0284c7; border-color: #0ea5e9; font-weight: bold; }
QListWidget { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #1e293b; }
QListWidget::item:selected { background: #1e3a5f; }
QFrame#card { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; }
QLabel#key { color: #64748b; font-size: 11px; font-weight: bold; background: transparent; }
QLabel#val { font-size: 16px; font-weight: bold; background: transparent; }
QTabWidget::pane { border: 1px solid #1e293b; border-radius: 8px; }
QTabBar::tab { background: #0f172a; padding: 7px 16px; border: 1px solid #1e293b; color: #94a3b8; }
QTabBar::tab:selected { background: #1e3a5f; color: #e2f2ff; }
QTableWidget { background: #0f172a; gridline-color: #1e293b; border: none; }
QHeaderView::section { background: #13203a; color: #94a3b8; padding: 5px; border: none; }
QComboBox { background: #13203a; border: 1px solid #2c5282; border-radius: 6px; padding: 4px 8px; }
QStatusBar { background: #0f2a3d; color: #8fb3d1; }
QProgressBar { background: #13203a; border: none; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #0ea5e9; border-radius: 4px; }
"""


class MainWindow(QMainWindow):
    request = Signal(int, str, object, bool, bool)

    def __init__(self, weights: Path):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.items: list[Item] = []
        self.pending = 0
        self._build()
        self.thread = QThread(self)
        self.worker = Worker(weights)
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.read)
        self.worker.done.connect(self.on_done)
        self.worker.loaded.connect(lambda s: self.model_info.setText(f"모델 v5b (ONNX, CPU) · {s}"))
        self.thread.start()

    # ---- layout
    def _build(self):
        tb = QToolBar(movable=False)
        self.addToolBar(tb)
        title = QLabel("  ⚛  mv2npp 계기 순시 점검  ")
        title.setStyleSheet("font-size:17px; font-weight:bold; color:#e2f2ff; background:transparent")
        tb.addWidget(title)
        for text, slot, name in [("📷 사진 열기", self.open_files, ""),
                                 ("📂 폴더 열기", self.open_folder, ""),
                                 ("🖼 예시 사진", self.open_examples, ""),
                                 ("▶ 전체 다시 판독", self.read_all, "primary"),
                                 ("💾 보고서 저장 (CSV)", self.save_csv, ""),
                                 ("ⓘ 정보", self.about, "")]:
            act = QAction(text, self, triggered=slot)
            tb.addAction(act)
            if name:
                tb.widgetForAction(act).setObjectName(name)

        self.list = QListWidget()
        self.list.setMinimumWidth(250)
        self.list.currentRowChanged.connect(self.show_item)

        self.views = {k: ImageView(t) for k, t in [
            ("drawn", "사진을 열거나 여기로 끌어다 놓으세요"), ("crop", "-"), ("front", "-"),
            ("photo", "-")]}
        self.tabs = QTabWidget()
        for k, t in [("drawn", "검출 · 키포인트"), ("crop", "계기 확대"), ("front", "정면 보정"),
                     ("photo", "원본")]:
            self.tabs.addTab(self.views[k], t)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        card = QFrame(objectName="card")
        cl = QVBoxLayout(card)
        self.meter = QSvgWidget()
        self.meter.setFixedSize(300, 250)
        cl.addWidget(self.meter, alignment=Qt.AlignCenter)
        rl.addWidget(card)
        self.tag_box = QComboBox()
        self.tag_box.addItem(NO_TAG, None)
        for t, inst in REGISTRY.items():
            self.tag_box.addItem(f"{t}  ({inst.scale_min:g}–{inst.scale_max:g} {inst.unit})", t)
        self.tag_box.activated.connect(self.change_tag)
        rl.addWidget(QLabel("계기 태그", objectName="key"))
        rl.addWidget(self.tag_box)
        self.fast = QCheckBox("빠른 모드 (회전 검증 없이 1회 판독)")
        self.check = QCheckBox("등록된 눈금을 OCR로 확인")
        self.check.setChecked(True)
        rl.addWidget(self.fast)
        rl.addWidget(self.check)
        grid = QGridLayout()
        self.kv = {}
        for i, k in enumerate(["상태", "눈금 범위", "범위 출처", "경보 범위", "바늘 위치",
                               "회전 판독 차이", "OCR 눈금 확인", "처리 시간"]):
            f = QFrame(objectName="card")
            fl = QVBoxLayout(f)
            fl.setContentsMargins(10, 6, 10, 6)
            fl.addWidget(QLabel(k, objectName="key"))
            v = QLabel("-", objectName="val")
            v.setWordWrap(True)
            fl.addWidget(v)
            self.kv[k] = v
            grid.addWidget(f, i // 2, i % 2)
        rl.addLayout(grid)
        rl.addStretch()
        right.setFixedWidth(340)

        top = QSplitter()
        top.addWidget(self.list)
        top.addWidget(self.tabs)
        top.addWidget(right)
        top.setStretchFactor(1, 1)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellClicked.connect(lambda r, _c: self.list.setCurrentRow(r))

        split = QSplitter(Qt.Vertical)
        split.addWidget(top)
        split.addWidget(self.table)
        split.setSizes([620, 240])
        self.setCentralWidget(split)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.model_info = QLabel("모델 v5b (ONNX, CPU)")
        self.summary = QLabel("")
        self.progress = QProgressBar(maximumWidth=200)
        self.progress.hide()
        sb.addWidget(self.model_info)
        sb.addWidget(QLabel(" | "))
        sb.addWidget(self.summary, 1)
        sb.addPermanentWidget(QLabel("합성 데이터로만 학습 · 실제 원전 사진 미검증  "))
        sb.addPermanentWidget(self.progress)

    # ---- opening photos
    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "사진 열기", "",
                                                "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        self.add_paths([Path(f) for f in files])

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "폴더 열기")
        if d:
            self.add_paths(sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMAGE_EXT))

    def open_examples(self):
        self.add_paths(sorted(p for p in EXAMPLES.glob("*") if p.suffix.lower() in IMAGE_EXT))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXT) \
                if p.is_dir() else [p] if p.suffix.lower() in IMAGE_EXT else []
        self.add_paths(paths)

    def add_paths(self, paths: list[Path]):
        known = {it.path for it in self.items}
        new = [p for p in paths if p not in known]
        for p in new:
            self.items.append(Item(p, tag_from_name(p)))
            self.list.addItem(QListWidgetItem())
            self.table.insertRow(self.table.rowCount())
            self._refresh_row(len(self.items) - 1)
        for i in range(len(self.items) - len(new), len(self.items)):
            self.submit(i)
        if new and self.list.currentRow() < 0:
            self.list.setCurrentRow(len(self.items) - len(new))

    # ---- reading
    def submit(self, i: int):
        it = self.items[i]
        it.reading, it.images, it.error = None, {}, ""
        self._refresh_row(i)
        self.pending += 1
        self.progress.setMaximum(max(self.progress.maximum(), self.pending))
        self.progress.show()
        self.request.emit(i, str(it.path), it.tag, self.fast.isChecked(), self.check.isChecked())

    def read_all(self):
        self.progress.setMaximum(0)
        for i in range(len(self.items)):
            self.submit(i)

    @Slot(int, object, object, str)
    def on_done(self, i, r, images, error):
        it = self.items[i]
        it.reading, it.images, it.error = r, images, error
        self.pending -= 1
        self.progress.setValue(self.progress.maximum() - self.pending)
        if self.pending == 0:
            self.progress.hide()
            self.progress.setMaximum(0)
        self._refresh_row(i)
        if i == self.list.currentRow():
            self.show_item(i)

    def change_tag(self):
        i = self.list.currentRow()
        if i >= 0:
            self.items[i].tag = self.tag_box.currentData()
            self.submit(i)
            self.show_item(i)

    # ---- display
    def _refresh_row(self, i: int):
        it, r = self.items[i], self.items[i].reading
        st = it.status or "판독 중…"
        value = "-" if r is None or r.value is None else f"{r.value:.3g} {r.unit}".strip()
        li = self.list.item(i)
        li.setText(f"●  {it.tag or '-'}   {it.path.name}\n     {value}   [{st}]")
        li.setForeground(QBrush(QColor(STATUS_COLORS.get(status_of(r), "#94a3b8")
                                       if r is not None else "#64748b")))
        cells = [it.path.name, it.tag or "-",
                 "-" if r is None or r.value is None else f"{r.value:.3g}",
                 "-" if r is None else (r.unit or "-"),
                 "-" if r is None or r.scale_min is None else f"{r.scale_min:g}–{r.scale_max:g}",
                 "-" if r is None else SOURCE.get(r.scale_source, "-"), st,
                 "-" if r is None or r.spread is None else f"{r.spread:.1f}",
                 "-" if r is None else f"{sum(r.ms.values()):.0f}"]
        for j, v in enumerate(cells):
            cell = QTableWidgetItem(v)
            if j == 6 and v in ROW_BG:
                cell.setBackground(QColor(ROW_BG[v]))
            self.table.setItem(i, j, cell)
        self._summary()

    def _summary(self):
        sts = [it.status for it in self.items]
        done = [s for s in sts if s]
        self.summary.setText(
            f"점검 {len(sts)}건 · 판독 {sum(s not in ('RETAKE', 'ERROR', '') for s in sts)} · "
            f"경보 {sum(s in ('HIGH', 'LOW') for s in sts)} · 재촬영 {sts.count('RETAKE')}"
            + (f" · 판독 중 {len(sts) - len(done)}" if len(done) < len(sts) else ""))

    def show_item(self, i: int):
        if i < 0:
            return
        it, r = self.items[i], self.items[i].reading
        self.tag_box.setCurrentIndex(max(0, self.tag_box.findData(it.tag)))
        for k, v in self.views.items():
            v.set_image(it.images.get(k))
        if r is None:
            self.views["drawn"].setText(it.error or "판독 중…")
            self.meter.load(QByteArray())
            for v in self.kv.values():
                v.setText("-")
            return
        inst = REGISTRY.get(it.tag) if it.tag else None
        band = inst.band if inst else None
        self.meter.load(QByteArray(meter_svg(r, band).encode()))
        st = status_of(r)
        self.kv["상태"].setText(STATUS_KO.get(r.status, r.status) if r.value is None
                              else (st or "판독"))
        self.kv["상태"].setStyleSheet(f"color:{STATUS_COLORS[st]}")
        self.kv["눈금 범위"].setText("-" if r.scale_min is None
                                  else f"{r.scale_min:g}–{r.scale_max:g} {r.unit}")
        self.kv["범위 출처"].setText(SOURCE.get(r.scale_source, "-"))
        self.kv["경보 범위"].setText("-" if band is None else
                                  f"{'' if band.low is None else f'{band.low:g}'}–"
                                  f"{'' if band.high is None else f'{band.high:g}'} {inst.unit}")
        self.kv["바늘 위치"].setText("-" if r.fraction is None else f"{r.fraction * 100:.1f}%")
        self.kv["회전 판독 차이"].setText("-" if r.spread is None else f"{r.spread:.1f}% FS")
        self.kv["OCR 눈금 확인"].setText({"match": "일치", "mismatch": "불일치 (다른 계기?)",
                                        "no_ocr": "숫자를 못 읽음"}.get(r.scale_check, "-"))
        self.kv["처리 시간"].setText(f"{sum(r.ms.values()):.0f} ms")

    # ---- report and info
    def save_csv(self):
        if not self.items:
            return
        path, _ = QFileDialog.getSaveFileName(self, "보고서 저장", "gauge_round.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["photo", "tag", "status", "alarm", "value", "unit", "scale_min",
                        "scale_max", "scale_source", "scale_check", "needle_fraction",
                        "spread_pct", "ms", "error"])
            for it in self.items:
                r = it.reading
                if r is None:
                    w.writerow([it.path.name, it.tag or "", "", "", "", "", "", "", "", "", "",
                                "", "", it.error])
                    continue
                w.writerow([it.path.name, it.tag or "", r.status, r.alarm,
                            "" if r.value is None else f"{r.value:.6g}", r.unit,
                            r.scale_min, r.scale_max, r.scale_source, r.scale_check,
                            "" if r.fraction is None else f"{r.fraction:.4f}",
                            "" if r.spread is None else f"{r.spread:.2f}",
                            f"{sum(r.ms.values()):.0f}", it.error])
        self.statusBar().showMessage(f"저장했습니다: {path}", 5000)

    def about(self):
        QMessageBox.about(self, APP_NAME, (
            "<h3>mv2npp · 아날로그 계기 자동 판독</h3>"
            "<p>원전 현장 순시용 머신비전. 합성 렌더로만 학습한 YOLO pose 모델(v5b)로 계기를 찾고 "
            "7개 키포인트로 문자판을 정면 보정해 바늘을 읽습니다. 세 번 회전해 읽어 5% 넘게 "
            "다르면 재촬영(RETAKE)을 요청합니다.</p>"
            "<p>실제 원전 사진으로는 검증하지 않았습니다.</p>"
            "<p>코드: <a href='https://github.com/uhyon/mv2npp-gauge-reading'>"
            "github.com/uhyon/mv2npp-gauge-reading</a> (MIT)<br>"
            "예시 사진: 자체 합성 렌더(두 장은 CC BY 2.0 배경 사진 위), Roboflow Universe "
            "\"Analog Gauge Reading\" 사진(CC BY 4.0). 출처: demo/examples/ATTRIBUTION.md</p>"))

    def closeEvent(self, e):
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(e)


def selftest() -> int:
    """Read the example photos without a window (the Windows build's CI check)."""
    from gauge_reader.reader import GaugeReader
    weights = find_weights()
    if weights is None:
        print("no weights")
        return 1
    try:
        import rapidocr_onnxruntime  # noqa: F401
        ocr = True
    except ImportError:
        ocr = False
    reader = GaugeReader(weights, ocr=ocr)
    paths = sorted(p for p in EXAMPLES.glob("*") if p.suffix.lower() in IMAGE_EXT)
    values = 0
    for p in paths:
        r, _ = analyse(reader, p, tag_from_name(p), False, True)
        values += r.value is not None
        print(f"{p.name}: {r.status} {r.value} {r.unit} {r.alarm} "
              f"({sum(r.ms.values()):.0f} ms)")
    print(f"ocr={ocr} read {values}/{len(paths)}")
    return 0 if paths and values >= len(paths) - 1 else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    font = QFont()
    font.setFamilies(["Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR",
                      "WenQuanYi Zen Hei", "Segoe UI"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(QSS)
    weights = find_weights()
    if weights is None:
        f, _ = QFileDialog.getOpenFileName(None, "모델 파일(best.onnx)을 골라 주세요", "",
                                           "ONNX (*.onnx)")
        if not f:
            return 1
        weights = Path(f)
    win = MainWindow(weights)
    win.resize(1440, 900)
    win.show()
    args = [Path(a) for a in argv[1:] if not a.startswith("--")]
    paths = []
    for a in args:
        paths += sorted(p for p in a.iterdir() if p.suffix.lower() in IMAGE_EXT) \
            if a.is_dir() else [a]
    if paths:
        win.add_paths(paths)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
