"""Glove inspection GUI (PySide6).

  python app/gui.py                                   # choose a video / camera in the window
  python app/gui.py --source data/videos/demo.mp4     # start with a video (calib.png next to it is used)
  python app/gui.py --source 0 --calib my_board.png   # live webcam

Shows the rectified (top-down, mm-scaled) video with overlays, the current glove, totals per defect
type, the three sorting bins and a log table. Every glove is written to logs/gloves_*.csv and
logs/defects_*.csv. With an Arduino connected, each decision also moves the servo diverter.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QMainWindow, QPushButton, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gloveinspect import Inspector, load_config  # noqa: E402
from gloveinspect.logger import CsvLogger  # noqa: E402
from gloveinspect.overlay import DECISION_COLOURS, defect_label, render_frame  # noqa: E402
from gloveinspect.sorter import SerialSorter, SimSorter, list_ports  # noqa: E402

DEFECTS = ["hole", "pinhole", "tear", "missing_finger", "fused_fingers", "stain", "dirt", "cuff_defect", "out_of_tolerance"]
NICE = {"hole": "Hole", "pinhole": "Pinhole", "tear": "Edge tear", "missing_finger": "Missing / short finger",
        "fused_fingers": "Fused fingers", "stain": "Stain", "dirt": "Dirt / particle", "cuff_defect": "Cuff defect",
        "out_of_tolerance": "Out of tolerance"}
TYPE_NAMES = {"exam": "Exam glove", "longcuff": "Long-cuff glove", "garden": "Garden glove"}


def bgr_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Worker(QThread):
    frame_ready = Signal(QImage, float, str)
    gloves_done = Signal(object)
    finished_source = Signal()

    def __init__(self, inspector: Inspector, source, realtime=True, display_scale=0.6):
        super().__init__()
        self.insp = inspector
        self.source = source
        self.realtime = realtime
        self.scale = display_scale
        self._running = True
        self._paused = False
        self.shown = {}

    def stop(self):
        self._running = False

    def toggle_pause(self):
        self._paused = not self._paused
        return self._paused

    def run(self):
        cap = cv2.VideoCapture(self.source)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 20
        is_file = not isinstance(self.source, int)
        t_last, fps = time.time(), 0.0
        while self._running:
            if self._paused:
                self.msleep(50)
                continue
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            res = self.insp.process(frame)
            if res.finished:
                for r in res.finished:
                    self.shown[r.id] = r
                self.gloves_done.emit(res.finished)
            vis = render_frame(res, self.shown, self.scale)
            dt = time.time() - t_last
            t_last = time.time()
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)
            status = "learning empty plate…" if res.learning else f"{len(res.blobs)} glove(s) in view"
            self.frame_ready.emit(to_qimage(vis), fps, status)
            if is_file and self.realtime:
                wait = 1.0 / fps_src - (time.time() - t0)
                if wait > 0:
                    self.msleep(int(wait * 1000))
        rest = self.insp.flush()
        if rest:
            self.gloves_done.emit(rest)
        cap.release()
        self.finished_source.emit()


class BinWidget(QLabel):
    def __init__(self, label, colour):
        super().__init__()
        self.label, self.colour, self.count = label, colour, 0
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(70)
        self._paint(False)

    def _paint(self, flash):
        border = "#ffffff" if flash else self.colour
        self.setText(f"<div style='font-size:13px'>{self.label}</div><div style='font-size:26px;font-weight:600'>{self.count}</div>")
        self.setStyleSheet(f"background:{self.colour}; color:white; border:4px solid {border}; border-radius:8px;")

    def add(self):
        self.count += 1
        self._paint(True)

    def unflash(self):
        self._paint(False)

    def reset(self):
        self.count = 0
        self._paint(False)


class MainWindow(QMainWindow):
    def __init__(self, cfg_path=None):
        super().__init__()
        self.setWindowTitle("Glove Inspection & Sorting")
        self.cfg = load_config(cfg_path)
        self.insp = Inspector(self.cfg)
        self.worker: Worker | None = None
        self.sorter = SimSorter()
        self.logger: CsvLogger | None = None
        self.totals = {d: 0 for d in DEFECTS}
        self.source = None
        self.calib_path = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        central = QWidget()
        root = QHBoxLayout(central)
        left = QVBoxLayout()
        self.video = QLabel("Open a video or camera")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(800, 480)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video.setStyleSheet("background:#1a1a19; color:#c3c2b7;")
        left.addWidget(self.video, 1)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Open video…")
        self.cam_box = QComboBox()
        self.cam_box.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.btn_cam = QPushButton("Start camera")
        self.btn_calib = QPushButton("Calibrate (checkerboard image)…")
        self.btn_relearn = QPushButton("Re-learn empty plate")
        self.btn_pause = QPushButton("Pause")
        for b in (self.btn_open, self.cam_box, self.btn_cam, self.btn_calib, self.btn_relearn, self.btn_pause):
            bar.addWidget(b)
        left.addLayout(bar)
        self.status = QLabel("not calibrated – measurements use the fallback scale")
        self.status.setStyleSheet("color:#52514e")
        left.addWidget(self.status)
        root.addLayout(left, 5)

        right = QVBoxLayout()
        g_cur = QGroupBox("Current glove")
        gl = QGridLayout(g_cur)
        self.cur = {}
        for i, k in enumerate(["Glove", "Type", "Hand / direction", "Length", "Width", "Decision"]):
            gl.addWidget(QLabel(k + ":"), i, 0)
            lab = QLabel("–")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            gl.addWidget(lab, i, 1)
            self.cur[k] = lab
        self.cur_defects = QListWidget()
        self.cur_defects.setMaximumHeight(110)
        gl.addWidget(self.cur_defects, 6, 0, 1, 2)
        right.addWidget(g_cur)

        g_bins = QGroupBox("Sorting bins")
        bl = QHBoxLayout(g_bins)
        self.bins = {k: BinWidget(k, bgr_hex(DECISION_COLOURS[k])) for k in ("PASS", "REWORK", "REJECT")}
        for w in self.bins.values():
            bl.addWidget(w)
        right.addWidget(g_bins)
        ar = QHBoxLayout()
        self.port_box = QComboBox()
        self.port_box.addItems(["(no Arduino – simulated)"] + list_ports())
        self.btn_port = QPushButton("Connect")
        ar.addWidget(QLabel("Servo:"))
        ar.addWidget(self.port_box, 1)
        ar.addWidget(self.btn_port)
        right.addLayout(ar)

        g_tot = QGroupBox("Totals per defect type")
        tl = QVBoxLayout(g_tot)
        self.tot_table = QTableWidget(len(DEFECTS), 2)
        self.tot_table.setHorizontalHeaderLabels(["Defect", "Count"])
        self.tot_table.verticalHeader().setVisible(False)
        self.tot_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i, d in enumerate(DEFECTS):
            self.tot_table.setItem(i, 0, QTableWidgetItem(NICE[d]))
            self.tot_table.setItem(i, 1, QTableWidgetItem("0"))
        self.tot_table.setMaximumHeight(300)
        tl.addWidget(self.tot_table)
        self.total_label = QLabel("Gloves inspected: 0")
        tl.addWidget(self.total_label)
        right.addWidget(g_tot)

        g_log = QGroupBox("Log")
        ll = QVBoxLayout(g_log)
        self.log_table = QTableWidget(0, 8)
        self.log_table.setHorizontalHeaderLabels(["#", "Type", "Hand", "Dir", "L mm", "W mm", "Defects", "Decision"])
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        ll.addWidget(self.log_table)
        self.log_path = QLabel("CSV: –")
        self.log_path.setStyleSheet("color:#52514e; font-size:10px")
        ll.addWidget(self.log_path)
        right.addWidget(g_log, 1)
        root.addLayout(right, 3)
        self.setCentralWidget(central)

        self.btn_open.clicked.connect(self.open_video)
        self.btn_cam.clicked.connect(lambda: self.start_source(self.cam_box.currentIndex()))
        self.btn_calib.clicked.connect(self.choose_calibration)
        self.btn_relearn.clicked.connect(self.relearn)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_port.clicked.connect(self.connect_port)

    # ------------------------------------------------------------------ actions
    def calibrate(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            self.status.setText(f"cannot read {path}")
            return
        try:
            r = self.insp.calibrate(img)
            self.calib_path = path
            self.status.setText(f"calibrated from {path.name}: {r.px_per_mm:.1f} px/mm, board RMS {r.rms_mm:.3f} mm")
        except RuntimeError as e:
            self.status.setText(f"calibration failed: {e}")

    def choose_calibration(self):
        p, _ = QFileDialog.getOpenFileName(self, "Checkerboard image", str(ROOT), "Images (*.png *.jpg *.jpeg *.bmp)")
        if p:
            self.calibrate(Path(p))

    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", str(ROOT / "data"), "Video (*.mp4 *.avi *.mov *.mkv)")
        if p:
            self.start_source(p)

    def start_source(self, source, realtime=True):
        self.stop_worker()
        if isinstance(source, str):
            calib = Path(source).with_name("calib.png")
            if calib.exists():
                self.calibrate(calib)
        self.source = source
        self.insp.start_background_learning()
        self.insp.reset_tracking()
        self.logger = CsvLogger(self.cfg.path(self.cfg["logging"]["dir"]), str(source))
        self.log_path.setText(f"CSV: {_rel(self.logger.gloves_path)}")
        self.worker = Worker(self.insp, source, realtime)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.gloves_done.connect(self.on_gloves)
        self.worker.finished_source.connect(lambda: self.status.setText(self.status.text() + "  |  end of video"))
        self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None

    def relearn(self):
        self.insp.start_background_learning()

    def pause(self):
        if self.worker:
            self.btn_pause.setText("Resume" if self.worker.toggle_pause() else "Pause")

    def connect_port(self):
        port = self.port_box.currentText()
        self.sorter.close()
        if port.startswith("("):
            self.sorter = SimSorter()
            return
        try:
            self.sorter = SerialSorter(port, self.cfg["sorter"]["baud"])
            self.status.setText(f"servo sorter connected on {port}")
        except Exception as e:
            self.sorter = SimSorter()
            self.status.setText(f"could not open {port}: {e}")

    # ------------------------------------------------------------------ slots
    @Slot(QImage, float, str)
    def on_frame(self, img, fps, status):
        pm = QPixmap.fromImage(img).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
        self.video.setPixmap(pm)
        calib = "calibrated" if self.insp.rect is not None and self.insp.rect.calibrated else "NOT calibrated"
        self.setWindowTitle(f"Glove Inspection & Sorting – {fps:4.1f} FPS – {calib} – {status}")

    @Slot(object)
    def on_gloves(self, recs):
        for w in self.bins.values():
            w.unflash()
        for rec in recs:
            self.show_current(rec)
            reply = self.sorter.sort(rec.id, rec.decision.label)
            self.bins[rec.decision.label].add()
            if self.logger:
                self.logger.log(rec)
            for d in rec.defects:
                self.totals[d.type] += 1
            row = self.log_table.rowCount()
            self.log_table.insertRow(row)
            vals = [rec.id, rec.type, rec.hand, "up" if rec.direction == "fingers_up" else "down",
                    f"{rec.length_mm:.1f}", f"{rec.width_mm:.1f}", ", ".join(sorted({d.type for d in rec.defects})) or "–",
                    rec.decision.label]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c == 7:
                    it.setForeground(Qt.GlobalColor.white)
                    it.setBackground(QColor_hex(bgr_hex(DECISION_COLOURS[rec.decision.label])))
                self.log_table.setItem(row, c, it)
            self.log_table.scrollToBottom()
            if reply:
                self.status.setText(f"Arduino: {reply}")
        for i, d in enumerate(DEFECTS):
            self.tot_table.item(i, 1).setText(str(self.totals[d]))
        self.total_label.setText(f"Gloves inspected: {sum(w.count for w in self.bins.values())}")

    def show_current(self, rec):
        spec = self.cfg["glove_types"].get(rec.type, {})
        tol = self.cfg["tolerance"]
        ok_l = abs(rec.length_dev_mm) <= tol["length"]
        ok_w = abs(rec.width_dev_mm) <= tol["width"]
        self.cur["Glove"].setText(f"#{rec.id}  ({rec.n_obs} frames)")
        self.cur["Type"].setText(f"{TYPE_NAMES.get(rec.type, rec.type)}  (p = {rec.type_conf:.2f})")
        hand = {"R": "right", "L": "left", "?": "undecided"}[rec.hand]
        self.cur["Hand / direction"].setText(f"{hand} hand, fingers {'up' if rec.direction == 'fingers_up' else 'down'}")
        self.cur["Length"].setText(f"{rec.length_mm:.1f} mm  (spec {spec.get('length', 0):.0f} ± {tol['length']:.0f})  "
                                   f"{'OK' if ok_l else 'OUT'}")
        self.cur["Width"].setText(f"{rec.width_mm:.1f} mm  (spec {spec.get('width', 0):.0f} ± {tol['width']:.0f})  "
                                  f"{'OK' if ok_w else 'OUT'}")
        col = bgr_hex(DECISION_COLOURS[rec.decision.label])
        self.cur["Decision"].setText(f"<b style='color:{col}'>{rec.decision.label}</b>  fuzzy quality "
                                     f"{rec.decision.score:.0f}/100")
        self.cur_defects.clear()
        for d in rec.defects:
            self.cur_defects.addItem(f"{NICE[d.type]}: {defect_label(d)}")
        if not rec.defects:
            self.cur_defects.addItem("no defects")

    def closeEvent(self, e):
        self.stop_worker()
        self.sorter.close()
        super().closeEvent(e)


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return Path(p).name


def QColor_hex(h):
    from PySide6.QtGui import QColor
    return QColor(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="video file or camera index")
    ap.add_argument("--calib", type=Path, help="checkerboard image for mm calibration")
    ap.add_argument("--config", default=None)
    ap.add_argument("--snapshot", type=Path, help="(docs) run the source, save a window screenshot, exit")
    ap.add_argument("--snapshot-after", type=int, default=6, help="gloves to wait for before the snapshot")
    a = ap.parse_args()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    w = MainWindow(a.config)
    w.resize(1500, 860)
    w.show()
    if a.calib:
        w.calibrate(a.calib)
    if a.source is not None:
        src = int(a.source) if a.source.isdigit() else a.source
        w.start_source(src, realtime=a.snapshot is None)
        if a.snapshot:
            def maybe_snap(_recs):
                if sum(b.count for b in w.bins.values()) >= a.snapshot_after:
                    w.worker.toggle_pause()
                    app.processEvents()
                    time.sleep(0.3)
                    app.processEvents()
                    w.grab().save(str(a.snapshot))
                    w.close()
            w.worker.gloves_done.connect(maybe_snap)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
