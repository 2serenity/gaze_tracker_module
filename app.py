"""
Визуализация направления взгляда. MediaPipe + калибратор (Ridge).
Камера и обработка в отдельных потоках.
"""

import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    print("Требуется Python с модулем tkinter.")
    sys.exit(1)

from extractor import GazeExtractor
from calibrator import GazeCalibrator

# Точки калибровки [0, 1]
CALIBRATION_MAP = np.column_stack([
    np.tile(np.linspace(0.2, 0.8, 4), 4),
    np.repeat(np.linspace(0.2, 0.8, 4), 4),
])
np.random.shuffle(CALIBRATION_MAP)
FRAMES_PER_POINT = 25


class GazeVisualizationApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Визуализация направления взгляда")
        self.root.geometry("900x600")
        self.root.minsize(700, 500)

        self.cap = None
        self.extractor = None
        self.calibrator = None
        self.running = False
        self.calibrating = True
        self.calib_point_idx = 0
        self.frames_at_point = 0
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.screen_size = (800, 450)
        self.calib_path_var = tk.StringVar(value="")
        self.camera_idx_var = tk.IntVar(value=0)
        self._calib_target = (0.5, 0.5)

        self._build_ui()
        self._init_core()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Калибровка (файл):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.calib_path_var, width=35).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Обзор...", command=self._browse_calib).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Загрузить", command=self._load_calib).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Сохранить", command=self._save_calib).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="Камера:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Spinbox(top, from_=0, to=4, width=3, textvariable=self.camera_idx_var).pack(side=tk.LEFT, padx=2)
        self.btn_start = ttk.Button(top, text="Старт", command=self._toggle_stream)
        self.btn_start.pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Выход", command=self._quit).pack(side=tk.LEFT)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        cam_frame = ttk.LabelFrame(main, text="Камера", padding=4)
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cam_label = ttk.Label(cam_frame, text="Нажмите «Старт» для запуска камеры", anchor=tk.CENTER)
        self.cam_label.pack(fill=tk.BOTH, expand=True)
        screen_frame = ttk.LabelFrame(main, text="Направление взгляда на экране", padding=4)
        screen_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.screen_canvas = tk.Canvas(
            screen_frame, width=self.screen_size[0], height=self.screen_size[1],
            bg="#1a1a2e", highlightthickness=1, highlightbackground="#444"
        )
        self.screen_canvas.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Нажмите «Старт». В начале — калибровка по точкам.")
        ttk.Label(main, textvariable=self.status_var).pack(side=tk.BOTTOM, pady=4)

    def _browse_calib(self):
        path = filedialog.askopenfilename(
            title="Файл калибровки",
            filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")]
        )
        if path:
            self.calib_path_var.set(path)

    def _init_core(self):
        self.extractor = GazeExtractor()
        self.calibrator = GazeCalibrator()
        default_calib = ROOT / "calib.pkl"
        if default_calib.exists():
            self.calib_path_var.set(str(default_calib))
            self._load_calib()

    def _load_calib(self):
        path = self.calib_path_var.get().strip()
        if not path or not Path(path).exists():
            messagebox.showwarning("Внимание", "Укажите существующий файл калибровки (.pkl).")
            return
        try:
            self.calibrator = GazeCalibrator.load(path)
            self.status_var.set("Калибровка загружена. Нажмите «Старт».")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить калибровку:\n{e}")

    def _save_calib(self):
        if self.calibrator is None or not self.calibrator.fitted:
            messagebox.showwarning("Внимание", "Нет обученной калибровки (сначала откалибруйте).")
            return
        path = self.calib_path_var.get().strip() or str(ROOT / "calib.pkl")
        try:
            self.calibrator.save(path)
            self.status_var.set(f"Калибровка сохранена: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _camera_reader(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)

    def _process_frame_worker(self):
        w, h = self.screen_size[0], self.screen_size[1]
        while self.running and self.extractor is not None and self.calibrator is not None:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            gaze_x, gaze_y = 0.5 * w, 0.5 * h
            key_points = self.extractor.extract(frame)
            if key_points is not None:
                if self.calibrating:
                    tx, ty = self._calib_target
                    self.calibrator.add(key_points, tx, ty)
                else:
                    x_norm, y_norm = self.calibrator.predict(key_points)
                    gaze_x = x_norm * w
                    gaze_y = y_norm * h
            try:
                self.result_queue.put_nowait((frame.copy(), gaze_x, gaze_y))
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put_nowait((frame.copy(), gaze_x, gaze_y))

    def _toggle_stream(self):
        if self.running:
            self.running = False
            self.btn_start.config(text="Старт")
            if getattr(self, "worker_thread", None):
                self.worker_thread.join(timeout=1.0)
                self.worker_thread = None
            if getattr(self, "camera_thread", None):
                self.camera_thread.join(timeout=1.0)
                self.camera_thread = None
            if self.cap:
                self.cap.release()
                self.cap = None
            for q in (self.frame_queue, self.result_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            self.status_var.set("Остановлено.")
            return
        if self.extractor is None or self.calibrator is None:
            messagebox.showwarning("Внимание", "Модули не инициализированы.")
            return
        self.extractor.reset_reference()
        calib_path = self.calib_path_var.get().strip()
        if Path(calib_path).exists():
            try:
                self.calibrator = GazeCalibrator.load(calib_path)
            except Exception:
                self.calibrator = GazeCalibrator()
        else:
            self.calibrator = GazeCalibrator()
        self.cap = cv2.VideoCapture(self.camera_idx_var.get())
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру.")
            return
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.calib_point_idx = 0
        self.frames_at_point = 0
        self.calibrating = not self.calibrator.fitted
        self._calib_target = (CALIBRATION_MAP[0, 0], CALIBRATION_MAP[0, 1])
        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_reader, daemon=True)
        self.camera_thread.start()
        self.worker_thread = threading.Thread(target=self._process_frame_worker, daemon=True)
        self.worker_thread.start()
        self.btn_start.config(text="Стоп")
        self.status_var.set("Калибровка: смотрите в зелёную точку. Затем — траектория взгляда.")
        self._update_frame()

    def _update_frame(self):
        if not self.running:
            return
        if self.calibrating:
            self.frames_at_point += 1
            if self.frames_at_point >= FRAMES_PER_POINT:
                self.frames_at_point = 0
                self.calib_point_idx += 1
                if self.calib_point_idx >= len(CALIBRATION_MAP):
                    self.calibrator.fit()
                    self.calibrating = False
                    self.status_var.set("Калибровка завершена. Траектория взгляда.")
                else:
                    self._calib_target = (
                        float(CALIBRATION_MAP[self.calib_point_idx, 0]),
                        float(CALIBRATION_MAP[self.calib_point_idx, 1]),
                    )
        result = None
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        if result is not None:
            frame, gaze_x, gaze_y = result
            px, py = int(gaze_x), int(gaze_y)
            self.screen_canvas.delete("gaze")
            self.screen_canvas.delete("calib")
            if self.calibrating:
                cx = int(self._calib_target[0] * self.screen_size[0])
                cy = int(self._calib_target[1] * self.screen_size[1])
                self.screen_canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill="#0f0", outline="#fff", width=2, tags="calib")
            r = 12
            self.screen_canvas.create_oval(px - r, py - r, px + r, py + r, fill="#e94560", outline="#fff", width=2, tags="gaze")
            frame_show = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB)
            try:
                from PIL import Image, ImageTk
                img = Image.fromarray(frame_rgb)
                img.thumbnail((400, 400))
                self.photo = ImageTk.PhotoImage(image=img)
                self.cam_label.config(image=self.photo, text="")
            except ImportError:
                self.cam_label.config(image="", text="[Видео]")
        self.root.after(25, self._update_frame)

    def _quit(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()


def main():
    app = GazeVisualizationApp()
    app.run()


if __name__ == "__main__":
    main()
