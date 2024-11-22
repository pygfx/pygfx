# ruff: noqa: N802
# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

"""
Interactive Segmentation with SAM2
==================================

This script demonstrates a real-time interactive segmentation application using pygfx points as prompts to the SAM2 (Segment Anything Model 2) model. SAM2 relies on PyTorch for inference, and the GUI is built using Qt (PySide6).

Additional dependencies required to run this example:
    pip install PySide6 torch
    pip install git+https://github.com/facebookresearch/sam2.git

Once the application is running, you can click and drag the green point to interactively segment the image. The model will update the segmentation mask in real-time as you move the point around.
"""

from pathlib import Path
from queue import LifoQueue
from threading import Event

import imageio.v3 as iio
import numpy as np
import torch
from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from wgpu.gui.qt import WgpuCanvas

import pygfx as gfx


class SAMPoint(QtWidgets.QWidget):
    segmentation_mask_signal = Signal(object)

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("SAM2 Real-time Point Segmentation")
        self.resize(800, 800)

        self.canvas = WgpuCanvas(parent=self, max_fps=-1)
        self.renderer = gfx.WgpuRenderer(self.canvas, show_fps=True)
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(0)
        self.camera.local.scale_y = -1
        self.controller = gfx.PanZoomController(
            self.camera, register_events=self.renderer
        )

        self.canvas.request_draw(self.animate)

        self.reset_view_button = QtWidgets.QPushButton("Reset View", self)
        self.reset_view_button.clicked.connect(self.reset_view)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.reset_view_button)

        base_image = iio.imread("imageio:astronaut.png")
        self.image_shape = base_image.shape[:2]
        self.mask_image = None
        self.edit_world_object = None

        material = gfx.ImageBasicMaterial(clim=(0, 255))
        self.base_image = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(base_image, dim=2)), material
        )

        cmap = np.zeros((2, 4), dtype=np.float32)
        cmap[0, :] = (0, 0, 0, 0)
        cmap[1, :] = (0.8, 0, 1, 0.3)
        mask_material = gfx.ImageBasicMaterial(
            clim=(0, len(cmap) - 1), map=gfx.Texture(cmap, dim=1)
        )
        initial_mask = np.zeros(self.image_shape, dtype=np.uint8)
        overlay_texture = gfx.Texture(initial_mask, dim=2)
        self.mask_image = gfx.Image(gfx.Geometry(grid=overlay_texture), mask_material)
        base_x, base_y, _ = self.base_image.local.position
        self.mask_image.local.position = base_x, base_y, 3

        point_size = int(np.sqrt(self.image_shape[0] * self.image_shape[1]) * 0.02)
        point_material = gfx.PointsMarkerMaterial(
            size=point_size,
            color="lightgreen",
            size_space="world",
            edge_color="black",
            edge_width=1,
            pick_write=True,
        )
        self.dot_marker = gfx.Points(
            gfx.Geometry(positions=[[0, 0, 3]]),
            point_material,
        )
        self.scene.add(self.base_image)
        self.scene.add(self.mask_image)
        self.scene.add(self.dot_marker)

        self.segmentation_queue = LifoQueue(maxsize=1)
        self.segmentation_stop_event = Event()
        self.segmentation_mask_signal.connect(self.update_segmentation_mask)

        self.camera.show_object(self.scene)

        self.scene.add_event_handler(
            self.pointer_event_handler,
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )

        self.segmentation_runner = SegmentationRunner(
            image=base_image,
            segmentation_queue=self.segmentation_queue,
            segmentation_mask_signal=self.segmentation_mask_signal,
            stop_event=self.segmentation_stop_event,
        )
        self.segmentation_runner.start()

        start_position = (347, 321)
        self.dot_marker.local.position = (
            start_position[0],
            start_position[1],
            3,
        )
        self.segmentation_queue.put(start_position)

    def screen_to_world(self, xy):
        x_ndc = (xy[0] / self.renderer.logical_size[0]) * 2 - 1
        y_ndc = -(xy[1] / self.renderer.logical_size[1]) * 2 + 1
        ndc_pos = np.array([x_ndc, y_ndc, 0, 1])
        inv_matrix = np.linalg.inv(
            self.camera.projection_matrix @ self.camera.view_matrix
        )
        world_pos = inv_matrix @ ndc_pos
        world_pos /= world_pos[3]

        return np.array([world_pos[0], world_pos[1], 0])

    def pointer_event_handler(self, event):
        if event.type == "pointer_down":
            pick_info = event.pick_info
            world_object = pick_info.get("world_object")
            if world_object == self.dot_marker:
                self.edit_world_object = self.dot_marker
                self.scene.set_pointer_capture(event.pointer_id, event.root)
                return

        elif event.type == "pointer_move" and self.edit_world_object is not None:
            world_pos = self.screen_to_world((event.x, event.y))
            x_clamped = np.clip(world_pos[0], 0, self.image_shape[1])
            y_clamped = np.clip(world_pos[1], 0, self.image_shape[0])
            self.edit_world_object.local.position = (x_clamped, y_clamped, 3)
            self.segmentation_queue.put((x_clamped, y_clamped))

        elif event.type == "pointer_up" and self.edit_world_object is not None:
            self.edit_world_object = None
            self.scene.release_pointer_capture(event.pointer_id)

    def animate(self):
        self.renderer.render(self.scene, self.camera)

    def reset_view(self):
        self.camera.show_object(self.scene)
        self.canvas.update()

    def update_segmentation_mask(self, mask):
        if mask is None:
            return

        self.mask_image.geometry.grid.data[...] = mask
        size = self.mask_image.geometry.grid.size
        self.mask_image.geometry.grid.update_range(offset=(0, 0, 0), size=size)
        self.canvas.update()

    def closeEvent(self, event):
        self.segmentation_stop_event.set()
        if self.segmentation_runner is not None:
            while not self.segmentation_runner.isFinished():
                self.segmentation_runner.terminate()
                self.segmentation_runner.wait()
        self.segmentation_queue = None
        self.segmentation_runner = None
        event.accept()


class SegmentationRunner(QThread):
    def __init__(
        self,
        *,
        image,
        segmentation_queue,
        segmentation_mask_signal,
        stop_event,
    ):
        super().__init__()
        self.image = image
        self.segmentation_queue = segmentation_queue
        self.segmentation_mask_signal = segmentation_mask_signal
        self.stop_event = stop_event
        self.predictor = None
        self.init_predictor()

    def init_predictor(self):
        file_path = Path(__file__).parent.parent
        model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        model_type = "sam2.1_hiera_tiny.pt"
        model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model_path = file_path / "data" / model_type

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {model_url} to {model_path}")
            torch.hub.download_url_to_file(model_url, str(model_path), progress=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = SAM2ImagePredictor(
            build_sam2(
                model_config,
                model_path,
                device=device,
            )
        )
        self.predictor.set_image(self.image)

    def run(self):
        while not self.stop_event.is_set():
            sam_point = self.segmentation_queue.get()
            if sam_point is None:
                self.segmentation_mask_signal.emit(None)
                continue

            if self.predictor is None:
                self.stop_event.set()

            mask_input = None

            point_coords = np.array([[sam_point[0], sam_point[1]]], dtype="float32")
            point_labels = np.array([1])
            mask, _score, prev_low_res_mask = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=False,
            )
            mask_input = prev_low_res_mask
            self.segmentation_mask_signal.emit(mask[0].astype("uint8"))

        self.finished.emit()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    m = SAMPoint()
    m.show()
    app.exec()
