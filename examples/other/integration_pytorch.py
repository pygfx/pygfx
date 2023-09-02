# run_example = false

"""
Pytorch Integration
=============================

Integration example of pygfx with pytorch lightning and pytorch geometric.

This example demonstrate how to train a graph neural network on a 3D mesh, while continuously rendering
the results on a pygfx window in a separate process. The network tries to predict the Gaussian curvature
of each point on the mesh by overfitting the ground-truth curvature.

"""

# standard library
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List,
    Any,
    cast)
from dataclasses import dataclass
import queue

# numpy
import numpy as np

# sklearn
from sklearn.decomposition import PCA

# trimesh
import trimesh

# libigl
import igl

# pygfx
import pygfx as gfx
from pygfx import (
    Viewport,
    Renderer,
    Scene,
    Group,
    Background,
    BackgroundMaterial,
    Color,
    Camera,
    PerspectiveCamera,
    NDCCamera,
    PointLight,
    OrbitController,
    Geometry,
    WorldObject,
    Mesh,
    MeshPhongMaterial,
    Buffer)
from pygfx.renderers import WgpuRenderer

# wgpu
from wgpu.gui.auto import WgpuCanvas, run

# pytorch
import torch
from torch import (
    optim,
    nn)
from torch.nn import (
    ReLU,
    GELU,
    Tanh,
    Sigmoid)
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

# pytorch lightning
from lightning.pytorch import (
    Trainer,
    LightningModule)
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.datamodule import LightningDataModule

# torch geometric
import torch_geometric
from torch_geometric.nn import (
    MessagePassing)
from torch_geometric.data import (
    Dataset,
    Batch,
    Data)
from torch_geometric.loader import DataLoader
from torch_geometric import transforms


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    k: np.ndarray
    pred_k: np.ndarray


@dataclass
class SceneState:
    gt_group: Group = Group()
    pred_group: Group = Group()
    gt_mesh: WorldObject = None
    pred_mesh: WorldObject = None
    gt_wireframe: WorldObject = None
    pred_wireframe: WorldObject = None
    first_data: bool = True


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


activations = {
    'relu': ReLU,
    'gelu': GELU,
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    'sine': Sine
}


def canonical_form(vertices: np.ndarray) -> np.ndarray:
    # Compute the center of mass
    center_of_mass = np.mean(a=vertices, axis=0)

    # Translate the mesh so that the center of mass is at the origin
    vertices -= center_of_mass

    # Perform PCA to find the main axes of the data
    pca = PCA(n_components=3)
    pca.fit(vertices)

    # The PCA components are the axes that we want to align with the coordinate axes.
    # We need to reorder the components so that the first and third components are the ones
    # with the largest and second largest variance, and the second component is the one
    # with the smallest variance.
    indices = np.argsort(pca.explained_variance_)[::-1]  # Indices of components in descending order of variance
    indices = np.roll(indices, shift=-1)  # Shift indices to align with XZ directions first
    rotation_matrix = pca.components_[indices]

    # Apply the rotation to the vertices
    vertices = np.dot(vertices, rotation_matrix)

    # Compute the maximum dimension of the mesh
    max_dim = np.max(np.abs(vertices))

    # Scale the mesh so that its maximum dimension is 1
    vertices /= max_dim

    return vertices


def create_mlp(features: List[int], activation: str = 'relu', batch_norm: bool = False, softmax: bool = False) -> nn.Sequential:
    activation = activations[activation]
    layers_list = []
    for i in range(len(features) - 1):
        layers_list.append(torch.nn.Linear(features[i], features[i + 1]))
        if batch_norm is True:
            layers_list.append(torch.nn.BatchNorm1d(features[i + 1]))
        if i < len(features) - 2:
            layers_list.append(activation())
    if softmax is True:
        layers_list.append(torch.nn.Softmax(dim=-1))
    model = torch.nn.Sequential(*layers_list)
    return model


def create_gnn(convolutions: List[MessagePassing], activation: str = 'relu', batch_norm: bool = False) -> torch_geometric.nn.Sequential:
    activation = activations[activation]
    layers_list = []
    for i, convolution in enumerate(convolutions):
        layers_list.append((convolution, 'x, edge_index -> x'))
        if batch_norm:
            layers_list.append(torch.nn.BatchNorm1d(convolution.out_channels))
        layers_list.append(activation())
    model = torch_geometric.nn.Sequential('x, edge_index', layers_list)
    return model


def append_moments(x: torch.Tensor) -> torch.Tensor:
    second_order_moments = torch.einsum('bi,bj->bij', x, x)

    # Get the upper triangular indices
    rows, cols = torch.triu_indices(second_order_moments.shape[1], second_order_moments.shape[2])

    # Extract the upper triangular part for each MxM matrix
    upper_triangular_values = second_order_moments[:, rows, cols]

    appended_x = torch.cat((x, upper_triangular_values.view(x.shape[0], -1)), dim=1)
    return appended_x


class ExampleDataset(Dataset):
    def __init__(self):
        super().__init__()
        file_path = Path(__file__).parents[1] / 'data/retinal.obj'
        self._mesh = trimesh.load(file_obj=file_path)
        self._transforms = transforms.Compose([transforms.FaceToEdge(remove_faces=False)])

    def len(self):
        return 1

    def get(self, idx):
        # place mesh in canonical form, aligned with its PCA principal directions
        vertices = self._mesh.vertices.astype(np.float32)
        vertices = canonical_form(vertices=vertices)
        faces = self._mesh.faces

        # calculate principal curvatures at each points
        _, _, pv1, pv2 = igl.principal_curvature(v=vertices, f=faces)

        # calculate gaussian curvature
        k = pv1 * pv2
        data = Data(
            x=torch.from_numpy(vertices),
            k=torch.from_numpy(k),
            face=torch.from_numpy(faces).T)

        # infer edges based on face data
        data = self._transforms(data)

        return data


class ExampleDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self._train_dataset = ExampleDataset()
        self._validation_dataset = ExampleDataset()

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=1)


class ExampleModule(LightningModule):
    def __init__(self):
        super().__init__()
        convolutions = [
            torch_geometric.nn.conv.GATv2Conv(9, 16, head=2),
            torch_geometric.nn.conv.GATv2Conv(16, 32, head=2),
        ]
        self._gnn = create_gnn(convolutions=convolutions, activation='sine', batch_norm=False)
        self._mlp = create_mlp(features=[32, 16, 8, 4, 1], activation='sine', batch_norm=False)
        self._loss_fn = nn.SmoothL1Loss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch: Batch, index: int):
        data = batch[0]
        x = append_moments(x=data.x)
        embeddings = self._gnn(x=x, edge_index=data.edge_index)
        pred_k = self._mlp(embeddings).squeeze(dim=1)
        loss = self._loss_fn(pred_k, data.k)

        return {
            'loss': loss,
            'pred_k': pred_k,
            'gt_k': data.k
        }


class PyGfxCallback(Callback):
    def __init__(self):
        super().__init__()
        self._queue = mp.Queue()
        self._process = None

    @staticmethod
    def _rescale_k(k: np.ndarray) -> np.ndarray:
        # rescale gaussian curvature so it will range between 0 and 1
        min_val = k.min()
        max_val = k.max()
        rescaled_k = (k - min_val) / (max_val - min_val)
        return rescaled_k

    @staticmethod
    def _get_vertex_colors_from_k(k: np.ndarray) -> np.ndarray:
        k = PyGfxCallback._rescale_k(k=k)
        k_one_minus = 1 - k

        # convex combinations between red and blue colors, based on the predicted gaussian curvature
        c1 = np.column_stack((k_one_minus, np.zeros_like(k), np.zeros_like(k), np.ones_like(k)))
        c2 = np.column_stack((np.zeros_like(k), np.zeros_like(k), k, np.ones_like(k)))
        c = c1 + c2

        return c

    @staticmethod
    def _create_world_object_for_mesh(mesh_data: MeshData, k: np.ndarray, wireframe: bool = False, wireframe_thickness=1.5, color: Color = '#ffffff') -> WorldObject:
        c = PyGfxCallback._get_vertex_colors_from_k(k=k)
        geometry = Geometry(
            indices=np.ascontiguousarray(mesh_data.faces),
            positions=np.ascontiguousarray(mesh_data.vertices),
            colors=np.ascontiguousarray(c))

        material = MeshPhongMaterial(
            color=color,
            wireframe=wireframe,
            wireframe_thickness=wireframe_thickness,
            vertex_colors=True)

        mesh = Mesh(
            geometry=geometry,
            material=material)
        return mesh

    @staticmethod
    def _create_world_object_for_text(text_string: str, anchor: str, position: np.ndarray, font_size: int) -> WorldObject:
        geometry = gfx.TextGeometry(
            text=text_string,
            anchor=anchor,
            font_size=font_size,
            screen_space=True)

        material = gfx.TextMaterial(
            color=Color("#ffffff"),
            outline_color=Color('#000000'),
            outline_thickness=0.5)

        text = gfx.Text(
            geometry=geometry,
            material=material)

        text.local.position = position
        return text

    @staticmethod
    def _create_background(top_color: Color, bottom_color: Color) -> Background:
        return Background(geometry=None, material=BackgroundMaterial(top_color, bottom_color))

    @staticmethod
    def _create_background_scene(renderer: Renderer, color: Color) -> Tuple[Viewport, Camera, Background]:
        viewport = Viewport(renderer)
        camera = NDCCamera()
        background = PyGfxCallback._create_background(top_color=color, bottom_color=color)
        return viewport, camera, background

    @staticmethod
    def _create_scene(renderer: Renderer, light_color: Color, background_top_color: Color, background_bottom_color: Color, rect_length: int = 1) -> Tuple[Viewport, Camera, Scene]:
        viewport = Viewport(renderer)
        camera = PerspectiveCamera()
        camera.show_rect(left=-rect_length, right=rect_length, top=-rect_length, bottom=rect_length, view_dir=(-1, -1, -1), up=(0, 0, 1))
        controller = OrbitController(camera=camera, register_events=viewport)
        light = PointLight(color=light_color, intensity=5, decay=0)
        background = PyGfxCallback._create_background(top_color=background_top_color, bottom_color=background_bottom_color)
        scene = Scene()
        scene.add(light)
        scene.add(background)
        return viewport, camera, scene

    @staticmethod
    def _create_text_scene(text_string: str, position: np.ndarray, anchor: str, font_size: int) -> Tuple[Scene, Camera]:
        scene = Scene()
        camera = gfx.ScreenCoordsCamera()
        text = PyGfxCallback._create_world_object_for_text(
            text_string=text_string,
            anchor=anchor,
            position=position,
            font_size=font_size)
        scene.add(text)
        return scene, camera

    @staticmethod
    def _plot_handler(
            in_queue: Queue,
            light_color: Color = Color("#ffffff"),
            background_color: Color = Color('#ffffff'),
            scene_background_top_color: Color = Color("#bbbbbb"),
            scene_background_bottom_color: Color = Color("#666666")):
        border_size = 5.0
        text_position = np.array([10, 10, 0])
        text_font_size = 30
        text_anchor = 'bottom-left'
        scene_state = SceneState()

        renderer = WgpuRenderer(WgpuCanvas())

        background_viewport, background_camera, background_scene = PyGfxCallback._create_background_scene(
            renderer=renderer,
            color=background_color)

        gt_mesh_viewport, gt_mesh_camera, gt_mesh_scene = PyGfxCallback._create_scene(
            renderer=renderer,
            light_color=light_color,
            background_top_color=scene_background_top_color,
            background_bottom_color=scene_background_bottom_color)

        pred_mesh_viewport, pred_mesh_camera, pred_mesh_scene = PyGfxCallback._create_scene(
            renderer=renderer,
            light_color=light_color,
            background_top_color=scene_background_top_color,
            background_bottom_color=scene_background_bottom_color)

        gt_mesh_scene.add(scene_state.gt_group)
        pred_mesh_scene.add(scene_state.pred_group)

        gt_mesh_text_scene, gt_mesh_text_camera = PyGfxCallback._create_text_scene(
            text_string='Ground Truth',
            position=text_position,
            anchor=text_anchor,
            font_size=text_font_size)

        pred_mesh_text_scene, pred_mesh_text_camera = PyGfxCallback._create_text_scene(
            text_string='Prediction',
            position=text_position,
            anchor=text_anchor,
            font_size=text_font_size)

        @renderer.add_event_handler("resize")
        def on_resize(event: Optional[gfx.WindowEvent] = None):
            w, h = renderer.logical_size
            w2, h2 = w / 2, h / 2
            gt_mesh_viewport.rect = 0, 0, w2 - border_size, h
            pred_mesh_viewport.rect = w2 + border_size, 0, w2 - border_size, h

        def animate():
            try:
                mesh_data = cast(MeshData, in_queue.get_nowait())

                # if that's the first mesh-data message, create meshes
                if scene_state.first_data:
                    scene_state.gt_mesh = PyGfxCallback._create_world_object_for_mesh(
                        mesh_data=mesh_data,
                        k=mesh_data.k)
                    scene_state.pred_mesh = PyGfxCallback._create_world_object_for_mesh(
                        mesh_data=mesh_data,
                        k=mesh_data.pred_k)
                    scene_state.gt_group.add(scene_state.gt_mesh)
                    scene_state.pred_group.add(scene_state.pred_mesh)
                    scene_state.first_data = False
                # otherwise, update mesh colors
                else:
                    c_pred = PyGfxCallback._get_vertex_colors_from_k(k=mesh_data.pred_k)
                    scene_state.pred_mesh.geometry.colors = Buffer(data=c_pred)
            except queue.Empty:
                pass

            # place point-light at the camera's position
            gt_mesh_scene.children[0].local.position = gt_mesh_camera.local.position
            pred_mesh_scene.children[0].local.position = pred_mesh_camera.local.position

            # render white background
            background_viewport.render(background_scene, background_camera)

            # render ground-truth mesh
            gt_mesh_viewport.render(gt_mesh_scene, gt_mesh_camera)
            gt_mesh_viewport.render(gt_mesh_text_scene, gt_mesh_text_camera)

            # render prediction mesh
            pred_mesh_viewport.render(pred_mesh_scene, pred_mesh_camera)
            pred_mesh_viewport.render(pred_mesh_text_scene, pred_mesh_text_camera)

            renderer.flush()
            renderer.request_draw()

        on_resize()
        renderer.request_draw(animate)
        run()

    def on_fit_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        self._process = mp.Process(
            target=PyGfxCallback._plot_handler,
            args=(self._queue,))
        self._process.start()

    def on_fit_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        self._process.terminate()
        self._process.join()

    def _on_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0) -> None:
        data = batch[0]

        # create mesh-data message
        vertices = data.x.detach().cpu().numpy()
        faces = data.face.detach().cpu().numpy().T.astype(np.int32)
        k = data.k.detach().cpu().numpy()
        pred_k = outputs['pred_k'].detach().cpu().numpy()
        mesh_data = MeshData(
            vertices=vertices,
            faces=faces,
            k=k,
            pred_k=pred_k)

        # send mesh-data message to the rendering process
        self._queue.put(obj=mesh_data)

        # print current loss
        print(f'Loss: {outputs["loss"]}')

    def on_train_batch_end(self, *args, **kwargs) -> None:
        self._on_batch_end(*args)

    def on_validation_batch_end(self, *args, **kwargs) -> None:
        self._on_batch_end(*args)


if __name__ == "__main__":
    datamodule = ExampleDataModule()
    model = ExampleModule()
    callbacks = [PyGfxCallback()]
    trainer = Trainer(
        max_epochs=-1,
        callbacks=callbacks,
        limit_val_batches=0,
        log_every_n_steps=0,
        num_sanity_val_steps=0,
        enable_progress_bar=False)
    trainer.fit(model=model, datamodule=datamodule)
