# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

"""
Pytorch Integration
===================

Pygfx Integration with PyTorch Lightning and PyTorch Geometric: A Practical Example.

This code example demonstrates the integration of Pygfx with PyTorch Lightning and PyTorch Geometric for training a
graph neural network on a 3D mesh and simultaneously rendering the results in real-time using Pygfx.

ARCHITECTURE:
The code example follows a modular architecture, separating the concerns of data loading, model definition, training,
and rendering. The main components of the architecture are:

1. Dataset (ExampleDataset):
   - Responsible for loading and preprocessing the 3D mesh data.
   - Calculates the Gaussian curvature at each vertex using principal curvatures.
   - Applies transforms to infer edges based on face data.
   - Provides a single data sample containing vertices, faces, and Gaussian curvature values.

2. Data Module (ExampleDataModule):
   - Defines the train and validation datasets using the ExampleDataset.
   - Provides data loaders for training and validation.
   - Integrates seamlessly with the PyTorch Lightning training pipeline.

3. Model (ExampleModule):
   - Defines the graph neural network architecture using PyTorch Lightning.
   - Uses a combination of graph convolutional layers (ResGatedGraphConv) for message passing and an MLP for final prediction.
   - Implements the training step, including the forward pass and loss calculation.
   - Configures the optimizer for training.

4. Rendering (SceneHandler):
   - Handles the creation and rendering of the Pygfx scene.
   - Creates two viewports: one for the ground truth mesh and another for the predicted mesh.
   - Receives mesh data messages from the training process via a multiprocessing queue.
   - Updates the meshes in the scene based on the received mesh data.
   - Colors the meshes based on the Gaussian curvature values.

5. Callback (PyGfxCallback):
   - Facilitates communication between the training process and the rendering process.
   - Creates a separate process for Pygfx rendering.
   - Sends mesh data messages to the rendering process via a multiprocessing queue.
   - Updates the meshes in the Pygfx scene based on the training progress and results.

INTERACTION:
The interaction between the different components of the code example is as follows:

1. The ExampleDataset loads and preprocesses the 3D mesh data, calculating the Gaussian curvature at each vertex.
   It provides a single data sample containing vertices, faces, and Gaussian curvature values.

2. The ExampleDataModule defines the train and validation datasets using the ExampleDataset and provides data loaders
   for training and validation.

3. The ExampleModule defines the graph neural network architecture using PyTorch Lightning. It implements the training
   step, including the forward pass and loss calculation, and configures the optimizer for training.

4. During training, the PyTorch Lightning Trainer uses the ExampleDataModule to load the data and the ExampleModule to
   define the model and training process.

5. The PyGfxCallback, registered as a callback with the Trainer, creates a separate process for Pygfx rendering using
   the SceneHandler.

6. After each training or validation batch, the PyGfxCallback sends the mesh data (vertices, faces, ground truth
   curvature, and predicted curvature) to the rendering process via a multiprocessing queue.

7. The SceneHandler, running in a separate process, receives the mesh data messages from the queue and updates the
   meshes in the Pygfx scene accordingly. It colors the meshes based on the Gaussian curvature values.

8. The Pygfx rendering process continuously renders the scene, displaying the ground truth and predicted meshes in
   real-time as the training progresses.

This architecture allows for seamless integration of Pygfx with PyTorch Lightning and PyTorch Geometric, enabling
real-time visualization of the training progress and results on a 3D mesh. The modular design separates the concerns
of data loading, model definition, training, and rendering, making the code more maintainable and extensible.

In order to run this example, you should install the following dependencies:

    pip install libigl lightning torch_geometric

"""

# standard library
from pathlib import Path
from typing import Optional, Tuple, List, Any, cast
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
from pygfx.renderers import WgpuRenderer

# wgpu
from wgpu.gui.auto import WgpuCanvas, run

# pytorch
import torch
from torch import optim, nn
from torch.nn import ReLU, GELU, Tanh, Sigmoid
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

# pytorch lightning
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.datamodule import LightningDataModule

# torch geometric
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Dataset, Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric import transforms


@dataclass
class MeshData:
    """
    A dataclass representing mesh data.

    This dataclass is used to store and transfer mesh data between the training process and the rendering process.
    It contains the vertices, faces, ground truth Gaussian curvature values, and predicted Gaussian curvature values
    for a given mesh.

    Attributes:
        vertices (np.ndarray): The vertices of the mesh, stored as a NumPy array of shape (num_vertices, 3).
        faces (np.ndarray): The faces of the mesh, stored as a NumPy array of shape (num_faces, 3).
        k (np.ndarray): The ground truth Gaussian curvature values at each vertex, stored as a NumPy array of shape (num_vertices,).
        pred_k (np.ndarray): The predicted Gaussian curvature values at each vertex, stored as a NumPy array of shape (num_vertices,).
    """

    vertices: np.ndarray
    faces: np.ndarray
    k: np.ndarray
    pred_k: np.ndarray


@dataclass
class SceneState:
    """
    A dataclass representing the state of the pygfx scene.

    This dataclass is used to store and manage the state of the pygfx scene, including the ground truth mesh,
    predicted mesh, and their corresponding group objects. It also keeps track of whether it's the first data
    received by the scene.

    Attributes:
        gt_group (gfx.Group): A group object for the ground truth mesh, used to organize and manipulate the mesh in the scene.
        pred_group (gfx.Group): A group object for the predicted mesh, used to organize and manipulate the mesh in the scene.
        gt_mesh (gfx.WorldObject): The ground truth mesh object, representing the actual mesh in the scene.
        pred_mesh (gfx.WorldObject): The predicted mesh object, representing the predicted mesh in the scene.
        first_data (bool): A flag indicating if it's the first data received by the scene. Used to determine if the meshes need to be created or updated.
    """

    gt_group: gfx.Group = gfx.Group()  # noqa: RUF009
    pred_group: gfx.Group = gfx.Group()  # noqa: RUF009
    gt_mesh: gfx.WorldObject = None
    pred_mesh: gfx.WorldObject = None
    first_data: bool = True


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


# a few pytorch activations to experiment with
activations = {
    "relu": ReLU,
    "gelu": GELU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "sine": Sine,
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
    indices = np.argsort(pca.explained_variance_)[
        ::-1
    ]  # Indices of components in descending order of variance
    indices = np.roll(
        indices, shift=-1
    )  # Shift indices to align with XZ directions first
    rotation_matrix = pca.components_[indices]

    # Apply the rotation to the vertices
    vertices = np.dot(vertices, rotation_matrix)

    # Compute the maximum dimension of the mesh
    max_dim = np.max(np.abs(vertices))

    # Scale the mesh so that its maximum dimension is 1
    vertices /= max_dim

    return vertices


def create_mlp(
    features: List[int],
    activation: str = "relu",
    batch_norm: bool = False,
    softmax: bool = False,
) -> nn.Sequential:
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


def create_gnn(
    convolutions: List[MessagePassing],
    activation: str = "relu",
    batch_norm: bool = False,
) -> torch_geometric.nn.Sequential:
    activation = activations[activation]
    layers_list = []
    for _i, convolution in enumerate(convolutions):
        layers_list.append((convolution, "x, edge_index -> x"))
        if batch_norm:
            layers_list.append(torch.nn.BatchNorm1d(convolution.out_channels))
        layers_list.append(activation())
    model = torch_geometric.nn.Sequential("x, edge_index", layers_list)
    return model


class ExampleDataset(Dataset):
    """
    A PyTorch Geometric dataset class for loading and preprocessing a 3D mesh.

    This dataset class is responsible for loading a 3D mesh file, calculating the Gaussian curvature at each vertex
    using the principal curvatures, and preparing the data for the graph neural network. It applies transforms to
    infer edges based on face data.

    The dataset is initialized with a specific mesh file path and a set of transforms. The `get` method is used to
    retrieve a single data sample, which includes the vertices, faces, and Gaussian curvature values. The mesh is
    placed in a canonical form, aligned with its PCA principal directions, before being returned.

    Methods:
        __init__(): Initializes the dataset with a specific mesh file path and a set of transforms.
        len(): Returns the length of the dataset (always 1 in this case).
        get(idx): Retrieves a single data sample from the dataset, including the vertices, faces, and Gaussian curvature values.
    """

    def __init__(self):
        super().__init__()
        file_path = Path(__file__).parents[1] / "data/retinal.obj"
        self._mesh = trimesh.load(file_obj=file_path)
        self._transforms = transforms.Compose(
            [transforms.FaceToEdge(remove_faces=False)]
        )

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
            face=torch.from_numpy(faces).T,
        )

        # infer edges based on face data
        data = self._transforms(data)

        return data


class ExampleDataModule(LightningDataModule):
    """
    A PyTorch Lightning data module for the example dataset.

    This data module is responsible for defining the train and validation datasets using the ExampleDataset class.
    It provides the data loaders for training and validation, allowing seamless integration with the PyTorch Lightning
    training pipeline.

    The data module is initialized with train and validation datasets, which are instances of the ExampleDataset class.
    The `train_dataloader` method returns the data loader for the training dataset, specifying the batch size.

    Methods:
        __init__(): Initializes the data module with train and validation datasets.
        train_dataloader(): Returns the data loader for the training dataset.
    """

    def __init__(self):
        super().__init__()
        self._train_dataset = ExampleDataset()
        self._validation_dataset = ExampleDataset()

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=1)


class ExampleModule(LightningModule):
    """
    A PyTorch Lightning module defining the graph neural network architecture.

    This module defines the architecture of the graph neural network used for predicting the Gaussian curvature
    of a 3D mesh. It uses a combination of graph convolutional layers (ResGatedGraphConv) for message passing
    and a multilayer perceptron (MLP) for the final prediction.

    The module is initialized with a specific configuration of graph convolutional layers and MLP layers. The
    `configure_optimizers` method defines the optimizer used for training the model. The `training_step` method
    defines the forward pass and loss calculation for a single training step.

    Methods:
        __init__(): Initializes the module with a specific configuration of graph convolutional layers and MLP layers.
        configure_optimizers(): Defines the optimizer used for training the model.
        training_step(batch, index): Defines the forward pass and loss calculation for a single training step.
    """

    def __init__(self):
        super().__init__()
        convolutions = [
            torch_geometric.nn.conv.ResGatedGraphConv(3, 16),
            torch_geometric.nn.conv.ResGatedGraphConv(16, 32),
            torch_geometric.nn.conv.ResGatedGraphConv(32, 64),
            torch_geometric.nn.conv.ResGatedGraphConv(64, 128),
        ]
        self._gnn = create_gnn(
            convolutions=convolutions, activation="gelu", batch_norm=False
        )
        self._mlp = create_mlp(
            features=[128, 64, 32, 16, 8, 4, 1], activation="gelu", batch_norm=False
        )
        self._loss_fn = nn.SmoothL1Loss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch: Batch, index: int):
        data = batch[0]
        embeddings = self._gnn(x=data.x, edge_index=data.edge_index)
        pred_k = self._mlp(embeddings).squeeze(dim=1)
        loss = self._loss_fn(pred_k, data.k)

        return {"loss": loss, "pred_k": pred_k, "gt_k": data.k}


class SceneHandler:
    """
    A class for handling the creation and rendering of the pygfx scene.

    This class is responsible for creating and managing the pygfx scene, which includes two viewports: one for
    displaying the ground truth mesh and another for the predicted mesh. It receives mesh data messages from the
    training process via a multiprocessing queue and updates the meshes in the scene accordingly.

    The class provides a `start` method that initializes the scene, viewports, cameras, and lighting. It sets up
    the rendering loop and handles the updating of the meshes based on the received mesh data messages. The meshes
    are colored based on the Gaussian curvature values.

    Methods:
        start(in_queue, light_color, background_color, scene_background_top_color, scene_background_bottom_color):
            Initializes the scene, viewports, cameras, and lighting, and starts the rendering loop.
            Receives mesh data messages from the training process via a multiprocessing queue and updates the meshes in the scene.
    """

    @staticmethod
    def _rescale_k(k: np.ndarray) -> np.ndarray:
        # rescale gaussian curvature so it will range between 0 and 1
        min_val = k.min()
        max_val = k.max()
        rescaled_k = (k - min_val) / (max_val - min_val)
        return rescaled_k

    @staticmethod
    def _get_vertex_colors_from_k(k: np.ndarray) -> np.ndarray:
        k = SceneHandler._rescale_k(k=k)
        k_one_minus = 1 - k

        # convex combinations between red and blue colors, based on the predicted gaussian curvature
        c1 = np.column_stack(
            (k_one_minus, np.zeros_like(k), np.zeros_like(k), np.ones_like(k))
        )
        c2 = np.column_stack((np.zeros_like(k), np.zeros_like(k), k, np.ones_like(k)))
        c = c1 + c2

        return c

    @staticmethod
    def _create_world_object_for_mesh(
        mesh_data: MeshData, k: np.ndarray, color: gfx.Color = "#ffffff"
    ) -> gfx.WorldObject:
        c = SceneHandler._get_vertex_colors_from_k(k=k)
        geometry = gfx.Geometry(
            indices=np.ascontiguousarray(mesh_data.faces),
            positions=np.ascontiguousarray(mesh_data.vertices),
            colors=np.ascontiguousarray(c),
        )

        material = gfx.MeshPhongMaterial(color=color, color_mode="vertex")

        mesh = gfx.Mesh(geometry=geometry, material=material)
        return mesh

    @staticmethod
    def _create_world_object_for_text(
        text_string: str, anchor: str, position: np.ndarray, font_size: int
    ) -> gfx.WorldObject:
        text = gfx.Text(
            text=text_string,
            anchor=anchor,
            font_size=font_size,
            screen_space=True,
            material=gfx.TextMaterial(
                color=gfx.Color("#ffffff"),
                outline_color=gfx.Color("#000000"),
                outline_thickness=0.5,
            ),
        )

        text.local.position = position
        return text

    @staticmethod
    def _create_background(
        top_color: gfx.Color, bottom_color: gfx.Color
    ) -> gfx.Background:
        return gfx.Background(
            geometry=None, material=gfx.BackgroundMaterial(top_color, bottom_color)
        )

    @staticmethod
    def _create_background_scene(
        renderer: gfx.Renderer, color: gfx.Color
    ) -> Tuple[gfx.Viewport, gfx.Camera, gfx.Background]:
        viewport = gfx.Viewport(renderer)
        camera = gfx.NDCCamera()
        background = SceneHandler._create_background(
            top_color=color, bottom_color=color
        )
        return viewport, camera, background

    @staticmethod
    def _create_scene(
        renderer: gfx.Renderer,
        light_color: gfx.Color,
        background_top_color: gfx.Color,
        background_bottom_color: gfx.Color,
        rect_length: int = 1,
    ) -> Tuple[gfx.Viewport, gfx.Camera, gfx.Scene]:
        viewport = gfx.Viewport(renderer)
        camera = gfx.PerspectiveCamera()
        camera.show_rect(
            left=-rect_length,
            right=rect_length,
            top=-rect_length,
            bottom=rect_length,
            view_dir=(-1, -1, -1),
            up=(0, 0, 1),
        )
        _ = gfx.OrbitController(camera=camera, register_events=viewport)
        light = gfx.PointLight(color=light_color, intensity=5, decay=0)
        background = SceneHandler._create_background(
            top_color=background_top_color, bottom_color=background_bottom_color
        )
        scene = gfx.Scene()
        scene.add(camera.add(light))
        scene.add(background)
        return viewport, camera, scene

    @staticmethod
    def _create_text_scene(
        text_string: str, position: np.ndarray, anchor: str, font_size: int
    ) -> Tuple[gfx.Scene, gfx.Camera]:
        scene = gfx.Scene()
        camera = gfx.ScreenCoordsCamera()
        text = SceneHandler._create_world_object_for_text(
            text_string=text_string,
            anchor=anchor,
            position=position,
            font_size=font_size,
        )
        scene.add(text)
        return scene, camera

    @staticmethod
    def _animate():
        try:
            mesh_data = cast("MeshData", SceneHandler._in_queue.get_nowait())

            # if that's the first mesh-data message, create meshes
            if SceneHandler._scene_state.first_data:
                SceneHandler._scene_state.gt_mesh = (
                    SceneHandler._create_world_object_for_mesh(
                        mesh_data=mesh_data, k=mesh_data.k
                    )
                )
                SceneHandler._scene_state.pred_mesh = (
                    SceneHandler._create_world_object_for_mesh(
                        mesh_data=mesh_data, k=mesh_data.pred_k
                    )
                )
                SceneHandler._scene_state.gt_group.add(
                    SceneHandler._scene_state.gt_mesh
                )
                SceneHandler._scene_state.pred_group.add(
                    SceneHandler._scene_state.pred_mesh
                )
                SceneHandler._scene_state.first_data = False
            # otherwise, update mesh colors
            else:
                c_pred = SceneHandler._get_vertex_colors_from_k(k=mesh_data.pred_k)
                SceneHandler._scene_state.pred_mesh.geometry.colors.data[:] = c_pred
                SceneHandler._scene_state.pred_mesh.geometry.colors.update_full()
        except queue.Empty:
            pass

        # render white background
        SceneHandler._background_viewport.render(
            SceneHandler._background_scene, SceneHandler._background_camera
        )

        # render ground-truth mesh
        SceneHandler._gt_mesh_viewport.render(
            SceneHandler._gt_mesh_scene, SceneHandler._gt_mesh_camera
        )
        SceneHandler._gt_mesh_viewport.render(
            SceneHandler._gt_mesh_text_scene, SceneHandler._gt_mesh_text_camera
        )

        # render prediction mesh
        SceneHandler._pred_mesh_viewport.render(
            SceneHandler._pred_mesh_scene, SceneHandler._pred_mesh_camera
        )
        SceneHandler._pred_mesh_viewport.render(
            SceneHandler._pred_mesh_text_scene, SceneHandler._pred_mesh_text_camera
        )

        SceneHandler._renderer.flush()
        SceneHandler._renderer.request_draw()

    @staticmethod
    def start(
        in_queue: Queue,
        light_color: gfx.Color = "#ffffff",
        background_color: gfx.Color = "#ffffff",
        scene_background_top_color: gfx.Color = "#bbbbbb",
        scene_background_bottom_color: gfx.Color = "#666666",
    ):
        border_size = 5.0
        text_position = np.array([10, 10, 0])
        text_font_size = 30
        text_anchor = "bottom-left"

        SceneHandler._scene_state = SceneState()
        SceneHandler._in_queue = in_queue
        SceneHandler._renderer = WgpuRenderer(WgpuCanvas())

        (
            SceneHandler._background_viewport,
            SceneHandler._background_camera,
            SceneHandler._background_scene,
        ) = SceneHandler._create_background_scene(
            renderer=SceneHandler._renderer, color=background_color
        )

        (
            SceneHandler._gt_mesh_viewport,
            SceneHandler._gt_mesh_camera,
            SceneHandler._gt_mesh_scene,
        ) = SceneHandler._create_scene(
            renderer=SceneHandler._renderer,
            light_color=light_color,
            background_top_color=scene_background_top_color,
            background_bottom_color=scene_background_bottom_color,
        )

        (
            SceneHandler._pred_mesh_viewport,
            SceneHandler._pred_mesh_camera,
            SceneHandler._pred_mesh_scene,
        ) = SceneHandler._create_scene(
            renderer=SceneHandler._renderer,
            light_color=light_color,
            background_top_color=scene_background_top_color,
            background_bottom_color=scene_background_bottom_color,
        )

        SceneHandler._gt_mesh_scene.add(SceneHandler._scene_state.gt_group)
        SceneHandler._pred_mesh_scene.add(SceneHandler._scene_state.pred_group)

        SceneHandler._gt_mesh_text_scene, SceneHandler._gt_mesh_text_camera = (
            SceneHandler._create_text_scene(
                text_string="Ground Truth",
                position=text_position,
                anchor=text_anchor,
                font_size=text_font_size,
            )
        )

        SceneHandler._pred_mesh_text_scene, SceneHandler._pred_mesh_text_camera = (
            SceneHandler._create_text_scene(
                text_string="Prediction",
                position=text_position,
                anchor=text_anchor,
                font_size=text_font_size,
            )
        )

        @SceneHandler._renderer.add_event_handler("resize")
        def on_resize(event: Optional[gfx.WindowEvent] = None):
            w, h = SceneHandler._renderer.logical_size
            w2 = w / 2
            SceneHandler._gt_mesh_viewport.rect = 0, 0, w2 - border_size, h
            SceneHandler._pred_mesh_viewport.rect = (
                w2 + border_size,
                0,
                w2 - border_size,
                h,
            )

        on_resize()
        SceneHandler._renderer.request_draw(SceneHandler._animate)
        run()


class PyGfxCallback(Callback):
    """
    A PyTorch Lightning callback for handling the communication between the training process and the rendering process.

    This callback is responsible for creating a separate process for the pygfx rendering and sending mesh data messages
    to the rendering process via a multiprocessing queue. It updates the meshes in the pygfx scene based on the training
    progress and results.

    The callback is initialized with a multiprocessing queue and a process for the rendering. The `on_fit_start` method
    is called at the start of the training loop and starts the rendering process. The `on_fit_end` method is called at
    the end of the training loop and terminates the rendering process. The `_on_batch_end` method is called after each
    training or validation batch and sends the mesh data message to the rendering process.

    Methods:
        __init__(): Initializes the callback with a multiprocessing queue and a process for the rendering.
        on_fit_start(trainer, pl_module): Called at the start of the training loop. Starts the rendering process.
        on_fit_end(trainer, pl_module): Called at the end of the training loop. Terminates the rendering process.
        _on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            Called after each training or validation batch. Sends the mesh data message to the rendering process.
    """

    def __init__(self):
        super().__init__()
        self._queue = mp.Queue()
        self._process = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._process = mp.Process(target=SceneHandler.start, args=(self._queue,))
        self._process.start()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._process.terminate()
        self._process.join()

    def _on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        data = batch[0]

        # create mesh-data message
        vertices = data.x.detach().cpu().numpy()
        faces = data.face.detach().cpu().numpy().T.astype(np.int32)
        k = data.k.detach().cpu().numpy()
        pred_k = outputs["pred_k"].detach().cpu().numpy()
        mesh_data = MeshData(vertices=vertices, faces=faces, k=k, pred_k=pred_k)

        # send mesh-data message to the rendering process
        self._queue.put(obj=mesh_data)

        # print current loss
        print(f"Loss: {outputs['loss']}")

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
        enable_progress_bar=False,
    )
    trainer.fit(model=model, datamodule=datamodule)
