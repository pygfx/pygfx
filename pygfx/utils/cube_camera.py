from ..objects._base import WorldObject
from ..cameras import PerspectiveCamera
from ..resources import Texture
from ..renderers import WgpuRenderer
from ..renderers.wgpu import GfxTextureView


def _is_cube_texture(texture):
    if not isinstance(texture, Texture):
        return False
    return texture.dim == 2 and len(texture.size) == 3 and texture.size[2] == 6


class _CubeCameraRenderer(WgpuRenderer):
    def __init__(self, target, blend_mode="default"):
        assert _is_cube_texture(target), "target must be a cube texture"

        super().__init__(target, blend_mode=blend_mode)

        # Pre generate views of different layers of the cube texture
        self._target_views = []
        for layer in range(6):
            self._target_views.append(
                GfxTextureView(
                    target,
                    view_dim="2d",
                    layer_range=(layer, layer + 1),
                    mip_range=(0, 1),
                )
            )

    def flush(self, layer):
        super().flush(self._target_views[layer])


class CubeCamera(WorldObject):
    """
    Create a camera array to help create a cube texture map from a viewpoint in a scene.

    Note that the texture data will be written directly to the internal "GPUTexture" object,
    not to the "data" attribute of target. That is, its data cannot be accessed from the CPU.
    """

    def __init__(self, target, near=0.1, far=1000, blend_mode="default"):
        super().__init__()

        self._renderer = _CubeCameraRenderer(target, blend_mode=blend_mode)

        fov = 90
        aspect = 1
        depth_range = near, far

        # By convention, cube maps are specified in a coordinate system in which positive-x is to the right when looking at the positive-z axis,
        # that is, it using a left-handed coordinate system.
        # Since gfx uses a right-handed coordinate system, environment maps used in gfx will have pos-x and neg-x swapped.
        # so camrea_px is actually looking at the neg-x direction, and camera_nx is looking at the pos-x direction.

        camera_px = PerspectiveCamera(fov, aspect, depth_range=depth_range)
        camera_px.world.reference_up = (0, 1, 0)
        camera_px.look_at((-1, 0, 0))
        self.add(camera_px)

        camera_nx = PerspectiveCamera(fov, aspect, depth_range=depth_range)
        camera_nx.world.reference_up = (0, 1, 0)
        camera_nx.look_at((1, 0, 0))
        self.add(camera_nx)

        camera_py = PerspectiveCamera(fov, aspect, depth_range=depth_range)
        camera_py.world.reference_up = (0, 0, -1)
        camera_py.look_at((0, 1, 0))
        self.add(camera_py)

        camera_ny = PerspectiveCamera(fov, aspect, depth_range=depth_range)
        camera_ny.world.reference_up = (0, 0, 1)
        camera_ny.look_at((0, -1, 0))
        self.add(camera_ny)

        camera_pz = PerspectiveCamera(fov, aspect, depth_range=depth_range)
        camera_pz.world.reference_up = (0, 1, 0)
        camera_pz.look_at((0, 0, 1))
        self.add(camera_pz)

        camera_nz = PerspectiveCamera(fov, aspect, depth_range=depth_range)
        camera_nz.world.reference_up = (0, 1, 0)
        camera_nz.look_at((0, 0, -1))
        self.add(camera_nz)

    @property
    def renderer(self):
        """The renderer used to render the scene to the cube texture."""

        return self._renderer

    def render(self, scene):
        """Render the scene from the cube camera's perspective, and write the result to the target texture."""

        renderer = self.renderer
        cameras = self.children
        assert len(cameras) == 6

        for layer, camera in enumerate(cameras):
            renderer.render(scene, camera, flush=False)
            renderer.flush(layer)
