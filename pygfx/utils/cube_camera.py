import wgpu
from ..linalg import Vector3
from ..objects._base import WorldObject
from ..cameras import PerspectiveCamera
from ..resources import Texture


class CubeCamera(WorldObject):
    """
    Create a camera array to help create a cube texture map from a viewpoint in a scene.

    Note that the texture data will be written directly to the internal "GPUTexture" object,
    not to the "data" attribute of target. That is, its data cannot be accessed from the CPU.
    """

    def __init__(self, near=0.1, far=1000):
        super().__init__()

        fov = 90
        aspect = 1

        # By convention, cube maps are specified in a coordinate system in which positive-x is to the right when looking at the positive-z axis,
        # that is, it using a left-handed coordinate system.
        # Since gfx uses a right-handed coordinate system, environment maps used in gfx will have pos-x and neg-x swapped.
        # so camrea_px is actually looking at the neg-x direction, and camera_nx is looking at the pos-x direction.

        camera_px = PerspectiveCamera(fov, aspect, near, far)
        camera_px.up.set(0, 1, 0)
        camera_px.look_at(Vector3(-1, 0, 0))
        self.add(camera_px)

        camera_nx = PerspectiveCamera(fov, aspect, near, far)
        camera_nx.up.set(0, 1, 0)
        camera_nx.look_at(Vector3(1, 0, 0))
        self.add(camera_nx)

        camera_py = PerspectiveCamera(fov, aspect, near, far)
        camera_py.up.set(0, 0, -1)
        camera_py.look_at(Vector3(0, 1, 0))
        self.add(camera_py)

        camera_ny = PerspectiveCamera(fov, aspect, near, far)
        camera_ny.up.set(0, 0, 1)
        camera_ny.look_at(Vector3(0, -1, 0))
        self.add(camera_ny)

        camera_pz = PerspectiveCamera(fov, aspect, near, far)
        camera_pz.up.set(0, 1, 0)
        camera_pz.look_at(Vector3(0, 0, 1))
        self.add(camera_pz)

        camera_nz = PerspectiveCamera(fov, aspect, near, far)
        camera_nz.up.set(0, 1, 0)
        camera_nz.look_at(Vector3(0, 0, -1))
        self.add(camera_nz)

    def _is_cube_texture(self, texture):
        if not isinstance(texture, Texture):
            return False
        return texture.dim == 2 and len(texture.size) == 3 and texture.size[2] == 6

    def update(self, renderer, scene, target):

        assert self._is_cube_texture(target), "Target must be a cube texture"

        camera_px, camera_nx, camera_py, camera_ny, camera_pz, camera_nz = self.children

        current_target = renderer._target
        current_target_tex_format = renderer._target_tex_format

        # We don't need to reconfigure the target context, so just access the private "_target" attribute here.

        renderer._target_tex_format = target.format
        if getattr(target, "_wgpu_texture", (-1, None))[1] is None:
            target._wgpu_usage |= wgpu.TextureUsage.RENDER_ATTACHMENT
            target._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING

        renderer._target = target.get_view(view_dim="2d", layer_range=range(1))
        renderer.render(scene, camera_px)

        renderer._target = target.get_view(view_dim="2d", layer_range=range(1, 2))
        renderer.render(scene, camera_nx)

        renderer._target = target.get_view(view_dim="2d", layer_range=range(2, 3))
        renderer.render(scene, camera_py)

        renderer._target = target.get_view(view_dim="2d", layer_range=range(3, 4))
        renderer.render(scene, camera_ny)

        renderer._target = target.get_view(view_dim="2d", layer_range=range(4, 5))
        renderer.render(scene, camera_pz)

        renderer._target = target.get_view(view_dim="2d", layer_range=range(5, 6))
        renderer.render(scene, camera_nz)

        renderer._target = current_target
        renderer._target_tex_format = current_target_tex_format

        return target
