import wgpu
import time
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu import Binding, WorldObjectShader, RenderMask


class ShadertoyMaterial(gfx.Material):
    uniform_type = dict(
        resolution="2xf4",
        time="f4",
        time_delta="f4",
        mouse="2xf4",
    )

    def __init__(self, *, resolution=(640, 480), **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution
        self._last_time = time.time()
        self.time_delta = 0

    @property
    def resolution(self):
        """The uniform resolution of the shadertoy."""
        return self.uniform_buffer.data["resolution"]

    @resolution.setter
    def resolution(self, resolution):
        self.uniform_buffer.data["resolution"] = resolution
        self.uniform_buffer.update_range(0, 1)

    @property
    def time(self):
        """The uniform time of the shadertoy."""
        return self.uniform_buffer.data["time"]

    @time.setter
    def time(self, time):
        self.uniform_buffer.data["time"] = time
        self.uniform_buffer.update_range(0, 1)

    @property
    def time_delta(self):
        """The uniform time_delta of the shadertoy."""
        return self.uniform_buffer.data["time_delta"]

    @time_delta.setter
    def time_delta(self, time_delta):
        self.uniform_buffer.data["time_delta"] = time_delta
        self.uniform_buffer.update_range(0, 1)

    @property
    def mouse(self):
        """The uniform mouse of the shadertoy."""
        return self.uniform_buffer.data["mouse"]

    @mouse.setter
    def mouse(self, mouse):
        self.uniform_buffer.data["mouse"] = mouse
        self.uniform_buffer.update_range(0, 1)

    def update(self):
        now = time.time()
        if self._last_time == 0:
            self._last_time = now

        self.time_delta = now - self._last_time
        self._last_time = now

        self.time += self.time_delta


class Shadertoy(gfx.WorldObject):

    """
    @param shader_code: The shader code to use.

    The code must contain a `fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{}` function.
    It has a parameter `frag_coord` which is the current pixel coordinate (in range 0..resolution),
    and it must return a vec4 color, which is the color of the pixel at that coordinate.

    some built-in variables are available: `i_time`, `i_tine_delta`, `i_resolution`, `i_mouse`
    todo: add more built-in variables

    @param resolution: The resolution of the shadertoy.
    """

    def __init__(self, shader_code: str, *, resolution=(640, 480), **kwargs):

        material = ShadertoyMaterial(resolution=resolution)

        super().__init__(None, material, **kwargs)

        self._renderer = gfx.WgpuRenderer(WgpuCanvas(max_fps=60, size=resolution))
        self._camera = gfx.NDCCamera()  # Does not actually use the camera
        self._scene = gfx.Scene()
        self._scene.add(self)

        self.shader_code = shader_code

        @self._renderer.add_event_handler("resize")
        def resize(event=None):
            w, h = self._renderer.logical_size
            self.material.resolution = (w, h)

        @self._renderer.add_event_handler("pointer_move", "pointer_down")
        def mouse_move(event):
            xy = event.x, event.y
            if event.button == 1 or 1 in event.buttons:
                self.material.mouse = xy

    def update(self):
        self.material.update()

    def show(self):
        def animate():
            self.update()
            self._renderer.render(self._scene, self._camera)
            self._renderer.request_draw()

        self._renderer.request_draw(animate)
        run()


@gfx.renderers.wgpu.register_wgpu_render_function(Shadertoy, ShadertoyMaterial)
class ShadertoyShader(WorldObjectShader):

    # Mark as render-shader (as opposed to compute-shader)
    type = "render"

    def __init__(self, wobject, **kwargs):
        super().__init__(wobject, **kwargs)
        self.shader_code = wobject.shader_code

    def get_bindings(self, wobject, shared):
        # Our only binding is a uniform buffer
        bindings = {
            0: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        # We draw triangles, no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        # Since we draw only one triangle we need just 3 vertices.
        # Our triangle is opaque (render mask 1).
        return {
            "indices": (3, 1),
            "render_mask": RenderMask.opaque,
        }

    def get_code(self):
        # Here we put together the full (templated) shader code
        return (
            self.code_definitions()
            + self.code_vertex()
            + self.code_global_variables()
            + self.shader_code
            + self.code_fragment()
        )

    def code_vertex(self):
        return """

        struct Varyings {
            @builtin(position) position : vec4<f32>,
            @location(0) uv : vec2<f32>,
        };


        @stage(vertex)
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
            var out: Varyings;
            if (index == u32(0)) {
                out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
                out.uv = vec2<f32>(0.0, 1.0);
            } else if (index == u32(1)) {
                out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
                out.uv = vec2<f32>(2.0, 1.0);
            } else {
                out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
                out.uv = vec2<f32>(0.0, -1.0);
            }
            return out;

        }
        """

    def code_global_variables(self):
        return """

        var<private> i_time: f32;
        var<private> i_resolution: vec2<f32>;
        var<private> i_time_delta: f32;
        var<private> i_mouse: vec2<f32>;

        // TODO: more global variables
        // var<private> i_frag_coord: vec2<f32>;


        """

    def code_fragment(self):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };

        fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
            // In Python, the below reads as
            // c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
            let t = color / 12.92;
            return select(f, t, color <= vec3<f32>(0.04045));
        }

        @stage(fragment)
        fn fs_main(in: Varyings) -> FragmentOutput {

            i_time = u_material.time;
            i_resolution = u_material.resolution;
            i_time_delta = u_material.time_delta;
            i_mouse = u_material.mouse;
            i_mouse.y = i_resolution.y - i_mouse.y;


            let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
            let frag_coord = uv * i_resolution;

            var fragColor = shader_main(frag_coord);

            var out: FragmentOutput;
            // out.color = vec4<f32>(fragColor.rgb, 1.0);
            out.color = vec4<f32>(srgb2physical(fragColor.rgb), 1.0);
            return out;
        }
        """


if __name__ == "__main__":

    # A simple example

    shader = Shadertoy(
        """
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
        let uv = frag_coord / i_resolution;

        if ( length(frag_coord - i_mouse) < 20.0 ) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }else{
            return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
        }

    }

    """,
        resolution=(800, 450),
    )

    shader.show()
