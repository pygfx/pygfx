"""
GPU Texture
=============

Example to show (and test) how to create a pygfx Texture from an existing wgpu GPUTexture
which might get drawn to from a different app.

"""
# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la
from wgpu_shadertoy import Shadertoy

# First create the external app with target "offscreen" texture
# shadertoy source: https://www.shadertoy.com/view/XtjfDy by FabriceNeyret2 CC-BY-NC-SA-3.0
shader_code = """
void mainImage(out vec4 O, vec2 u) {
    vec2 U = u+u - iResolution.xy;
    float T = 6.2832, l = length(U) / 30., L = ceil(l) * 6.,
          a = atan(U.x,U.y) - iTime * 2.*(fract(1e4*sin(L))-.5);
    O = .6 + .4* cos( floor(fract(a/T)*L) + vec4(0,23,21,0) )
        - max(0., 9.* max( cos(T*l), cos(a*L) ) - 8. ); }
"""
# it's important to use the same device!
# also the texture needs to support the usage TEXTURE_BINDING for the later use case - this isn't default behaviour. you can set that in canvas.configure()
shadertoy = Shadertoy(shader_code, resolution=(512, 512), offscreen=True, device=gfx.renderers.wgpu.get_shared().device)
gpu_texture = shadertoy._present_context.get_current_texture() # now this might change between different texture objects with the rendercanvas offscreen context
# perhaps we only get a new texture every other frame or so?... needs investigation
texture1 = gfx.Texture.from_gpu(gpu_texture) # use this new class method

# Then create the actual scene, in the visible canvas

canvas = RenderCanvas()  # for gallery scraper, bc we have 2 renderers
renderer2 = gfx.renderers.WgpuRenderer(canvas)
scene2 = gfx.Scene()

geometry2 = gfx.box_geometry(200, 200, 200)
material2 = gfx.MeshPhongMaterial(map=texture1)
cube2 = gfx.Mesh(geometry2, material2)
scene2.add(cube2)

camera2 = gfx.PerspectiveCamera(70, 16 / 9)
camera2.local.z = 400

scene2.add(gfx.AmbientLight(), camera2.add(gfx.DirectionalLight()))


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="xy")
    cube2.local.rotation = la.quat_mul(rot, cube2.local.rotation)

    shadertoy._draw_frame() # draw the external app... (can also be done in a different shedule etc)

    renderer2.render(scene2, camera2)

    renderer2.request_draw()


if __name__ == "__main__":
    renderer2.request_draw(animate)
    loop.run()
