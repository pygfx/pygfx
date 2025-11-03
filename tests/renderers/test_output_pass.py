"""
Tests for the output pass, which implements downsampling and upsampling with a variety of kernels.

We test the generated wgsl for the expected number of texture lookups,
and we run the pass to make sure it's error is within bounds, also for
the various optimizations that we apply.
"""

from pygfx.renderers.wgpu.engine.effectpasses import OutputPass, apply_templating

import wgpu
import numpy as np
import imageio.v3 as iio


# %% Testing generated source code


class MyOutpusPass(OutputPass):
    def set_scale_factor(self, factor):
        self._set_template_var(scaleFactor=float(factor))

    def resolve_wgsl(self):
        wgsl = apply_templating(self.wgsl, **self._template_vars)

        lines = []
        for line in wgsl.splitlines():
            if "textureSample" in line or "textureLoad" in line:
                if not line.strip().startswith("//"):
                    lines.append(line)
        return wgsl, lines


def test_outpass_scale_is_1():
    p = MyOutpusPass()

    p.set_scale_factor(1)

    # Always use linear filtering when scaleFactor == 1
    p.filter = "mitchell"
    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]

    # Also with other filter
    p.filter = "linear"
    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]

    # But can force nearest
    p.filter = "nearest"
    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordNear" in lines[0]


def test_outpass_filter_nearest():
    p = MyOutpusPass()
    p.filter = "nearest"

    p.set_scale_factor(0.5)  # upsampling, source is smaller

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordNear" in lines[0]

    p.set_scale_factor(2)  # downsampling, source is larger

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordNear" in lines[0]


def test_outpass_filter_linear():
    p = MyOutpusPass()
    p.filter = "linear"

    p.set_scale_factor(0.5)  # upsampling, source is smaller

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]

    p.set_scale_factor(2)  # downsampling, source is larger

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]


def test_outpass_filter_disk():
    _test_outpass_filter_medium("disk")


def test_outpass_filter_tent():
    _test_outpass_filter_medium("tent")


def _test_outpass_filter_medium(filter):
    p = MyOutpusPass()
    p.filter = filter

    # When the source is smaller, it always uses four samples
    p.set_scale_factor(0.9)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 4
    assert "texCoordEven" in lines[0]

    # Does not really matter how small
    p.set_scale_factor(0.01)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 4
    assert "texCoordEven" in lines[0]

    # When the source is larger, the kernel needs a larger support.
    p.set_scale_factor(1.1)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 9
    assert "texCoordNear" in lines[0]

    # We keep this support until halfway
    p.set_scale_factor(1.5)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 9
    assert "texCoordNear" in lines[0]

    # And then the kernel size is upped
    p.set_scale_factor(1.6)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordEven" in lines[0]

    # Until its a whole number again
    p.set_scale_factor(2.0)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordEven" in lines[0]

    # And then it bumps again
    p.set_scale_factor(2.1)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(2.5)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(2.6)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 36
    assert "texCoordEven" in lines[0]

    # But here is a fun one!
    # The scale is a round number, and because it's uneven, every
    # sample is at the center of a pixel in the source, so
    # an uneven kernel is sufficient, and smaller!
    p.set_scale_factor(3.0)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]


def test_outpass_filter_bspline():
    _test_outpass_filter_cubic("bspline")


def test_outpass_filter_mitchell():
    _test_outpass_filter_cubic("mitchell")


def test_outpass_filter_catmull():
    _test_outpass_filter_cubic("catmull")


def _test_outpass_filter_cubic(filter):
    p = MyOutpusPass()
    p.filter = filter

    # When the source is smaller, it always uses 16 samples
    p.set_scale_factor(0.9)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordEven" in lines[0]

    # Does not really matter how small
    p.set_scale_factor(0.01)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordEven" in lines[0]

    # When the source is larger, the kernel needs a larger support.
    p.set_scale_factor(1.1)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # We keep this support until halfway
    p.set_scale_factor(1.25)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # And then the kernel size is upped
    p.set_scale_factor(1.26)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 36
    assert "texCoordEven" in lines[0]

    # Until its a whole number again
    p.set_scale_factor(1.5)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 36
    assert "texCoordEven" in lines[0]

    # And then it bumps again
    p.set_scale_factor(1.6)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 49 - 4
    assert "texCoordNear" in lines[0]

    # Now we bump to a 7x7 kernel, at which point the corners are dropped
    p.set_scale_factor(1.75)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 49 - 4
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(1.76)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 64 - 4
    assert "texCoordEven" in lines[0]

    # We've reached factor 2
    p.set_scale_factor(2.0)

    _wgsl, lines = p.resolve_wgsl()
    if True:  # filter in ["mitchell", "bspline", "catmull"]:
        # assert len(lines) == 64- 4
        assert len(lines) in (12, 16)  # opt!
        assert "texCoordOrig" in lines[0]
    else:
        assert len(lines) == 64 - 4
        assert "texCoordEven" in lines[0]

    # Skip some beats
    p.set_scale_factor(2.9)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 12 * 12 - 4
    assert "texCoordEven" in lines[0]

    # And then the interesting one
    p.set_scale_factor(3.0)

    _wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 11 * 11 - 4
    assert "texCoordNear" in lines[0]


def test_extra_kernel_support():
    # Bit of a pseudo-test; the extraKernelSupport is only intended for testing
    # but we want to make sure it does the right thing.

    p = MyOutpusPass()

    # The wgsl pops corners for large corners as an optimization, since their
    # contibution is very small. We turn that feature off in this test to make the math simpler.
    p._set_template_var(optCorners=False)

    for filter in ["tent", "disk", "mitchell", "bspline", "catmull"]:
        for scale_factor in [0.9, 1.1, 1.4, 1.8]:
            p.filter = filter
            p.set_scale_factor(scale_factor)

            p._set_template_var(extraKernelSupport=0)
            _wgsl, lines = p.resolve_wgsl()
            kernel_width = int(len(lines) ** 0.5)

            p._set_template_var(extraKernelSupport=-0.5)
            _wgsl, lines = p.resolve_wgsl()
            assert len(lines) == (kernel_width - 1) * (kernel_width - 1)

            p._set_template_var(extraKernelSupport=0.5)
            _wgsl, lines = p.resolve_wgsl()
            assert len(lines) == (kernel_width + 1) * (kernel_width + 1)

            p._set_template_var(extraKernelSupport=1)
            _wgsl, lines = p.resolve_wgsl()
            assert len(lines) == (kernel_width + 2) * (kernel_width + 2)


# %% Testing produced images


class RunnableOutputPass(MyOutpusPass):
    def __init__(self):
        super().__init__()
        self.t1 = None
        self.t2 = None

    def set_image(self, image):
        assert image.dtype == IM_DTYPE, f"Image must be {IM_DTYPE}"
        assert image.shape[2] == 4, "Image must be RGBA"

        self.t1 = self._device.create_texture(
            size=(image.shape[1], image.shape[0], 1),
            format=TEX_FORMAT,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._write_texture(self.t1, image)

    def get_result(self, scale_factor, **template_vars):
        self._set_template_var(**template_vars)
        w, h = self.t1.size[:2]
        w, h = int(w / scale_factor), int(h / scale_factor)

        t2 = self._device.create_texture(
            size=(w, h, 1),
            format=TEX_FORMAT,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        command_encoder = self._device.create_command_encoder()

        self.render(
            command_encoder,
            self.t1.create_view(usage=wgpu.TextureUsage.TEXTURE_BINDING),
            None,
            t2.create_view(usage=wgpu.TextureUsage.RENDER_ATTACHMENT),
        )
        self._device.queue.submit([command_encoder.finish()])

        return self._read_texture(t2)

    def _write_texture(self, texture, image):
        h, w = image.shape[:2]
        self._device.queue.write_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            image,
            {
                "offset": 0,
                "bytes_per_row": w * BYTES_PER_PIXEL,
                "rows_per_image": h,
            },
            (w, h, 1),
        )

    def _read_texture(self, texture):
        w, h = texture.size[:2]
        data = self._device.queue.read_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": w * BYTES_PER_PIXEL,
                "rows_per_image": h,
            },
            (w, h, 1),
        )
        return np.frombuffer(data, IM_DTYPE).reshape(h, w, 4)


# Define pixel format in one place.
# If we want to do proper error measurements, we need a float format.
# But float32 is not blendable (unless using feature float32-blendable)
BYTES_PER_PIXEL = 4 * 2
IM_DTYPE = np.float16
TEX_FORMAT = wgpu.TextureFormat.rgba16float

# Create reference image
ref_image = np.zeros((512, 512, 4), IM_DTYPE)
ref_image[:, :, :3] = iio.imread("imageio:astronaut.png").astype(float) / 255
ref_image[:, :, 3] = 1.0

# Make it somewhat smaller, to speed up the tests
ref_image = ref_image[0:256, 0:256].copy()


def test_outpass_result_scale_1():
    p = RunnableOutputPass()

    p.set_image(ref_image)

    for filter in ["nearest", "linear", "tent", "bspline", "mitchell", "catmull"]:
        p.filter = filter
        im = p.get_result(1)
        max_err = np.nanmax(np.abs(im - ref_image))
        # On MacOS the error is zero, but on lavapipe and on Windows I found it to be nonzero
        assert max_err < 0.0005, (
            f"result mismatch at scale 1 with filter {filter!r}: err {max_err}"
        )


def test_outpass_result_tent():
    # I'm not sure why, but the tent filter does not a small
    # tolerance. It does not produce slightly different results depending on
    # wheter an even or uneven kernel is used. This happens even when I
    # tried emulating the tent filter with a cardinal cubic spline.
    # Similarly, when upsampling, letting the sampler do the interpolation
    # produces slightly different result (less than 1/255) than taking 4 samples
    # and doing the interpolation in the shader, but we want the performance!
    _test_that_kernel_is_exact_correct_size("tent", 0.4, 0.002)
    _test_that_kernel_is_exact_correct_size("tent", 0.9, 0.002)
    _test_that_kernel_is_exact_correct_size("tent", 1.1, 0.0005)
    _test_that_kernel_is_exact_correct_size("tent", 1.4, 0.0005)
    _test_that_kernel_is_exact_correct_size("tent", 1.8, 0.0005)
    _test_that_kernel_is_exact_correct_size("tent", 2.2, 0.0005)


def test_outpass_result_bspline():
    _test_that_kernel_is_exact_correct_size("bspline", 0.4)
    _test_that_kernel_is_exact_correct_size("bspline", 0.9)
    _test_that_kernel_is_exact_correct_size("bspline", 1.1)
    _test_that_kernel_is_exact_correct_size("bspline", 1.4)
    _test_that_kernel_is_exact_correct_size("bspline", 1.8, 0.001)
    _test_that_kernel_is_exact_correct_size("bspline", 2.2, 0.001)


def test_outpass_result_mitchell():
    _test_that_kernel_is_exact_correct_size("mitchell", 0.4)
    _test_that_kernel_is_exact_correct_size("mitchell", 0.9)
    _test_that_kernel_is_exact_correct_size("mitchell", 1.1)
    _test_that_kernel_is_exact_correct_size("mitchell", 1.4)
    _test_that_kernel_is_exact_correct_size("mitchell", 1.8, 0.001)
    _test_that_kernel_is_exact_correct_size("mitchell", 2.2, 0.001)


def test_outpass_result_catmull():
    _test_that_kernel_is_exact_correct_size("catmull", 0.4)
    _test_that_kernel_is_exact_correct_size("catmull", 0.9)
    _test_that_kernel_is_exact_correct_size("catmull", 1.1)
    _test_that_kernel_is_exact_correct_size("catmull", 1.4)
    _test_that_kernel_is_exact_correct_size("catmull", 1.8, 0.001)
    _test_that_kernel_is_exact_correct_size("catmull", 2.2, 0.001)


def _test_that_kernel_is_exact_correct_size(filter, scale_factor, tol=0.0019):
    """
    This test 3 things:

    We compare the output image to one that we create with a kernel that is
    2 elements larger in each dimension (but is even/odd just like the original).
    If the image does not match, the kernel is apparently too small.

    We compare the output image to one that we create with a kernel that
    is one element larger, and is thus odd if the orifinal is even (and vice versa).
    If the image does not match, the odd/even logic is off.

    We compare the output image to one that we create with a *smaller* kernel.
    If the image matches, our kernel could be smaller!
    """
    p = RunnableOutputPass()
    p.filter = filter

    p.set_image(ref_image)
    im1 = p.get_result(scale_factor, extraKernelSupport=None)
    im2 = p.get_result(scale_factor, extraKernelSupport=1)
    im3 = p.get_result(scale_factor, extraKernelSupport=0.5)
    im4 = p.get_result(scale_factor, extraKernelSupport=-0.5)

    info = f"filter={filter!r}, scale_factor={scale_factor})"

    max_err = np.nanmax(np.abs(im1 - im2))
    assert max_err < tol, f"kernel is too small for {info}: err {max_err}"

    max_err = np.nanmax(np.abs(im1 - im3))
    assert max_err < tol, f"kernel odd/even inconsistent for {info}: err {max_err}"

    tol2 = 0.0019
    if scale_factor > 1.7 or filter == "bspline":
        # For large scale-factors, and for bspline, for some kernels it would report it can be smaller,
        # but doing this would make the code more complex; we threath all cubic kernels the same.
        # and want the kernel size to be (somewhat) predictable.
        tol2 = 0.0001

    max_err = np.nanmax(np.abs(im1 - im4))
    assert max_err > tol2, f"kernel size could be smaller for {info}: err {max_err}"


def test_outpass_opt_scale2():
    p = RunnableOutputPass()
    p.set_image(ref_image)

    for tol, filter in [
        (0.0019, "tent"),
        (0.0019, "bspline"),
        (0.0019, "mitchell"),
        (0.0019, "catmull"),
    ]:
        p.filter = filter

        im1 = p.get_result(2, optScale2=True)
        im2 = p.get_result(2, optScale2=False)
        max_err = np.nanmax(np.abs(im1 - im2))
        print(f"opt_scale2 for {filter} max_err: {max_err}")

        assert max_err < tol, (
            f"optSF2 produces suboptimal results for {filter!r} ({max_err})"
        )


if __name__ == "__main__":
    test_outpass_scale_is_1()
    test_outpass_filter_nearest()
    test_outpass_filter_linear()
    test_outpass_filter_disk()
    test_outpass_filter_tent()
    test_outpass_filter_bspline()
    test_outpass_filter_mitchell()
    test_outpass_filter_catmull()

    test_extra_kernel_support()

    test_outpass_result_scale_1()
    test_outpass_result_tent()
    test_outpass_result_bspline()
    test_outpass_result_mitchell()
    test_outpass_result_catmull()

    test_outpass_opt_scale2()
