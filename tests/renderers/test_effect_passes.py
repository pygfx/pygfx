from pygfx.renderers.wgpu.engine.effectpasses import OutputPass, apply_templating

import pygfx as gfx
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
    p.filter = "cubic"
    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]

    # Also with other filter
    p.filter = "pyramid"
    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]

    # But can force nearest
    p.filter = "nearest"
    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordNear" in lines[0]


def test_outpass_filter_nearest():
    p = MyOutpusPass()
    p.filter = "nearest"

    p.set_scale_factor(0.5)  # upsampling, source is smaller

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordNear" in lines[0]

    p.set_scale_factor(2)  # downsampling, source is larger

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordNear" in lines[0]


def test_outpass_filter_linear():
    p = MyOutpusPass()
    p.filter = "linear"

    p.set_scale_factor(0.5)  # upsampling, source is smaller

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]

    p.set_scale_factor(2)  # downsampling, source is larger

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 1
    assert "texCoordOri" in lines[0]


def test_outpass_filter_disk():
    _test_outpass_filter_medium("disk")


def test_outpass_filter_pyramid():
    _test_outpass_filter_medium("pyramid")


def _test_outpass_filter_medium(filter):
    p = MyOutpusPass()
    p.filter = filter

    # When the source is smaller, it always uses four samples
    p.set_scale_factor(0.9)

    if filter == "pyramid":
        wgsl, lines = p.resolve_wgsl()
        assert len(lines) == 1
        assert "texCoordOrig" in lines[0]
    else:
        wgsl, lines = p.resolve_wgsl()
        assert len(lines) == 4
        assert "texCoordLeft" in lines[0]

    # Does not really matter how small
    p.set_scale_factor(0.01)

    if filter == "pyramid":
        wgsl, lines = p.resolve_wgsl()
        assert len(lines) == 1
        assert "texCoordOrig" in lines[0]
    else:
        wgsl, lines = p.resolve_wgsl()
        assert len(lines) == 4
        assert "texCoordLeft" in lines[0]

    # When the source is larger, the kernel needs a larger support.
    p.set_scale_factor(1.1)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 9
    assert "texCoordNear" in lines[0]

    # We keep this support until halfway
    p.set_scale_factor(1.5)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 9
    assert "texCoordNear" in lines[0]

    # And then the kernel size is upped
    p.set_scale_factor(1.6)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordLeft" in lines[0]

    # Until its a whole number again
    p.set_scale_factor(2.0)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordLeft" in lines[0]

    # And then it bumps again
    p.set_scale_factor(2.1)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(2.5)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(2.6)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 36
    assert "texCoordLeft" in lines[0]

    # But here is a fun one!
    # The scale is a round number, and because it's uneven, every
    # sample is at the center of a pixel in the source, so
    # an uneven kernel is sufficient, and smaller!
    p.set_scale_factor(3.0)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]


def test_outpass_filter_bspline():
    _test_outpass_filter_cubic("cubic")


def test_outpass_filter_cubic():
    _test_outpass_filter_cubic("bspline")


def _test_outpass_filter_cubic(filter):
    p = MyOutpusPass()
    p.filter = filter

    # When the source is smaller, it always uses 16 samples
    p.set_scale_factor(0.9)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordLeft" in lines[0]

    # Does not really matter how small
    p.set_scale_factor(0.01)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 16
    assert "texCoordLeft" in lines[0]

    # When the source is larger, the kernel needs a larger support.
    p.set_scale_factor(1.1)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # We keep this support until halfway
    p.set_scale_factor(1.25)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 25
    assert "texCoordNear" in lines[0]

    # And then the kernel size is upped
    p.set_scale_factor(1.26)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 36
    assert "texCoordLeft" in lines[0]

    # Until its a whole number again
    p.set_scale_factor(1.5)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 36
    assert "texCoordLeft" in lines[0]

    # And then it bumps again
    p.set_scale_factor(1.6)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 49 - 4
    assert "texCoordNear" in lines[0]

    # Now we bump to a 7x7 kernel, at which point the corners are dropped
    p.set_scale_factor(1.75)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 49 - 4
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(1.76)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 64 - 4
    assert "texCoordLeft" in lines[0]

    # We've reached factor 2
    p.set_scale_factor(2.0)

    wgsl, lines = p.resolve_wgsl()
    if True:  # filter in ["cubic", "bspline"]:
        # assert len(lines) == 64- 4
        assert len(lines) == 12  # opt!
        assert "texCoordOrig" in lines[0]
    else:
        assert len(lines) == 64 - 4
        assert "texCoordLeft" in lines[0]

    # Skip some beats
    p.set_scale_factor(2.9)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 12 * 12 - 4
    assert "texCoordLeft" in lines[0]

    # And then the interesting one
    p.set_scale_factor(3.0)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 11 * 11 - 4
    assert "texCoordNear" in lines[0]


def test_extra_kernel_support():
    # Bit of a pseudo-test; the extraKernelSupport is only intended for testing
    # but we want to make sure it does the right thing.

    p = MyOutpusPass()

    # The wgsl pops corners for large corners as an optimization, since their
    # contibution is very small. We turn that feature off in this test to make the math simpler.
    p._set_template_var(optCorners=False)

    for filter in ["pyramid", "disk", "cubic", "bspline"]:
        for scale_factor in [0.9, 1.1, 1.4, 1.8]:
            p.filter = filter
            p.set_scale_factor(scale_factor)

            p._set_template_var(extraKernelSupport=0)
            wgsl, lines = p.resolve_wgsl()
            kernel_width = int(len(lines) ** 0.5)

            p._set_template_var(extraKernelSupport=-0.5)
            wgsl, lines = p.resolve_wgsl()
            assert len(lines) == (kernel_width - 1) * (kernel_width - 1)

            p._set_template_var(extraKernelSupport=0.5)
            wgsl, lines = p.resolve_wgsl()
            assert len(lines) == (kernel_width + 1) * (kernel_width + 1)

            p._set_template_var(extraKernelSupport=1)
            wgsl, lines = p.resolve_wgsl()
            assert len(lines) == (kernel_width + 2) * (kernel_width + 2)


# %% Testing produced images


class TestableOutputPass(MyOutpusPass):
    def __init__(self):
        super().__init__()
        self.t1 = None
        self.t2 = None

    def set_image(self, image):
        assert image.dtype == np.float16, "Image must be float16"
        assert image.shape[2] == 4, "Image must be RGBA"

        self.t1 = self._device.create_texture(
            size=(image.shape[1], image.shape[0], 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._write_texture(self.t1, image)

    def get_result(self, scale_factor, **template_vars):
        self._set_template_var(**template_vars)
        w, h = self.t1.size[:2]
        w, h = int(w / scale_factor), int(h / scale_factor)

        t2 = self._device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba16float,
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
                "bytes_per_row": w * 4 * 2,
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
                "bytes_per_row": w * 4 * 2,
                "rows_per_image": h,
            },
            (w, h, 1),
        )
        return np.frombuffer(data, np.float16).reshape(h, w, 4)


# Create reference image
ref_image = np.zeros((512, 512, 4), np.float16)
ref_image[:, :, :3] = iio.imread("imageio:astronaut.png").astype(float) / 255
ref_image[:, :, 3] = 1.0

# Make it somewhat smaller, to speed up the tests
ref_image = ref_image[0:256, 0:256].copy()


def test_outpass_result_scale_1():
    p = TestableOutputPass()

    p.set_image(ref_image)

    for filter in ["nearest", "linear", "pyramid", "cubic", "bspline"]:
        p.filter = filter
        im = p.get_result(1)
        assert np.all(im == ref_image)


def test_outpass_result_pyramid():
    # I'm not sure why, but the pyramid filter does not a small
    # tolerance. It does not produce slightly different results depending on
    # wheter an even or uneven kernel is used. This happens even when I
    # tried emulating the pyramid filter with a cardinal cubic spline.
    # Similarly, when upsampling, letting the sampler do the interpolation
    # produces slightly different result (less than 1/255) than taking 4 samples
    # and doing the interpolation in the shader, but we want the performance!
    _test_that_kernel_is_exact_correct_size("pyramid", 0.4, 0.002)
    _test_that_kernel_is_exact_correct_size("pyramid", 0.9, 0.002)
    _test_that_kernel_is_exact_correct_size("pyramid", 1.1, 0.0005)
    _test_that_kernel_is_exact_correct_size("pyramid", 1.4, 0.0005)
    _test_that_kernel_is_exact_correct_size("pyramid", 1.8, 0.0005)
    _test_that_kernel_is_exact_correct_size("pyramid", 2.2, 0.0005)


def test_outpass_result_mitchell():
    _test_that_kernel_is_exact_correct_size("cubic", 0.4)
    _test_that_kernel_is_exact_correct_size("cubic", 0.9)
    _test_that_kernel_is_exact_correct_size("cubic", 1.1)
    _test_that_kernel_is_exact_correct_size("cubic", 1.4)
    _test_that_kernel_is_exact_correct_size("cubic", 1.8, 0.001)
    _test_that_kernel_is_exact_correct_size("cubic", 2.2, 0.001)


def test_outpass_result_bspline():
    _test_that_kernel_is_exact_correct_size("bspline", 0.4)
    _test_that_kernel_is_exact_correct_size("bspline", 0.9)
    _test_that_kernel_is_exact_correct_size("bspline", 1.1)
    _test_that_kernel_is_exact_correct_size("bspline", 1.4)
    _test_that_kernel_is_exact_correct_size("bspline", 1.8, 0.001)
    _test_that_kernel_is_exact_correct_size("bspline", 2.2, 0.001)


def _test_that_kernel_is_exact_correct_size(filter, scale_factor, tol=0.001):
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
    p = TestableOutputPass()
    p.filter = filter

    p.set_image(ref_image)
    im1 = p.get_result(scale_factor, extraKernelSupport=None)
    im2 = p.get_result(scale_factor, extraKernelSupport=1)
    im3 = p.get_result(scale_factor, extraKernelSupport=0.5)
    im4 = p.get_result(scale_factor, extraKernelSupport=-0.5)

    info = f"({filter!r} {scale_factor})"
    assert allclose(im1, im2, tol), f"kernel is apparently too small {info}"
    assert allclose(im1, im3, tol), f"kernel odd/even inconsistent result {info}"

    tol2 = 0.001
    if scale_factor > 1.7 or filter == "bspline":
        # For large scale-factors, and for bspline, for some kernels it would report it can be smaller,
        # but doing this would make the code more complex; we threath all cubic kernels the same.
        # and want the kernel size to be (somewhat) predictable.
        tol2 = 0.0001
    assert not allclose(im1, im4, tol2), f"kernel size could be smaller {info}"


def test_outpass_opt_scale2():
    p = TestableOutputPass()
    p.set_image(ref_image)

    for tol, filter in [(0.0019, "cubic"), (0.0025, "bspline")]:
        p.filter = filter

        im1 = p.get_result(2, optScale2=True)
        im2 = p.get_result(2, optScale2=False)
        maxerr = np.abs(im1 - im2).max()
        print(f"opt_scale2 maxerr for {filter}: {maxerr}")
        assert allclose(im1, im2, tol), f"optSF2 produces suboptimal results ({maxerr})"


def allclose(a, b, atol=0.0019):
    # With a max error of 0.5/256, any uint8 image should be equal.
    # Ok, there's factors like srgb vs linear etc. but we need *some* target tolerance :)
    if atol is None:
        atol = 0.0019
    return np.allclose(a, b, 0, atol)


##


if __name__ == "__main__":
    test_outpass_scale_is_1()
    test_outpass_filter_nearest()
    test_outpass_filter_linear()
    test_outpass_filter_disk()
    test_outpass_filter_pyramid()
    test_outpass_filter_bspline()
    test_outpass_filter_cubic()
    test_extra_kernel_support()

    test_outpass_result_scale_1()
    test_outpass_result_pyramid()
    test_outpass_result_mitchell()
    test_outpass_result_bspline()

    test_outpass_opt_scale2()
