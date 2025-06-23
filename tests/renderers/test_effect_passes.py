
from pygfx.renderers.wgpu.engine.effectpasses import OutputPass, apply_templating


class MyOutpusPass(OutputPass):

    def set_scale_factor(self, factor):
        self._set_template_var(scaleFactor=float(factor))

    def resolve_wgsl(self):
        wgsl = apply_templating(self.wgsl, **self._template_vars)

        lines = []
        for line in wgsl.splitlines():
            if "textureSample" in line or "textureLoad" in line:
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

    p.set_scale_factor(2) # downsampling, source is larger

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

    p.set_scale_factor(2) # downsampling, source is larger

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

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 4
    assert "texCoordLeft" in lines[0]

    # Does not really matter how small
    p.set_scale_factor(0.01)

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
    assert len(lines) == 49
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(1.75)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 49
    assert "texCoordNear" in lines[0]

    # Familiar
    p.set_scale_factor(1.76)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 64
    assert "texCoordLeft" in lines[0]

    # We've reached factor 2
    p.set_scale_factor(2.0)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 64
    assert "texCoordLeft" in lines[0]

    # Skip some beats
    p.set_scale_factor(2.9)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 12*12
    assert "texCoordLeft" in lines[0]


    # And then the interesting one
    p.set_scale_factor(3.0)

    wgsl, lines = p.resolve_wgsl()
    assert len(lines) == 11*11
    assert "texCoordNear" in lines[0]


if __name__ == "__main__":
    test_outpass_scale_is_1()
    test_outpass_filter_nearest()
    test_outpass_filter_linear()
    test_outpass_filter_disk()
    test_outpass_filter_pyramid()
    test_outpass_filter_bspline()
    test_outpass_filter_cubic()

