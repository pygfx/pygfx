def test_pyi_fonts(pyi_builder):
    pyi_builder.test_source(
        """
        import pygfx as gfx
        ff = gfx.font_manager.select_font("foo", gfx.font_manager.default_font_props)[0][1]
        print(ff.codepoints)
        assert len(ff.codepoints) > 100
        """
    )


def test_pyi_shaders(pyi_builder):
    pyi_builder.test_source(
        """
        from pygfx.renderers.wgpu import load_wgsl
        wgsl = load_wgsl("line.wgsl")
        assert "fn vs_main(" in wgsl
        """
    )
