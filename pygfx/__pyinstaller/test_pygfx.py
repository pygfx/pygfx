def test_pyi_pygfx(pyi_builder):
    pyi_builder.test_source(
        """
        import pygfx as gfx
        ff = gfx.font_manager.select_font("foo", gfx.font_manager.default_font_props)[0][1]
        print(ff.codepoints)
        assert len(ff.codepoints) > 100
        """
    )
