def test_pyi_pygfx(pyi_builder):
    pyi_builder.test_source(
        """
        import pygfx
        """
    )
