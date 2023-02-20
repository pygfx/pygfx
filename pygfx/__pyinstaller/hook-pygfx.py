from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("pygfx", include_datas=["pygfx/pkg_resources/*"])
