from PyInstaller.utils.hooks import collect_data_files


hiddenimports = ["pygfx.data_files"]
datas = []
datas += collect_data_files("pygfx.data_files")
datas += collect_data_files("pygfx.renderers.wgpu.wgsl")
