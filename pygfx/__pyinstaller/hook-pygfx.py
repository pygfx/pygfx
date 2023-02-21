from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


# until https://github.com/rougier/freetype-py/pull/162/files is released
# we ensure the freetype binaries are included here
binaries = collect_dynamic_libs("freetype")

hiddenimports = ["pygfx.data_files"]
datas = collect_data_files("pygfx.data_files")
