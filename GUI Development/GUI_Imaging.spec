from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(SPECPATH).resolve()

datas = [
    (str(project_root / "CP_models"), "CP_models"),
    (str(project_root / "images"), "images"),
    (str(project_root / "Guide.pdf"), "."),
]

for package_name in ("cellpose", "dearpygui", "matplotlib", "nd2"):
    datas += collect_data_files(package_name, include_py_files=False)

hiddenimports = []
for package_name in ("cellpose", "dearpygui", "skimage", "nd2"):
    hiddenimports += collect_submodules(package_name)


a = Analysis(
    ["GUI_Imaging.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GCL_analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GCL_analyzer",
)
