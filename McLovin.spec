# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app\\main.py'],
    pathex=['C:\\Users\\mohamedamine.laaraj\\projet_pfe\\multi-objective-optimization-of-neural-networks'],
    binaries=[],
    datas=[
        ('models/search_space.py', 'models'),
        ('data/tfidf_vectorizers.pkl', 'data'),
        ('data/label_encoders.pkl', 'data'),
        ('results/best_params.json', 'results'),
        ('results/best_model.pth', 'results'),
    ],
    hiddenimports=['models.search_space'],
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
    a.binaries,
    a.datas,
    [],
    name='McLovin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)