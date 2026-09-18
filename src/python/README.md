# Pose Format

[CPU ir MediaPipe Tasks naudojimas](pose_format/estimation/README.md)

[Pozų animavimas FBX modelyje](pose_format/animation/README.md)

## Publishing
```bash
pip install --upgrade build twine pyopenssl cryptography requests-toolbelt
rm -rf dist
python3 -m build
python3 -m twine upload dist/*
```
