# Releasing

How a version of ggml-compiler gets to PyPI. The workflow `.github/workflows/release.yml` builds
the sdist and the Linux x86-64 wheel, runs `tests/wheel_smoke.py` against the installed wheel, and
publishes both when a `v*` tag is pushed.

## Once

1. On PyPI, add a trusted publisher for the project `ggml-compiler` (for a first release, as a
   "pending publisher" under your account's publishing settings): owner `khanhnd61-vr`,
   repository `ggml-compiler`, workflow `release.yml`, environment `pypi`.
2. In the GitHub repository's settings, create the environment `pypi`. Requiring a reviewer there
   makes every publish wait for an approval.

No API token is stored anywhere: the publish job proves its identity to PyPI with GitHub's OIDC
token.

## Each release

1. Set the version in `ggml_compiler/__init__.py` (`__version__`). It is the only place: the
   package metadata and the runtime's CMake read it from there.
2. Move the release's notes in `CHANGELOG.md` under that version, with the date.
3. Run the full test suite on a machine with a GPU, big checkpoints included:
   `GGML_COMPILER_BIG=1 uv run pytest -q -p no:logging`.
4. Build and test the release artifacts without publishing: run the `release` workflow by hand
   (Actions, then "Run workflow"), or locally:

   ```bash
   SKBUILD_CMAKE_DEFINE="GGML_NATIVE=OFF" uv build -o dist
   uv venv /tmp/rel && VIRTUAL_ENV=/tmp/rel uv pip install dist/*.whl \
       --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
       --index-strategy unsafe-best-match
   cd /tmp && /tmp/rel/bin/python ~/path/to/ggml-compiler/tests/wheel_smoke.py
   uvx twine check dist/*
   ```

5. Commit, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`. The workflow publishes
   once the sdist and wheel jobs pass (and the `pypi` environment's reviewer approves, if set).
6. Check the release: `pip install ggml-compiler==0.1.0` in a clean environment, then run
   `tests/wheel_smoke.py` from outside the checkout.

A version on PyPI cannot be replaced. If a release is broken, yank it on PyPI and release the next
version.
