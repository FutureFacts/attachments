# Release Pipeline Status and Troubleshooting Notes (2025‑09‑09)

This note summarizes the current status of the “Release Stable to PyPI” workflow, the changes we made, the failures observed, and the concrete steps an expert can take to fix publishing.

## TL;DR

- The release workflow runs on tag pushes and via manual dispatch, builds fine, but PyPI publishing fails with a trusted‑publisher “invalid-publisher” error.
- There is also a tag/version mismatch: tags `v0.23.1` and `v0.23.2` exist, but `pyproject.toml` still declares `version = "0.23.0"`.
- Two options to fix publishing:
  1) Configure PyPI Trusted Publisher for this repo/workflow/environment.
  2) Add a `PYPI_API_TOKEN` repo secret and let the workflow use token‑based publishing.
- After fixing publishing, bump the package version (e.g., to `0.23.3`), commit, tag `v0.23.3`, and push to trigger a clean release.

## Repo / Environment

- Repository: `MaximeRivest/attachments`
- Current main HEAD: includes version `0.23.0` in `pyproject.toml`, changelog for `0.23.0`, release workflow fixes.
- Tags pushed: `v0.23.0`, `v0.23.1`, `v0.23.2` (note: mismatched with `pyproject.toml` which is `0.23.0`).
- Tests: pre‑commit Black/Ruff clean; pytest (uv) 37 passed on local runs.

## What changed in 0.23.x (relevant highlights)

- Inline OCR in PDF processor (`ocr:true` OCR‑only text, `ocr:auto` conditional; honors `lang:chi_sim`).
- IPYNB loader/presenter/processor wired correctly; empty notebooks yield empty text.
- SVG fallback to `data:image/svg+xml` when rasterizer is unavailable; resizing skips SVGs.
- Pre‑commit pytest runs via `uv run` to respect `.venv`.

## Release Workflow summary

File: `.github/workflows/release-stable.yml`

- Triggers:
  - `on.push.tags: 'v*.*.*'` (glob; no pre‑release suffixes)
  - `workflow_dispatch` with optional `tag` input
- Steps:
  - Checkout, set up Python 3.11
  - Install build deps (build, twine)
  - Generate DSL cheatsheet (`pip install -e . && python scripts/generate_dsl_cheatsheet.py`)
  - Build package (`python -m build`)
  - Verify package with `twine check`
  - Publish to PyPI:
    - If `PYPI_API_TOKEN` repo secret is set → token‑based publish (`user: __token__`, `password: ${{ secrets.PYPI_API_TOKEN }}`)
    - Else → trusted publishing (OIDC)
  - Create GitHub Release with:
    - Name: `Release ${{ steps.meta.outputs.tag }}`
    - Install line pinned to `${{ steps.meta.outputs.version }}` (derived from tag)

- Metadata step computes:
  - `tag`: from `github.ref_name` when running on tag; else from `inputs.tag`; else from `git describe --tags --abbrev=0`
  - `version`: `${tag#v}`

## Failures observed (with `gh` CLI)

- Listing recent runs:
  - `gh run list -R MaximeRivest/attachments -w ".github/workflows/release-stable.yml" -L 5`
- Recent runs `#18`, `#19`, `#20` ended in `completed/failure`.
- For run `#19` / `#18`, the failing step is “Publish to PyPI”. Logs show:
  - `Trusted publishing exchange failure: invalid-publisher` (no matching publisher found).
  - Claims (for debugging):
    - `sub: repo:MaximeRivest/attachments:environment:release`
    - `workflow_ref: MaximeRivest/attachments/.github/workflows/release-stable.yml@refs/tags/v0.23.2` (or branch for #18)
    - `ref: refs/tags/v0.23.2` (or branch)
    - `environment: release`
  - This indicates that PyPI Trusted Publisher is not configured for this repo/workflow/environment, or the configuration does not match the claims.

## Current constraints and fixes already applied

- GitHub Actions expression fix:
  - Replaced invalid `${{ github.ref_name | replace('v', '') }}` with `${{ replace(github.ref_name, 'v', '') }}`.
  - Added `workflow_dispatch` trigger and metadata step for robust tag/version usage.
- Dependency pin fix for CI:
  - Relaxed `pypdfium2` base dependency to `>=4.30.0` (4.30.1 was yanked), CI builds succeed.
- Added token fallback for PyPI publishing:
  - If `PYPI_API_TOKEN` is defined as a repo secret, the workflow uses token‑based publishing; otherwise it attempts trusted publishing (OIDC).

## Outstanding issues

1) PyPI Trusted Publishing is not configured:
   - Error: `invalid-publisher` during OIDC exchange.
   - Needs a Trusted Publisher configured in PyPI for project `attachments`.
   - Expected to match claims:
     - Repo: `MaximeRivest/attachments`
     - Workflow file: `.github/workflows/release-stable.yml`
     - Environment: `release`
     - Ref type: tags (for tag runs) and/or branches (if you also publish from `main`).

2) Version/tag mismatch:
   - `pyproject.toml` → `version = "0.23.0"`
   - Tags pushed → `v0.23.1`, `v0.23.2`
   - The build artifacts are `attachments-0.23.0` while the tag says `v0.23.2`.

## Recommended next steps (expert help requested)

Choose one publishing method:

A) Configure PyPI Trusted Publisher
- In PyPI (project: `attachments`) → Settings → Publishing → Trusted Publishers → Add
  - Owner: GitHub
  - Repository: `MaximeRivest/attachments`
  - Workflow file: `.github/workflows/release-stable.yml`
  - Environment: `release`
  - Accept tag refs (and optionally branch `main`, depending on desired triggers)
- Save and verify.
- Then bump version → tag → push:
  - Update `pyproject.toml` to `version = "0.23.3"`
  - Commit: `chore(release): 0.23.3`
  - Tag: `git tag v0.23.3 -m "release: 0.23.3"`
  - Push: `git push origin main v0.23.3`

B) Use a classic PyPI API token (no PyPI config needed)
- In GitHub repo: Settings → Secrets and variables → Actions → New secret
  - Name: `PYPI_API_TOKEN`
  - Value: Your PyPI token
- Then bump version → tag → push as above to re‑run workflow and publish.

Either path also fixes the version mismatch when you bump `pyproject.toml` and tag the same version.

## Useful `gh` commands

- List release runs:
  ```bash
  gh run list -R MaximeRivest/attachments -w ".github/workflows/release-stable.yml" -L 5
  ```
- Inspect a run and its jobs:
  ```bash
  gh run view -R MaximeRivest/attachments <run-id> --json jobs -q '.jobs[] | [.name, .databaseId, .url] | @tsv'
  gh run view -R MaximeRivest/attachments <run-id> --job <job-id> --log
  ```
- Manually dispatch the release workflow for a tag:
  ```bash
  gh workflow run -R MaximeRivest/attachments ".github/workflows/release-stable.yml" -f tag=v0.23.3
  ```

## Appendix: Current release workflow (key parts)

```yaml
on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      tag:
        description: 'Tag to release (e.g., v0.23.3)'
        required: false

jobs:
  release-stable:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Generate DSL cheatsheet
        run: |
          pip install -e .
          python scripts/generate_dsl_cheatsheet.py

      - name: Build package
        run: python -m build

      - name: Verify package
        run: |
          python -m twine check dist/*
          ls -la dist/

      - name: Compute release metadata
        id: meta
        run: |
          TAG="${{ github.ref_type == 'tag' && github.ref_name || inputs.tag }}"
          if [ -z "$TAG" ]; then TAG="$(git describe --tags --abbrev=0)"; fi
          echo "tag=$TAG" >> "$GITHUB_OUTPUT"
          echo "version=${TAG#v}" >> "$GITHUB_OUTPUT"

      - name: Publish to PyPI (token)
        if: ${{ secrets.PYPI_API_TOKEN != '' }}
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
          print-hash: true
          verbose: true

      - name: Publish to PyPI (trusted)
        if: ${{ secrets.PYPI_API_TOKEN == '' }}
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          print-hash: true
          verbose: true

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ steps.meta.outputs.tag }}
          name: "Release ${{ steps.meta.outputs.tag }}"
          body: |
            ```bash
            pip install attachments==${{ steps.meta.outputs.version }}
            ```
          prerelease: false
          files: dist/*
```

---

If you have preferences for Trusted Publisher vs API token, or if the PyPI project settings are known, we can execute the chosen path and finish the release immediately.

