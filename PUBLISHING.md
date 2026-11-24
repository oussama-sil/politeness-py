# Putting politeness-py on GitHub and PyPI

This folder (`politeness-py/`) is a self-contained repository: it contains
only the Python package, its tests and docs. None of the R package's code
is included. The R package's data (word lists, dictionaries, datasets,
model coefficients) is redistributed as JSON under its MIT licence, with
attribution in `README.md`, `LICENSE`, `CITATION.cff` and `HOW_IT_WORKS.md`.

## 1. Create the GitHub repository

1. On GitHub: **New repository** → name `politeness-py` → leave it **empty**
   (no README, licence or .gitignore; they already exist here).
2. In a terminal:

```bash
cd ~/projects/politeness-py
git init -b main
git add .
git commit -m "politeness-py 0.1.0: Python port of the R package politeness"
git remote add origin https://github.com/<your-username>/politeness-py.git
git push -u origin main
```

3. Add your repository links to the package metadata (they appear on PyPI)
   and make the README's documentation link absolute (PyPI cannot follow
   relative links):

```bash
U=<your-username>
cat >> pyproject.toml.urls <<EOF
Homepage = "https://github.com/$U/politeness-py"
Documentation = "https://github.com/$U/politeness-py/blob/main/HOW_IT_WORKS.md"
Issues = "https://github.com/$U/politeness-py/issues"
EOF
sed -i "/^\[project.urls\]/r pyproject.toml.urls" pyproject.toml && rm pyproject.toml.urls
sed -i "s#(HOW_IT_WORKS.md)#(https://github.com/$U/politeness-py/blob/main/HOW_IT_WORKS.md)#g; s#(CITATION.cff)#(https://github.com/$U/politeness-py/blob/main/CITATION.cff)#; s#(LICENSE)#(https://github.com/$U/politeness-py/blob/main/LICENSE)#" README.md
sed -i "s#<this repository>#https://github.com/$U/politeness-py#" README.md
git commit -am "Add repository URLs" && git push
```

4. Open the **Actions** tab: the `tests` workflow installs the package on
   Python 3.10–3.13 and runs the test suite. Wait until it is green.

5. Optional but recommended: on the repo page add the description
   *"Python port of the R package politeness (Yeomans, Kantor & Tingley)"*
   and topics `nlp`, `politeness`, `spacy`, `computational-social-science`.

## 2. Publish on PyPI (trusted publishing, no API token)

1. Create an account on <https://pypi.org> and enable two-factor auth.
2. PyPI → *Your account* → **Publishing** → *Add a new pending publisher*:
   * PyPI project name: `politeness-py`
   * Owner: `<your-username>`, Repository: `politeness-py`
   * Workflow name: `publish.yml`, Environment name: `pypi`
3. GitHub → repo **Settings → Environments → New environment** → `pypi`.
4. Make a release: GitHub → **Releases → Draft a new release** → tag
   `v0.1.0` (create on publish) → title `0.1.0` → paste the CHANGELOG entry →
   **Publish release**. The `publish` workflow builds, checks
   (`twine check --strict`) and uploads to PyPI.
5. Check: <https://pypi.org/project/politeness-py/>, then in a fresh
   environment:

```bash
python -m venv /tmp/t && . /tmp/t/bin/activate
pip install "politeness-py[all]" && python -m spacy download en_core_web_sm
python -c "import politeness_py as P; print(P.politeness(['Could you please help? Thanks!'], parser='spacy'))"
```

### Alternative: upload manually from your machine

```bash
pip install build twine
rm -rf dist && python -m build
twine check --strict dist/*
twine upload --repository testpypi dist/*   # optional dry run on test.pypi.org
twine upload dist/*                          # asks for a PyPI API token
```

## 3. Later releases

1. Bump the version in **both** `pyproject.toml` (`version = ...`) and
   `src/politeness_py/__init__.py` (`__version__`), and add a `CHANGELOG.md` entry.
2. `python -m pytest` (and `python -m pytest -m deep` if you regenerated the
   deep fixtures). If you changed a feature description, run
   `python tools/generate_features_md.py`.
3. Optional clean-install check: `python -m build && tools/check_clean_install.sh 3.12 wheel`
   (fresh virtualenv, `pip check`, full test suite against the installed package).
4. Commit, push, and publish a new GitHub release `vX.Y.Z`.

## 4. Acknowledging the original authors

Already done in the package: the README opens with the credit and has an
*Acknowledgements* section (authors, repository, CRAN, licence, funding,
"unofficial port, not affiliated or endorsed"), `LICENSE` keeps the original
copyright line, `CITATION.cff` asks users to cite the original paper, and
`HOW_IT_WORKS.md` and the package docstring credit the R package. As a
courtesy you may also tell the authors about the port, e.g. by opening an
issue at <https://github.com/myeomans/politeness/issues>.

## Repository contents

| path | what |
|---|---|
| `src/politeness_py/` | the package (pure Python) and `data/` (JSON exported from the R package) |
| `tests/` | 229 tests (+78 deep); `fixtures/` holds the R reference outputs (`fixtures/deep/` is generated locally and git-ignored) |
| `tests/r_reference/` | R scripts that regenerate the references from a clone of the original R package (`POLITENESS_R_ROOT=...`) |
| `HOW_IT_WORKS.md` | detailed documentation of the method and of the port |
| `FEATURES.md` | output dictionary: meaning of every column (generated by `tools/generate_features_md.py`) |
| `examples/` | runnable example scripts for every use case |
| `tools/` | FEATURES.md generator, benchmark |
| `.github/workflows/` | `tests.yml` (CI), `publish.yml` (PyPI release) |
