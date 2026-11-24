"""The documentation and the examples must keep working: run every example
script, every Python block of the README, the module doctests, and check
that the output dictionary covers every column and FEATURES.md is current."""
import doctest
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

import politeness_py as P

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("0*.py"))


def _env(tmp_path):
    env = dict(os.environ, POLITENESS_EXAMPLES_OUT=str(tmp_path), MPLBACKEND="Agg")
    pkg_dir = str(Path(P.__file__).resolve().parents[1])   # installed or src/
    env["PYTHONPATH"] = os.pathsep.join([pkg_dir, str(ROOT / "examples"), env.get("PYTHONPATH", "")])
    return env


@pytest.mark.skipif(not EXAMPLES, reason="examples/ not available")
@pytest.mark.parametrize("script", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_script_runs(script, tmp_path):
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=_env(tmp_path),
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "Traceback" not in r.stderr


def test_readme_python_blocks_run(tmp_path, monkeypatch):
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README not available")
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(encoding="utf-8"), flags=re.S)
    assert blocks
    monkeypatch.chdir(tmp_path)
    ns = {}   # the README is read top to bottom: later blocks use earlier names
    for code in blocks:
        exec(compile(code, "README.md", "exec"), ns)


def test_module_doctests():
    import politeness_py.features as F

    res = doctest.testmod(F, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert res.failed == 0


def test_feature_dictionary_covers_every_column():
    d = P.feature_dictionary()
    cols = set(P.politeness(["x"], parser="none").columns) | set(P.politeness(["x"], parser="spacy").columns)
    assert set(d.index) == cols
    assert d.notna().all().all() and (d["description"].str.len() > 0).all()
    for col, avail in d["availability"].items():
        in_none = col in P.politeness(["x"], parser="none").columns
        in_spacy = col in P.politeness(["x"], parser="spacy").columns
        assert (in_none, in_spacy) == {"both": (True, True), "both (rule differs)": (True, True),
                                       "spacy only": (False, True), "none only": (True, False)}[avail], col
    dnm = P.feature_dictionary("dnm")
    assert list(dnm.index) == list(P.politenessDNM(["x"]).columns)


def test_describe():
    df = P.politeness(["Thanks!", "Please, thanks."], parser="none")
    s = P.describe(df)
    assert list(s.index) == list(df.columns)
    assert s.loc["Gratitude", "share_of_texts"] == 1.0 and s.loc["Please", "max"] == 1


def test_features_md_up_to_date():
    gen = ROOT / "tools" / "generate_features_md.py"
    md = ROOT / "FEATURES.md"
    if not gen.exists() or not md.exists():
        pytest.skip("tools/ not available")
    spec = importlib.util.spec_from_file_location("gen_features", gen)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build() == md.read_text(encoding="utf-8"), "run: python tools/generate_features_md.py"


def test_public_api_complete_and_documented():
    for name in P.__all__:
        obj = getattr(P, name)
        assert obj.__doc__ and obj.__doc__.strip(), f"{name} has no docstring"
    assert P.__version__ and P.__r_package_version__ == "0.9.5"
