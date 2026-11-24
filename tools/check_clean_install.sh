#!/usr/bin/env bash
# Installs the built distribution into a fresh virtualenv (no pip cache) and
# runs `pip check` + the test suite against the installed package.
#
#   python -m build
#   tools/check_clean_install.sh <python, e.g. 3.12 or /path/to/python3.14> <wheel|sdist|minimal> [workdir]
#
# Needs `uv` (to create the venv) and, for PyICU, the ICU dev headers
# (apt install libicu-dev pkg-config).
set -o pipefail
PYV=$1; MODE=${2:-wheel}
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${3:-${TMPDIR:-/tmp}/politeness_py_clean_install}
V=$WORK/env_$(basename "$PYV")_$MODE
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$WORK"; rm -rf "$V"
uv venv -q --seed --python "$PYV" "$V" || exit 1
P=$V/bin/python
echo "== $PYV $MODE: $($P --version)"
WHL=$(ls "$REPO"/dist/*.whl); SD=$(ls "$REPO"/dist/*.tar.gz)
case $MODE in
  wheel)   $P -m pip install -q "$WHL[test]" || exit 2 ;;
  sdist)   $P -m pip install -q "$SD[test]" || exit 2 ;;
  minimal) $P -m pip install -q "$WHL" || exit 2 ;;
esac
echo "-- pip check:"; $P -m pip check; echo "pip check exit $?"
if [ "$MODE" = minimal ]; then
  cd /tmp && $P - <<'PY'
import politeness_py as P
df = P.politeness(["Hi! Could you please help? Thanks."], parser="none")
print("parser none OK", df.shape)
for call in (lambda: P.politeness(["a"], parser="spacy"),
             lambda: P.featurePlot(P.politeness(["a b", "c d"], parser="none"), split=[0, 1])):
    try:
        call()
    except ImportError as e:
        print("friendly error:", e)
PY
  exit 0
fi
$P -m spacy download en_core_web_sm >/dev/null 2>&1 || { echo "model download failed"; exit 3; }
cd "$REPO" && $P -c "import politeness_py; print('testing installed package at', politeness_py.__file__)"
$P -m pytest -q -p no:cacheprovider tests 2>&1 | tail -3
