"""Each tutorial under examples/ runs, and every number it prints agrees
with the output quoted in its docstring (and in the README) to the last
printed digit."""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
TUTORIALS = ["tutorial_gated_graphene.py", "tutorial_band_fit_model_check.py",
             "tutorial_thermopower_scattering.py"]
# examples/ ships in the sdist since 0.14.0 (MANIFEST.in); skip only where
# it is missing (an installed wheel, or an older sdist)
pytestmark = pytest.mark.skipif(
    not os.path.isdir(EXAMPLES), reason="examples/ is not in this "
    "distribution (run the tests from the repository)")
NUM = re.compile(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?")


def _quoted(path):
    text = open(path).read()
    block = re.split(r"Printed output \(hamop \d+\.\d+\.\d+\):\n\n", text,
                     maxsplit=1)[1]
    return block.split('"""', 1)[0]


def _ulp(token):
    """One unit in the last printed digit of a number token."""
    mant, _, exp = token.lower().partition("e")
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return 10.0 ** (-dec) * (10.0 ** int(exp) if exp else 1.0)


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_prints_the_quoted_numbers(name):
    path = os.path.join(EXAMPLES, name)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, path], capture_output=True,
                         text=True, check=True, env=env).stdout
    want = _quoted(path)
    got_lines, want_lines = out.strip().splitlines(), want.strip().splitlines()
    assert len(got_lines) == len(want_lines)
    for g, w in zip(got_lines, want_lines):
        assert NUM.sub("#", g) == NUM.sub("#", w)       # same text
        gn, wn = NUM.findall(g), NUM.findall(w)
        for a, b in zip(gn, wn):
            assert abs(float(a) - float(b)) <= 1.01 * _ulp(b), (g, w)


def test_readme_quotes_the_tutorial_output():
    readme = open(os.path.join(os.path.dirname(HERE), "README.md")).read()
    for name in TUTORIALS:
        assert _quoted(os.path.join(EXAMPLES, name)).strip() in readme
