"""
test_no_ground_truth.py
Structural guarantee: the estimator, controller and sensing code never touch ground truth.
"""
import ast
import pathlib
from tests.conftest import ROOT
 
FORBIDDEN = {"mujoco", "ur3pick.evaluation", "ur3pick.sim"}
CHECKED = ["sensing", "estimation", "control", "modelling"]
 
 
def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
 
 
def test_no_forbidden_imports():
    bad = []
    for pkg in CHECKED:
        for f in (ROOT / "ur3pick" / pkg).rglob("*.py"):
            for name in _imports(f):
                if any(name == b or name.startswith(b + ".") for b in FORBIDDEN):
                    bad.append(f"{f.relative_to(ROOT)} imports {name}")
    assert not bad, "ground-truth wall broken:\n" + "\n".join(bad)