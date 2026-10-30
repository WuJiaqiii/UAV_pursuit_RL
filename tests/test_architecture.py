"""Keep dependency direction, optional dependencies and source provenance intact."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

from pursuit_rl.artifacts import provenance


def test_package_dependencies_resolve_without_cycles_or_upward_imports():
    root = Path(__file__).resolve().parents[1] / "pursuit_rl"
    modules = {}
    for path in root.rglob("*.py"):
        parts = list(path.relative_to(root.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    allowed = {
        "config": set(),
        "simulation": set(),
        "control": set(),
        "artifacts": {"config"},
        "learning": {"config", "simulation", "artifacts"},
        "visualization": {"artifacts"},
        "evaluation": {
            "config",
            "simulation",
            "control",
            "artifacts",
            "learning",
            "visualization",
        },
        "training": {"config", "simulation", "artifacts", "learning", "evaluation"},
    }
    graph = {name: set() for name in modules}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        dependencies = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            if isinstance(node, ast.ImportFrom):
                dependencies.append(
                    importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                    if node.level
                    else node.module
                )
            elif isinstance(node, ast.Import):
                dependencies.extend(alias.name for alias in node.names)
        for target in dependencies:
            assert target != "main", f"Package code must not import CLI: {name}"
            if target == "pursuit_rl" or not target.startswith("pursuit_rl."):
                continue
            assert target in modules, f"Unresolved import: {name} -> {target}"
            layer = name.split(".")[1]
            other = target.split(".")[1]
            assert (
                other == layer or other in allowed[layer]
            ), f"Dependency points upward: {name} -> {target}"
            graph[name].add(target)

    visited, active = set(), set()

    def visit(name):
        assert name not in active, f"Circular module dependency: {name}"
        if name in visited:
            return
        active.add(name)
        for target in graph[name]:
            visit(target)
        active.remove(name)
        visited.add(name)

    for name in graph:
        visit(name)


def test_config_simulation_and_baseline_import_without_torch_or_matplotlib():
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'matplotlib'}:
            raise ImportError('Forbidden eager dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from main import build_parser
from pursuit_rl.config import Config
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.evaluation.benchmark import run_benchmark
cfg = Config.load()
env = PursuitBatch(cfg.to_dict(), n=2, seed=17, stage=0)
action = GuidanceController('PN').act(env, env.observe()[0])
assert action.shape == (2, 4)
assert build_parser().parse_args(['train']).command == 'train'
assert 'torch' not in sys.modules and 'matplotlib' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_source_fingerprint_covers_other_packages_defaults_and_entry(
    tmp_path, monkeypatch
):
    root = tmp_path / "pursuit_rl"
    (root / "artifacts").mkdir(parents=True)
    paths = [
        root / "training/trainer.py",
        root / "config/defaults.yaml",
        tmp_path / "main.py",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before", encoding="utf8")
    monkeypatch.setattr(provenance, "__file__", str(root / "artifacts/provenance.py"))
    previous = provenance.source_fingerprint()
    for path in paths:
        path.write_text("after", encoding="utf8")
        current = provenance.source_fingerprint()
        assert current != previous
        previous = current
