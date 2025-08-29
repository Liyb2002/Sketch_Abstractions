import re
import random
from build123d import export_stl, export_step
from pathlib import Path
import shutil


def eval_with_range(expr: str, context: dict):
    """
    Evaluate an expression that may contain a range like [0.4, 0.8].
    """
    range_match = re.search(r'\[(.*?)\]', expr)
    if range_match:
        a, b = map(float, range_match.group(1).split(','))
        factor = random.uniform(a, b)
        expr = re.sub(r'\[.*?\]', str(factor), expr)
    return eval(expr, {}, context)


def func_export_stl(obj, path):
    shape = getattr(obj, "part", obj)  # unwrap BuildPart -> Part
    export_stl(shape, path)

def func_export_step(obj, path):
    shape = getattr(obj, "part", obj)  # unwrap BuildPart -> Part
    export_step(shape, path)


def clean_dir(path: Path) -> None:
    """
    Ensure directory exists, then delete all its contents.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)  # always ensure it exists

    for child in path.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child, ignore_errors=True)
        except Exception as e:
            print(f"Warning: failed to remove {child}: {e}")
