"""Small, real-service local integration; independent from synthetic V2."""
from pathlib import Path
import sys

_deps = Path(__file__).resolve().parents[2] / "runtime/smallstack/deps"
if _deps.is_dir() and str(_deps) not in sys.path:
    sys.path.insert(0, str(_deps))
