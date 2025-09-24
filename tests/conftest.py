import sys
from pathlib import Path

# Ensure the `src` directory is on sys.path for imports like `from src.functions import ...`.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
