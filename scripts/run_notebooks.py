#!/usr/bin/env python
"""Execute every notebook in order (offline) and save the outputs in place.

Used by CI as a smoke test, and before committing so GitHub renders results.
    python scripts/run_notebooks.py            # execute + save outputs
    python scripts/run_notebooks.py --check    # execute only, do not write
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    check_only = "--check" in sys.argv
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        if not check_only:
            nbformat.write(nb, path)
        print(f"OK  {path.name}")


if __name__ == "__main__":
    main()
