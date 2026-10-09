from pathlib import Path
import contextlib
import os
import runpy
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import numpy
from PIL import Image


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    root = project_root()
    projects = ("Proj3-2", "Proj3-5", "Proj3-6")
    print("Digital Image Processing Project 1")
    print("Running Proj3-2, Proj3-5 and Proj3-6...")
    try:
        with (root / "run_log.txt").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                for project in projects:
                    script = root / project / "main.py"
                    if not script.is_file():
                        raise FileNotFoundError(f"Missing script: {script}")
                    print(f"Running {project}...")
                    runpy.run_path(str(script), run_name="__main__")
        print("Complete. Outputs are in each project/output folder.")
        if "--no-open" not in sys.argv:
            os.startfile(root)
        return 0
    except Exception:
        error = traceback.format_exc()
        (root / "run_error.txt").write_text(error, encoding="utf-8")
        print(error)
        if sys.stdin and sys.stdin.isatty():
            input("Processing failed. See run_error.txt. Press Enter to close.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
