from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    for project in ("Proj3-2", "Proj3-5", "Proj3-6"):
        print(f"Running {project}...")
        subprocess.run([sys.executable, str(ROOT / project / "main.py")], check=True)
    print("All projects completed.")


if __name__ == "__main__":
    main()

