"""
main.py  —  Local development launcher.

Spawns the refactored backend (via the app factory) as a subprocess,
then opens the browser automatically.

Usage:
    python main.py

For production on Render/Railway use:
    gunicorn 'backend:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser


def main() -> None:
    proc = None
    try:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

        print("\n" + "=" * 52)
        print("  Starting AI Proctored Exam Portal")
        print("=" * 52)
        print("  backend — refactored blueprint structure")
        print("  CANDIDATE: 240110012345 / Pass@1234")
        print("  TEACHER:   teacher1 / Teacher@123")
        print("=" * 52)

        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        def open_browser() -> None:
            time.sleep(2.5)
            print("\n  Opening http://localhost:5000\n")
            webbrowser.open("http://localhost:5000")

        threading.Thread(target=open_browser, daemon=True).start()

        proc.wait()

    except KeyboardInterrupt:
        print("\n  Shutting down…")

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("  Done.\n")


if __name__ == "__main__":
    main()
