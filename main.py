"""
main.py  —  Local development launcher.

Spawns app.py (which already includes the merged vision engine)
as a subprocess, then opens the browser automatically.

Usage:
    python main.py

Do NOT use this in production — on Render/Railway use:
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
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
        # Resolve the absolute path to app.py regardless of the cwd
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

        print("\n" + "=" * 52)
        print("  🚀  Starting AI Proctored Exam Portal")
        print("=" * 52)
        print("  app.py — Flask + Vision Engine (merged)")
        print("  CANDIDATE: 240110012345 / Pass@1234")
        print("  TEACHER:   teacher1 / Teacher@123")
        print("=" * 52)

        # Launch app.py with the same Python interpreter that is running main.py
        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        def open_browser() -> None:
            """Wait for the server to start then open the default browser."""
            time.sleep(2.5)
            print("\n  🌐  Opening http://localhost:5000\n")
            webbrowser.open("http://localhost:5000")

        threading.Thread(target=open_browser, daemon=True).start()

        # Block until app.py exits (e.g. Ctrl-C)
        proc.wait()

    except KeyboardInterrupt:
        print("\n  ⛔  Shutting down…")

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("  ✅  Done.\n")


if __name__ == "__main__":
    main()