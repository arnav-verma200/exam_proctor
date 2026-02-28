"""
main.py — Local development launcher
Starts app.py and opens the browser automatically.

All AI proctoring now runs in the browser via face-api.js,
so proctor.py is no longer needed.

Usage:
    python main.py
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    return subprocess.Popen(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

def wait_then_open_browser(url, delay=2.0):
    time.sleep(delay)
    print(f"\n  🌐  Opening browser → {url}\n")
    webbrowser.open(url)

def main():
    app_process = None
    try:
        print("\n" + "=" * 52)
        print("  🚀  Starting AI Proctored Exam Portal")
        print("=" * 52)
        print("  Starting app.py (port 5000) …")
        app_process = run_script("app.py")
        print("=" * 52)
        print("  CANDIDATE LOGINS")
        print("  240110012345 / Pass@1234  → Arjun Mehta")
        print("  240110056789 / Pass@5678  → Priya Sharma")
        print("  240110099001 / Pass@9900  → Rahul Singh")
        print("  TEACHER LOGINS")
        print("  teacher1 / Teacher@123")
        print("  admin    / Admin@2026")
        print("=" * 52)
        print("  ℹ  AI proctoring runs in the browser (face-api.js)")
        print("  ℹ  No proctor.py needed anymore")
        print("=" * 52)

        threading.Thread(
            target=wait_then_open_browser,
            args=("http://localhost:5000",),
            daemon=True
        ).start()

        app_process.wait()

    except KeyboardInterrupt:
        print("\n  ⛔  Shutting down…")
    finally:
        if app_process:
            app_process.terminate()
            try: app_process.wait(timeout=3)
            except: pass
        print("  ✅  Done.\n")

if __name__ == "__main__":
    main()