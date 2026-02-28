import subprocess
import sys
import os
import time
import webbrowser
import threading

def run_script(script_name):
    """Starts a Python script as a subprocess."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    return subprocess.Popen(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

def wait_then_open_browser(url, delay=2.5):
    """Wait a moment for servers to boot, then open the browser."""
    time.sleep(delay)
    print(f"\n  🌐  Opening browser → {url}\n")
    webbrowser.open(url)

def main():
    proctor_process = None
    app_process = None
    try:
        print("\n" + "=" * 52)
        print("  🚀  Starting AI Proctored Exam Portal")
        print("=" * 52)
        print("  Starting proctor.py  (port 5001) …")
        proctor_process = run_script("proctor.py")

        print("  Starting app.py      (port 5000) …")
        app_process = run_script("app.py")

        print("=" * 52)
        print("  CANDIDATE LOGINS")
        print("  240110012345 / Pass@1234  → Arjun Mehta")
        print("  240110056789 / Pass@5678  → Priya Sharma")
        print("  240110099001 / Pass@9900  → Rahul Singh")
        print("  TEACHER LOGIN")
        print("  teacher1 / Teacher@123")
        print("  admin    / Admin@2026")
        print("=" * 52)

        # Open browser automatically after short delay
        threading.Thread(
            target=wait_then_open_browser,
            args=("http://localhost:5000",),
            daemon=True
        ).start()

        # Wait for both processes to complete
        proctor_process.wait()
        app_process.wait()

    except KeyboardInterrupt:
        print("\n  ⛔  Shutting down…")
    finally:
        if proctor_process:
            proctor_process.terminate()
            try: proctor_process.wait(timeout=3)
            except: pass
        if app_process:
            app_process.terminate()
            try: app_process.wait(timeout=3)
            except: pass
        print("  ✅  Processes terminated.\n")

if __name__ == "__main__":
    main()