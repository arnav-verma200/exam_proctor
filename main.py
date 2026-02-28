"""
main.py — Local development launcher.
Starts the single merged app.py (which includes the vision engine).
Run: python main.py
"""
import subprocess, sys, os, time, webbrowser, threading

def main():
    proc = None
    try:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        print("\n" + "="*52)
        print("  🚀  Starting AI Proctored Exam Portal")
        print("="*52)
        print("  app.py — Flask + Vision Engine (merged)")
        print("  CANDIDATE: 240110012345 / Pass@1234")
        print("  TEACHER:   teacher1 / Teacher@123")
        print("="*52)
        proc = subprocess.Popen([sys.executable, script],
                                cwd=os.path.dirname(os.path.abspath(__file__)))
        def open_browser():
            time.sleep(2.5)
            print("\n  🌐  Opening http://localhost:5000\n")
            webbrowser.open("http://localhost:5000")
        threading.Thread(target=open_browser, daemon=True).start()
        proc.wait()
    except KeyboardInterrupt:
        print("\n  ⛔  Shutting down…")
    finally:
        if proc:
            proc.terminate()
            try: proc.wait(timeout=3)
            except: pass
        print("  ✅  Done.\n")

if __name__ == "__main__":
    main()