"""
app.py  —  Legacy entry point (delegates to factory).

Kept for backward compatibility.  Prefer:
    gunicorn 'backend:create_app()' ...
or:
    python main.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

from backend import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*52}")
    print(f"  Open -> http://localhost:{port}")
    print(f"{'='*52}")
    print("  CANDIDATE: 240110012345 / Pass@1234")
    print("  TEACHER:   teacher1 / Teacher@123")
    print(f"{'='*52}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
