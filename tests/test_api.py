def test_login_valid_credentials(client):
    resp = client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["candidate"]["name"] == "Arjun Mehta"


def test_login_invalid_credentials(client):
    resp = client.post("/api/login", json={
        "app_number": "240110012345", "password": "wrong",
    })
    assert resp.status_code == 401


def test_login_missing_fields(client):
    resp = client.post("/api/login", json={"app_number": "", "password": ""})
    assert resp.status_code in (400, 401)


def test_exam_start_with_agreement(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    resp = client.post("/api/exam/start", json={"agreed": True})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "end_time" in data


def test_exam_start_without_agreement(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    resp = client.post("/api/exam/start", json={"agreed": False})
    assert resp.status_code == 400


def test_exam_start_unauthenticated(client):
    resp = client.post("/api/exam/start", json={"agreed": True})
    assert resp.status_code == 401


def test_save_answer(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    client.post("/api/exam/start", json={"agreed": True})
    resp = client.post("/api/exam/answer", json={
        "question_id": "q1", "answer": "A",
    })
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_save_answer_without_exam(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    resp = client.post("/api/exam/answer", json={
        "question_id": "q1", "answer": "A",
    })
    assert resp.status_code == 400


def test_exam_submit(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    client.post("/api/exam/start", json={"agreed": True})
    client.post("/api/exam/answer", json={"question_id": "q1", "answer": "A"})
    resp = client.post("/api/exam/submit", json={
        "answers": {"q2": "B", "q3": "C"},
    })
    assert resp.status_code == 200
    assert resp.get_json()["total_answered"] == 3


def test_exam_submit_twice(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    client.post("/api/exam/start", json={"agreed": True})
    client.post("/api/exam/submit", json={})
    resp = client.post("/api/exam/submit", json={})
    assert resp.status_code == 400


def test_proctor_status(client):
    resp = client.get("/proctor/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status_text" in data


def test_proctor_event_logging(client):
    resp = client.post("/api/proctor/event", json={
        "app_number": "240110012345",
        "event_type": "Tab Switch",
        "confidence": 90,
    })
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_proctor_events_list(client):
    client.post("/api/proctor/event", json={
        "app_number": "240110012345", "event_type": "Tab Switch", "confidence": 80,
    })
    resp = client.get("/api/proctor/events")
    assert resp.status_code == 200
    assert resp.get_json()["total"] >= 1


def test_proctor_events_filtered_by_type(client):
    client.post("/api/proctor/event", json={
        "app_number": "240110012345", "event_type": "Tab Switch", "confidence": 80,
    })
    resp = client.get("/api/proctor/events?type=Tab+Switch")
    assert resp.status_code == 200
    for e in resp.get_json()["events"]:
        assert e["event_type"] == "Tab Switch"


def test_proctor_summary(client):
    resp = client.get("/api/proctor/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "logged_in" in data


def test_teacher_login(client):
    resp = client.post("/api/teacher/login", json={
        "username": "teacher1", "password": "Teacher@123",
    })
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_teacher_login_invalid(client):
    resp = client.post("/api/teacher/login", json={
        "username": "teacher1", "password": "wrong",
    })
    assert resp.status_code == 401


def test_teacher_students(client):
    client.post("/api/login", json={
        "app_number": "240110012345", "password": "Pass@1234",
    })
    resp = client.get("/api/teacher/students")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "students" in data
    assert data["total"] >= 1


def test_exam_info(client):
    resp = client.get("/api/exam/info")
    assert resp.status_code == 200
    assert resp.get_json()["exam"]["title"] == "JEE (Main) 2026"
