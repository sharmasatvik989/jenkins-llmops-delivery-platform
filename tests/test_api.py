from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_liveness():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_classification_is_rejected():
    response = client.post("/v1/classify", json={"text": ""})
    assert response.status_code == 422

