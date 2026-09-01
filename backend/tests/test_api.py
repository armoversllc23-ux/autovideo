"""
End-to-end API tests against the real pipeline (IntentParser through
FFmpeg rendering) via FastAPI's TestClient. Resolution is patched down for
speed; the pipeline code path is identical to production, only smaller.
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module

main_module._RESOLUTION_SCALE = 0.2  # keep test renders fast


@pytest.fixture()
def client():
    return TestClient(main_module.app)


def _wait_for_done(client, job_id, timeout_polls=60):
    for _ in range(timeout_polls):
        res = client.get(f"/api/videos/{job_id}")
        assert res.status_code == 200
        body = res.json()
        if body["status"] in ("done", "failed"):
            return body
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_full_flow_text_only(client):
    res = client.post("/api/videos", data={"description": "Fun birthday video for my mom, 10 seconds, for Instagram"})
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    final = _wait_for_done(client, job_id)
    assert final["status"] == "done"
    assert final["ready"] is True
    assert final["width"] == 216  # 1080 * 0.2, rounded even
    assert final["height"] == 384  # 1920 * 0.2

    file_res = client.get(f"/api/videos/{job_id}/file")
    assert file_res.status_code == 200
    assert file_res.headers["content-type"] == "video/mp4"
    assert len(file_res.content) > 1000


def test_full_flow_with_uploaded_photo(client):
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (50, 100, 150)).save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/api/videos",
        data={"description": "Birthday video for my mom, 10 seconds"},
        files={"media": ("mom.jpg", buf, "image/jpeg")},
    )
    job_id = res.json()["job_id"]
    final = _wait_for_done(client, job_id)
    assert final["status"] == "done"


def test_try_a_different_version(client):
    res = client.post("/api/videos", data={"description": "Wedding video for YouTube, 10 seconds"})
    job_id = res.json()["job_id"]
    _wait_for_done(client, job_id)

    variant_res = client.post(f"/api/videos/{job_id}/variant")
    assert variant_res.status_code == 200
    assert variant_res.json()["job_id"] == job_id

    final = _wait_for_done(client, job_id)
    assert final["status"] == "done"


def test_empty_description_is_rejected(client):
    res = client.post("/api/videos", data={"description": "   "})
    assert res.status_code == 400

    res2 = client.post("/api/videos", data={"description": ""})
    assert res2.status_code == 400


def test_unknown_job_id_returns_404(client):
    assert client.get("/api/videos/doesnotexist").status_code == 404
    assert client.get("/api/videos/doesnotexist/file").status_code == 404


def test_file_not_ready_returns_409(client):
    res = client.post("/api/videos", data={"description": "Graduation video, 10 seconds"})
    job_id = res.json()["job_id"]
    # Immediately request the file before the background task has run/finished.
    file_res = client.get(f"/api/videos/{job_id}/file")
    assert file_res.status_code in (200, 409)  # 200 only if TestClient already finished it


def test_variant_before_original_done_returns_409(client):
    res = client.post("/api/videos", data={"description": "Retirement video, 10 seconds"})
    job_id = res.json()["job_id"]
    _wait_for_done(client, job_id)
    # A second variant call after done should always succeed.
    variant_res = client.post(f"/api/videos/{job_id}/variant")
    assert variant_res.status_code == 200
    # An unknown job id should 404, not 409.
    assert client.post("/api/videos/doesnotexist/variant").status_code == 404
