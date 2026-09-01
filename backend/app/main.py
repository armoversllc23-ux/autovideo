"""
FastAPI API layer — the only thing the frontend talks to.

Deliberately tiny surface area (see ARCHITECTURE.md section 5/6): there is
no endpoint that lets the frontend choose occasion/platform/palette/etc.
Those only ever come out of the pipeline. The four endpoints below are the
whole contract:

  POST /api/videos             — start a job from text (+ optional media)
  GET  /api/videos/{id}        — poll status / get result metadata
  GET  /api/videos/{id}/file   — download/stream the rendered video
  POST /api/videos/{id}/variant — "Try a different version" (same script)
"""
from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .intent_parser import IntentParser
from .media_selector import MediaSelector
from .models import JobStatus, RenderJob
from .renderer import Renderer
from .storyboard_generator import StoryboardGenerator
from .storage import job_store

app = FastAPI(title="AutoVideo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_intent_parser = IntentParser()
_storyboard_generator = StoryboardGenerator()
_media_selector = MediaSelector()
_renderer = Renderer()

# Full resolution in the API path; tests use a smaller scale directly
# against the pipeline modules (see backend/tests) to stay fast.
_RESOLUTION_SCALE = 1.0


# --------------------------------------------------------------------------
# Response schemas (kept separate from the internal RenderJob so the API
# never leaks internal fields like file-system paths).
# --------------------------------------------------------------------------

class CreateVideoResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_message: str
    error: Optional[str] = None
    ready: bool = False
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


def _to_status_response(job: RenderJob) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_message=job.progress_message,
        error=job.error,
        ready=job.status == JobStatus.DONE,
        duration_seconds=job.output_meta.duration_seconds if job.output_meta else None,
        width=job.output_meta.width if job.output_meta else None,
        height=job.output_meta.height if job.output_meta else None,
    )


# --------------------------------------------------------------------------
# The pipeline itself — this is the one function that walks
# IntentParser -> StoryboardGenerator -> MediaSelector -> Renderer,
# narrating friendly progress messages for the frontend's progress screen
# at each step (see brief section 6).
# --------------------------------------------------------------------------

def _run_pipeline(job_id: str, description: str, uploaded_paths: list[str], variant_seed: int = 0) -> None:
    try:
        job_store.update(job_id, status=JobStatus.PARSING, progress_message="Understanding your description...")
        intent = _intent_parser.parse(description, has_media=bool(uploaded_paths))

        job_store.update(job_id, status=JobStatus.STORYBOARDING, progress_message="Writing your story...")
        storyboard = _storyboard_generator.generate(intent)

        job_store.update(job_id, status=JobStatus.SELECTING_MEDIA, progress_message="Creating visuals and music...")
        media_paths = [Path(p) for p in uploaded_paths]
        storyboard = _media_selector.plan_media(storyboard, uploaded_media=media_paths, variant_seed=variant_seed)
        job_store.update(job_id, storyboard=storyboard)

        job_store.update(job_id, status=JobStatus.RENDERING, progress_message="Rendering your video...")
        job_dir = job_store.job_dir(job_id)
        output_meta = _renderer.render(storyboard, job_dir, resolution_scale=_RESOLUTION_SCALE, variant_seed=variant_seed)

        job_store.update(job_id, status=JobStatus.DONE, progress_message="Your video is ready!", output_meta=output_meta)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            progress_message="Something went wrong.",
            error=f"{exc}",
        )
        traceback.print_exc()


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.post("/api/videos", response_model=CreateVideoResponse)
async def create_video(
    background_tasks: BackgroundTasks,
    description: str = Form(""),
    media: list[UploadFile] = File(default=[]),
) -> CreateVideoResponse:
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="Please describe the video you want to create.")

    job = job_store.create_job()
    job_dir = job_store.job_dir(job.job_id)
    uploads_dir = job_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for i, upload in enumerate(media):
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix or ".bin"
        dest = uploads_dir / f"media_{i:02d}{suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(str(dest))

    job_store.set_uploaded_paths(job.job_id, saved_paths)
    background_tasks.add_task(_run_pipeline, job.job_id, description, saved_paths)
    return CreateVideoResponse(job_id=job.job_id)


@app.get("/api/videos/{job_id}", response_model=JobStatusResponse)
async def get_video_status(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return _to_status_response(job)


@app.get("/api/videos/{job_id}/file")
async def get_video_file(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if job.status != JobStatus.DONE or not job.output_meta:
        raise HTTPException(status_code=409, detail="Video is not ready yet.")
    return FileResponse(job.output_meta.path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.post("/api/videos/{job_id}/variant", response_model=CreateVideoResponse)
async def create_variant(job_id: str, background_tasks: BackgroundTasks) -> CreateVideoResponse:
    """'Try a different version': re-runs MediaSelector + Renderer with a
    new variant seed, holding the same StoryboardPlan (script/captions)
    from the original job. Falls out of the architecture for free because
    MediaSelector/Renderer are decoupled from StoryboardGenerator."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if job.status != JobStatus.DONE or not job.storyboard:
        raise HTTPException(status_code=409, detail="Original video is not ready yet.")

    uploaded_paths = job_store.get_uploaded_paths(job_id)
    variant_seed = job_store.next_variant_seed(job_id)

    job_store.update(job_id, status=JobStatus.SELECTING_MEDIA, progress_message="Creating a new version...")
    background_tasks.add_task(_run_variant_pipeline, job_id, uploaded_paths, variant_seed)
    return CreateVideoResponse(job_id=job_id)


def _run_variant_pipeline(job_id: str, uploaded_paths: list[str], variant_seed: int) -> None:
    try:
        job = job_store.get(job_id)
        storyboard = job.storyboard  # same scenes/captions/narration as before

        job_store.update(job_id, status=JobStatus.SELECTING_MEDIA, progress_message="Creating visuals and music...")
        media_paths = [Path(p) for p in uploaded_paths]
        storyboard = _media_selector.plan_media(storyboard, uploaded_media=media_paths, variant_seed=variant_seed)
        job_store.update(job_id, storyboard=storyboard)

        job_store.update(job_id, status=JobStatus.RENDERING, progress_message="Rendering your video...")
        job_dir = job_store.job_dir(job_id)
        output_meta = _renderer.render(
            storyboard, job_dir, resolution_scale=_RESOLUTION_SCALE, variant_seed=variant_seed
        )
        job_store.update(job_id, status=JobStatus.DONE, progress_message="Your video is ready!", output_meta=output_meta)
    except Exception as exc:  # noqa: BLE001
        job_store.update(job_id, status=JobStatus.FAILED, progress_message="Something went wrong.", error=f"{exc}")
        traceback.print_exc()


# --------------------------------------------------------------------------
# Static frontend (single vanilla HTML/CSS/JS page — see frontend/).
# Mounted last so it doesn't shadow the /api routes above.
# --------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
