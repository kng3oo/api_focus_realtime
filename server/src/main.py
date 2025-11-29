from __future__ import annotations
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .camera_worker import (
    start_camera_loop,
    push_frame,
    get_live,
    get_session,
    save_snapshot_now,
    make_screenshots_zip,
    shared_state
)
from .schemas import SnapshotRequest

BASE_DIR = Path(__file__).resolve().parents[1]  # server/
RUNS_DIR = BASE_DIR / "runs"
SHOTS_DIR = BASE_DIR / "screenshots"


app = FastAPI(title="Focus Realtime API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # client가 로컬에서 뜨므로 CORS 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    start_camera_loop()

@app.get("/")
def root():
    return {"ok": True, "message": "Focus Realtime API is running"}

@app.post("/api/frame")
async def api_frame(frame: UploadFile = File(...), user: str = Form(...)):
    data = await frame.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"ok": False, "error": "invalid image"}, status_code=400)

    push_frame(user, img)
    return {"ok": True}

@app.get("/api/live")
def api_live(user: str = Query(...)):
    return get_live(user)

@app.get("/api/session")
def api_session(user: str = Query(...)):
    return get_session(user)

@app.post("/api/snapshot")
def api_snapshot(req: SnapshotRequest):
    ok = save_snapshot_now(req.user)
    return {"ok": ok}

@app.get("/api/download/log")
def download_log(user: str = Query(...)):
    # camera_worker에서 meta에 csv_path 저장
    with shared_state["lock"]:
        meta = shared_state["meta"].get(user)
    if not meta:
        return JSONResponse({"ok": False, "error": "no session"}, status_code=404)

    p = meta.get("csv_path")
    if not p or not Path(p).exists():
        return JSONResponse({"ok": False, "error": "log not found"}, status_code=404)

    return FileResponse(p, media_type="text/csv", filename="focus_log.csv")

@app.get("/api/download/screenshots")
def download_screenshots(user: str = Query(...)):
    zip_path = make_screenshots_zip(user)
    if not zip_path or not Path(zip_path).exists():
        return JSONResponse({"ok": False, "error": "zip not found"}, status_code=404)
    return FileResponse(zip_path, media_type="application/zip", filename="screenshots.zip")
