from __future__ import annotations
import time
import threading
import queue
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .model_utils import FocusPipeline
from .logger import SafeCSVLogger
from .storage import safe_user_id, now_tag, draw_overlay_bgr, zip_dir
import cv2


BASE_DIR = Path(__file__).resolve().parents[1]   # server/
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
SHOTS_DIR = BASE_DIR / "screenshots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SHOTS_DIR.mkdir(parents=True, exist_ok=True)


# 공유 상태(서버가 /api/live 로 제공)
shared_state = {
    "lock": threading.Lock(),
    "fps": 0.0,
    "latest": {},        # user -> latest dict
    "series": {},        # user -> dict of deques
    "meta": {},          # user -> frames, saved, csv_path, shots_dir
}

_frame_q: "queue.Queue[Tuple[str, np.ndarray]]" = queue.Queue(maxsize=4)
_stop_event = threading.Event()

_pipeline = FocusPipeline(model_path=str(MODELS_DIR / "best_model.pth"))

# 사용자별 로거/경로 캐시
_user_loggers: Dict[str, SafeCSVLogger] = {}
_user_last_save_ts: Dict[str, float] = {}


def _ensure_user(user: str):
    uid = safe_user_id(user)
    user_run_dir = RUNS_DIR / uid
    user_shot_dir = SHOTS_DIR / uid
    user_run_dir.mkdir(parents=True, exist_ok=True)
    user_shot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = user_run_dir / "focus_log.csv"
    if uid not in _user_loggers:
        _user_loggers[uid] = SafeCSVLogger(
            str(csv_path),
            header=[
                "ts","focus","emotion_score","blink_score","gaze_score","neck_score",
                "top_emotion","ear","blinks","yaw","pitch","roll",
                "w_emotion","w_blink","threshold"
            ]
        )

    with shared_state["lock"]:
        if user not in shared_state["series"]:
            shared_state["series"][user] = {
                "time": deque(maxlen=1800),
                "focus": deque(maxlen=1800),
            }
        if user not in shared_state["meta"]:
            shared_state["meta"][user] = {
                "frames": 0,
                "saved": 0,
                "csv_path": str(csv_path),
                "shots_dir": str(user_shot_dir),
                "uid": uid
            }

    return uid, csv_path, user_shot_dir


def push_frame(user: str, frame_bgr: np.ndarray):
    _ensure_user(user)
    try:
        # 큐가 꽉 차면 오래된 프레임 버리고 최신만 반영
        if _frame_q.full():
            try:
                _frame_q.get_nowait()
            except Exception:
                pass
        _frame_q.put_nowait((user, frame_bgr))
    except Exception:
        pass


def _maybe_save_snapshot(user: str, frame_bgr: np.ndarray, result: Dict[str, Any], reason: str):
    uid, _, shots_dir = _ensure_user(user)

    # 너무 자주 저장되지 않게(예: 3초 쿨다운)
    now = time.time()
    last = _user_last_save_ts.get(uid, 0.0)
    if reason == "auto" and now - last < 3.0:
        return False

    out = draw_overlay_bgr(
        frame_bgr=frame_bgr,
        user_display=user,         # 한글 표기(폰트 있으면 정상)
        metrics=result,
        weights=result.get("weights", {"emotion":0.7,"blink":0.3}),
        save_width=1280
    )

    fname = f"{now_tag()}_{uid}_{reason}.png"
    path = shots_dir / fname
    cv2.imwrite(str(path), out)

    _user_last_save_ts[uid] = now
    with shared_state["lock"]:
        shared_state["meta"][user]["saved"] += 1
    return True


def save_snapshot_now(user: str) -> bool:
    # 마지막 프레임이 없으면 저장 불가 → main.py 에서 처리
    with shared_state["lock"]:
        latest = shared_state["latest"].get(user)
    if not latest:
        return False
    frame = latest.get("_last_frame")
    if frame is None:
        return False
    result = latest.get("_last_result")
    if not result:
        return False
    return _maybe_save_snapshot(user, frame, result, "manual")


def make_screenshots_zip(user: str) -> Optional[str]:
    uid, _, shot_dir = _ensure_user(user)
    out_zip = (SHOTS_DIR / uid) / "screenshots.zip"
    zip_dir(Path(shot_dir), out_zip)
    return str(out_zip)


def get_live(user: str) -> Dict[str, Any]:
    with shared_state["lock"]:
        s = shared_state["series"].get(user)
        latest = shared_state["latest"].get(user, {})
        fps = shared_state.get("fps", 0.0)
    if not s:
        return {"focus": [], "fps": fps, "latest": {}}
    return {
        "focus": list(s["focus"])[-180:],  # 최근만
        "fps": fps,
        "latest": {k:v for k,v in latest.items() if not k.startswith("_")}
    }


def get_session(user: str) -> Dict[str, Any]:
    with shared_state["lock"]:
        m = shared_state["meta"].get(user)
    if not m:
        return {"frames": 0, "saved": 0}
    return {"frames": m["frames"], "saved": m["saved"]}


def start_camera_loop():
    if getattr(start_camera_loop, "_started", False):
        return
    start_camera_loop._started = True

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    print("🔁 Processing worker thread started.")


def stop_camera_loop():
    _stop_event.set()
    for lg in _user_loggers.values():
        lg.close()


def _loop():
    t_q = deque(maxlen=50)

    while not _stop_event.is_set():
        try:
            user, frame = _frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        uid, csv_path, shot_dir = _ensure_user(user)

        t_q.append(time.time())
        if len(t_q) >= 2:
            fps = len(t_q) / max(1e-6, (t_q[-1] - t_q[0]))
            with shared_state["lock"]:
                shared_state["fps"] = fps

        # 분석
        result = _pipeline.score(frame)

        # CSV 로그 저장 (매 프레임 flush)
        yaw, pitch, roll = (result.get("angles") or (0.0, 0.0, 0.0))
        _user_loggers[uid].write([
            now_tag(),
            result.get("focus"),
            result.get("emotion_score"),
            result.get("blink_score"),
            result.get("gaze_score"),
            result.get("neck_score"),
            result.get("top_emotion"),
            result.get("ear"),
            result.get("blinks"),
            yaw, pitch, roll,
            result.get("weights", {}).get("emotion", 0.7),
            result.get("weights", {}).get("blink", 0.3),
            result.get("threshold", 0.40),
        ])

        # shared_state 갱신
        with shared_state["lock"]:
            shared_state["meta"][user]["frames"] += 1
            shared_state["series"][user]["time"].append(time.time())
            shared_state["series"][user]["focus"].append(float(result.get("focus") or 0.0))

            # latest에 프레임을 그대로 넣으면 /api가 무거워지므로 숨김키로만 보관
            shared_state["latest"][user] = {
                **result,
                "_last_frame": frame,
                "_last_result": result,
            }

        # 자동 저장(40% 미만)
        if (result.get("focus") is not None) and (float(result["focus"]) < float(result.get("threshold", 0.40))):
            _maybe_save_snapshot(user, frame, result, "auto")
