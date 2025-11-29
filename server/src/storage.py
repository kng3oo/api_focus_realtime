from __future__ import annotations
import os
import re
import io
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont


def safe_user_id(user: str) -> str:
    # 파일명은 ASCII + 안정성(한글/중국어/이모지 방지)
    h = hashlib.sha1(user.encode("utf-8")).hexdigest()[:10]
    return f"user_{h}"

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_korean_font() -> Optional[str]:
    candidates = [
        # Windows (한글)
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\Malgun.ttf",
        # Linux (Noto 계열 가능 경로들)
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def draw_overlay_bgr(
    frame_bgr: np.ndarray,
    user_display: str,
    metrics: Dict[str, Any],
    weights: Dict[str, float],
    save_width: int = 1280
) -> np.ndarray:
    # 가독성 위해 최소 폭 보정
    h, w = frame_bgr.shape[:2]
    if w < save_width:
        scale = save_width / max(1, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    font_path = find_korean_font()
    if font_path:
        font = ImageFont.truetype(font_path, 28)
        font_small = ImageFont.truetype(font_path, 22)
    else:
        # 폰트 없으면 기본(한글 깨질 수 있음). 그래도 서버는 죽지 않게.
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 패널 배경
    pad = 12
    x0, y0 = 12, 12
    panel_w, panel_h = 560, 240
    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill=(5, 10, 25, 180))

    def p(v: Optional[float]) -> str:
        if v is None: return "N/A"
        return f"{v*100:.1f}%"

    focus = metrics.get("focus")
    emo   = metrics.get("emotion_score")
    blink = metrics.get("blink_score")
    gaze  = metrics.get("gaze_score")
    neck  = metrics.get("neck_score")
    top_emotion = metrics.get("top_emotion")

    lines = [
        f"User: {user_display}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Focus: {p(focus)}   (E:{weights.get('emotion',0):.2f} + B:{weights.get('blink',0):.2f})",
        f"Emotion: {p(emo)}   Top: {top_emotion}",
        f"Blink: {p(blink)}   EAR: {metrics.get('ear', 'N/A')}",
        f"Gaze: {p(gaze)}     Neck: {p(neck)}",
    ]

    y = y0 + pad
    for i, t in enumerate(lines):
        draw.text((x0 + pad, y), t, font=(font if i < 3 else font_small), fill=(230, 230, 230))
        y += 34 if i < 2 else 28

    # 하단 Focus 바
    out = np.array(img)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    H, W = out_bgr.shape[:2]
    bar_x1, bar_y1 = 12, H - 40
    bar_x2, bar_y2 = W - 12, H - 16
    cv2.rectangle(out_bgr, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 70, 90), -1)
    if isinstance(focus, (int, float)):
        fill = int((bar_x2 - bar_x1) * max(0.0, min(1.0, float(focus))))
        cv2.rectangle(out_bgr, (bar_x1, bar_y1), (bar_x1 + fill, bar_y2), (0, 220, 255), -1)

    return out_bgr


def zip_dir(dir_path: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in dir_path.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(dir_path))
