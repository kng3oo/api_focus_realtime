from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np
import mediapipe as mp
import torch
import timm
from torchvision import transforms


DEFAULT_LABELS = ["anger","disgust","fear","happy","none","sad","surprise"]


def _safe_torch_load(path: str, device: str):
    # torch 2.6+ weights_only 기본값 변경 대응 + 구버전 대응
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class EmotionModel:
    def __init__(self, model_path: str, labels=DEFAULT_LABELS, img_size=384, device: Optional[str]=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = labels
        self.img_size = img_size

        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(labels))

        self.ready = False
        if os.path.exists(model_path):
            sd = _safe_torch_load(model_path, self.device)
            # sd 형태가 다양한 경우 대응
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # module. prefix 제거
            if isinstance(sd, dict):
                new_sd = {}
                for k, v in sd.items():
                    nk = k.replace("module.", "")
                    new_sd[nk] = v
                sd = new_sd
            try:
                self.model.load_state_dict(sd, strict=False)
                self.model.to(self.device).eval()
                self.ready = True
            except Exception:
                self.ready = False

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        # 정책: 부정 감정 합 감점
        self.negative = set(["anger","disgust","fear","sad"])

    def score(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, float], Optional[str]]:
        if not self.ready:
            # 모델이 없거나 실패해도 서버가 죽지 않게
            return 0.5, {lab: 1.0/len(self.labels) for lab in self.labels}, "none"

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        probs = {lab: float(p) for lab, p in zip(self.labels, prob)}
        top = max(probs, key=probs.get)

        neg_sum = sum(probs.get(k, 0.0) for k in self.negative)
        score = float(np.clip(1.0 - neg_sum, 0.0, 1.0))
        return score, probs, top


class BlinkEstimator:
    def __init__(self, ear_close=0.18, ear_open=0.30, win=3):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.left_idx  = [33, 160, 158, 133, 153, 144]
        self.right_idx = [263, 387, 385, 362, 380, 373]
        self.ear_close, self.ear_open = ear_close, ear_open
        self.win = win
        self.hist = []
        self.blinks = 0
        self.prev_closed = False

    def _ear(self, e):
        A = np.linalg.norm(e[1]-e[5])
        B = np.linalg.norm(e[2]-e[4])
        C = np.linalg.norm(e[0]-e[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def score(self, frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None, None

        lm = res.multi_face_landmarks[0].landmark
        def xy(i): return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        L = np.array([xy(i) for i in self.left_idx])
        R = np.array([xy(i) for i in self.right_idx])
        ear = float((self._ear(L) + self._ear(R)) / 2.0)

        s = (ear - self.ear_close) / (self.ear_open - self.ear_close + 1e-6)
        blink_score = float(np.clip(s, 0.0, 1.0))

        closed = ear < self.ear_close
        self.hist.append(closed)
        if len(self.hist) > self.win:
            self.hist = self.hist[-self.win:]

        now_closed = sum(self.hist) >= (self.win//2 + 1)
        blink_evt = (self.prev_closed and not now_closed)
        if blink_evt:
            self.blinks += 1
        self.prev_closed = now_closed

        # blink_rate는 서버에서 fps 기반 계산이 애매하니 "누적 blink"를 참고값으로 둠
        return blink_score, ear, float(self.blinks)


class GazeEstimator:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.left_eye_corners  = [33, 133]
        self.right_eye_corners = [362, 263]
        self.left_iris  = [468, 469, 470, 471]
        self.right_iris = [473, 474, 475, 476]

    def _center(self, pts):
        pts = np.asarray(pts, dtype=np.float32)
        return pts.mean(axis=0)

    def score(self, frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None
        lm = res.multi_face_landmarks[0].landmark
        def xy(i): return np.array([lm[i].x*w, lm[i].y*h], dtype=np.float32)

        Lc = [xy(i) for i in self.left_eye_corners]
        Rc = [xy(i) for i in self.right_eye_corners]
        Li = [xy(i) for i in self.left_iris]
        Ri = [xy(i) for i in self.right_iris]

        eye_center  = (self._center(Lc) + self._center(Rc)) / 2.0
        iris_center = (self._center(Li) + self._center(Ri)) / 2.0
        eye_w = np.linalg.norm(np.array(Lc[0]) - np.array(Lc[1])) + 1e-6

        off = (iris_center - eye_center) / eye_w
        dist = np.linalg.norm(off)
        gaze_score = float(np.clip(1.0 - dist * 2.0, 0.0, 1.0))
        return gaze_score, off


class HeadPoseEstimator:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.model_3d = np.array([
            [0.0,   0.0,   0.0],     # nose tip (1)
            [0.0,  -63.6, -12.5],    # chin (152)
            [-43.3, 32.7, -26.0],    # left eye corner (33)
            [ 43.3, 32.7, -26.0],    # right eye corner (263)
            [-28.9,-28.9, -24.1],    # left mouth corner (61)
            [ 28.9,-28.9, -24.1],    # right mouth corner (291)
        ], dtype=np.float64)
        self.idxs = [1, 152, 33, 263, 61, 291]

    def _euler_from_R(self, R):
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            roll  = np.degrees(np.arctan2(R[1,0], R[0,0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            roll  = 0.0
        return yaw, pitch, roll

    def score(self, frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[float,float,float]]]:
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None

        lm = res.multi_face_landmarks[0].landmark
        pts2d = np.array([[lm[i].x * w, lm[i].y * h] for i in self.idxs], dtype=np.float64)

        f = max(h, w)
        cam_mtx = np.array([[f, 0, w/2],
                            [0, f, h/2],
                            [0, 0,   1 ]], dtype=np.float64)
        dist = np.zeros((4,1))

        ok, rvec, tvec = cv2.solvePnP(self.model_3d, pts2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None
        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._euler_from_R(R)

        s = 1.0 - (abs(yaw)/25 + abs(pitch)/25 + abs(roll)/20)/3
        score = float(np.clip(s, 0.0, 1.0))
        return score, (float(yaw), float(pitch), float(roll))


def fuse_focus(emotion: Optional[float], blink: Optional[float], w_emotion=0.7, w_blink=0.3) -> float:
    e = float(emotion or 0.0)
    b = float(blink or 0.0)
    return float(np.clip(w_emotion*e + w_blink*b, 0.0, 1.0))


class FocusPipeline:
    def __init__(self, model_path: str):
        self.emotion = EmotionModel(model_path=model_path)
        self.blink   = BlinkEstimator()
        self.gaze    = GazeEstimator()
        self.neck    = HeadPoseEstimator()

        self.weights = {"emotion": 0.7, "blink": 0.3}
        self.threshold = 0.40

    def score(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        e_score, e_probs, top = self.emotion.score(frame_bgr)
        b_score, ear, blinks  = self.blink.score(frame_bgr)
        g_score, g_off        = self.gaze.score(frame_bgr)
        n_score, angles       = self.neck.score(frame_bgr)

        focus = fuse_focus(e_score, b_score, self.weights["emotion"], self.weights["blink"])

        return {
            "focus": focus,
            "emotion_score": e_score,
            "blink_score": b_score,
            "gaze_score": g_score,
            "neck_score": n_score,
            "emotion_probs": e_probs,
            "top_emotion": top,
            "ear": ear,
            "blinks": blinks,
            "gaze_offset": g_off.tolist() if g_off is not None else None,
            "angles": angles,  # (yaw,pitch,roll)
            "weights": dict(self.weights),
            "threshold": self.threshold
        }
