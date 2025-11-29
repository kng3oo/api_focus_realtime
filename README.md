# Focus Realtime (Local camera -> Server 분석)

## 로컬 실행(내 PC에서 서버+클라이언트 확인)

### 1) 서버 실행
```bash
cd focus_realtime/server
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements_local.txt
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

### 2) 클라이언트 실행(카메라가 되려면 file:// 더블클릭 말고 localhost로 띄우세요)
cd focus_realtime/client
python -m http.server 5173


폴더구조
API_focus_realtime/
├─ client/
│  ├─ index.html
│  └─ app.js
├─ server/
│  ├─ models/
│  │  └─ best_model.pth              # (로컬/EC2에서 다운로드로 채움)
│  ├─ runs/                          # CSV 로그 저장
│  ├─ screenshots/                   # 조건부 캡처 저장(오버레이 포함)
│  ├─ requirements_local.txt
│  ├─ requirements_ec2.txt
│  └─ src/
│     ├─ __init__.py
│     ├─ main.py
│     ├─ camera_worker.py
│     ├─ model_utils.py
│     ├─ storage.py
│     ├─ logger.py
│     └─ schemas.py
├─ cloudformation.yaml
└─ README.md

cd ~/focus_realtime
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000

sudo yum install -y mesa-libGL"# api_focus_realtime" 
