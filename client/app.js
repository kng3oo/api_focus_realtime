// ====== 설정 ======
// 로컬 테스트: 서버가 내 PC면 그대로 사용
// AWS 배포 후: 아래를 "http://EC2_PUBLIC_IP:8000" 로 바꾸고,
// client는 로컬에서 python http.server 로 띄우면 카메라가 정상 동작합니다.
const API_BASE = "http://127.0.0.1:8000";

// 캡처/전송 설정
const SEND_INTERVAL_MS = 200;   // ~5fps
const POLL_INTERVAL_MS = 700;
const CAPTURE_WIDTH = 960;      // 저장 이미지/텍스트 가독성 위해 너무 작게 보내지 않음
const JPEG_QUALITY = 0.85;

let userName = "";
let stream = null;
let captureTimer = null;
let pollTimer = null;

const focusHist = [];
const emotionCount = {};

function $(id){ return document.getElementById(id); }

function avgPercent(arr, count){
  if (!arr.length) return 0;
  const slice = arr.slice(-count);
  if (!slice.length) return 0;
  const s = slice.reduce((a,b)=>a+b,0) / slice.length;
  return Math.round(s * 100);
}

async function startCameraIfNeeded(){
  if (stream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    alert("카메라 접근 실패: navigator.mediaDevices.getUserMedia 가 없습니다.\n" +
          "해결: client는 반드시 로컬에서 http://localhost 로 열어주세요.\n" +
          "예) client 폴더에서: python -m http.server 5173");
    return;
  }

  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  $("cam").srcObject = stream;
}

function stopCamera(){
  if (stream){
    for (const t of stream.getTracks()) t.stop();
    stream = null;
    $("cam").srcObject = null;
  }
}

function stopSending(){
  if (captureTimer){ clearInterval(captureTimer); captureTimer = null; }
}

async function sendFrame(){
  const video = $("cam");
  if (!userName) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const ratio = video.videoHeight / video.videoWidth;
  const w = CAPTURE_WIDTH;
  const h = Math.round(w * ratio);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently:false });
  ctx.drawImage(video, 0, 0, w, h);

  const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", JPEG_QUALITY));
  if (!blob) return;

  const fd = new FormData();
  fd.append("frame", blob, "frame.jpg");
  fd.append("user", userName);

  // fire-and-forget
  fetch(`${API_BASE}/api/frame`, { method: "POST", body: fd }).catch(()=>{});
}

async function poll(){
  if (!userName) return; // 사용자 중지가 되어 있어도 계속 “이상한 세션” 쌓이지 않게
  try{
    const liveRes = await fetch(`${API_BASE}/api/live?user=${encodeURIComponent(userName)}`);
    const live = await liveRes.json();

    const sessRes = await fetch(`${API_BASE}/api/session?user=${encodeURIComponent(userName)}`);
    const sess = await sessRes.json();

    const focusArr = live.focus || [];
    const fps = live.fps || 0;
    const latest = live.latest || {};

    if (focusArr.length){
      const last = focusArr[focusArr.length - 1];
      focusHist.push(last);
      $("focusNow").innerText = Math.round(last * 100) + "%";
    }

    const fpsSafe = Math.max(5, fps || 10);
    const n10 = Math.round(fpsSafe * 10);
    const n60 = Math.round(fpsSafe * 60);
    const n600 = Math.round(fpsSafe * 600);

    $("avg10").innerText  = avgPercent(focusHist, n10) + "%";
    $("avg60").innerText  = avgPercent(focusHist, n60) + "%";
    $("avg600").innerText = avgPercent(focusHist, n600) + "%";

    $("fps").innerText = (typeof fps === "number") ? fps.toFixed(1) : fps;
    $("frames").innerText = sess.frames ?? 0;
    $("saved").innerText = sess.saved ?? 0;

    // latest 출력 (모든 수치 %화)
    const latestCopy = { ...latest };
    for (const k of ["focus","emotion_score","blink_score","gaze_score","neck_score"]){
      if (latestCopy[k] != null && typeof latestCopy[k] === "number"){
        latestCopy[k] = (latestCopy[k] * 100).toFixed(1) + "%";
      }
    }
    $("latest").innerText = JSON.stringify(latestCopy, null, 2);

    // 감정 빈도수 랭킹
    const topEmotion = latest.top_emotion;
    if (topEmotion){
      emotionCount[topEmotion] = (emotionCount[topEmotion] || 0) + 1;
    }
    const rankList = Object.entries(emotionCount)
      .sort((a,b)=>b[1]-a[1])
      .map(([k,v]) => `<li>${k}: ${v}회</li>`).join("");
    $("emotionRank").innerHTML = rankList || "<li>데이터 없음</li>";

  }catch(e){
    console.error(e);
  }
}

// ====== UI 핸들러 ======
async function onStart(){
  const name = $("username").value.trim();
  if (!name){
    alert("사용자 이름을 먼저 입력해주세요.");
    return;
  }
  userName = name;

  await startCameraIfNeeded();
  if (!stream) return;

  if (!captureTimer){
    captureTimer = setInterval(sendFrame, SEND_INTERVAL_MS);
  }
  if (!pollTimer){
    pollTimer = setInterval(poll, POLL_INTERVAL_MS);
  }
}

function onStop(){
  // 전송 중단 + 세션명 비우기(핵심)
  stopSending();
  userName = "";
  // 카메라도 같이 끌지 여부: 원하시면 끄는게 깔끔합니다.
  stopCamera();

  // UI 초기화
  $("focusNow").innerText = "0%";
  $("avg10").innerText = "0%";
  $("avg60").innerText = "0%";
  $("avg600").innerText = "0%";
  $("latest").innerText = "stopped";
}

async function onSnapshot(){
  const name = $("username").value.trim();
  if (!name){
    alert("사용자 이름을 먼저 입력하고 시작하세요.");
    return;
  }
  try{
    const res = await fetch(`${API_BASE}/api/snapshot`, {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ user: name })
    });
    const j = await res.json();
    alert(j.ok ? "✅ 현재 기준 이미지가 서버에 저장되었습니다." : "❌ 저장할 프레임이 아직 없습니다.");
  }catch(e){
    alert("이미지 저장 중 오류: " + e);
  }
}

function onDownloadLog(){
  const name = $("username").value.trim();
  if (!name){ alert("사용자 이름을 입력해주세요."); return; }
  window.open(`${API_BASE}/api/download/log?user=${encodeURIComponent(name)}`, "_blank");
}

function onDownloadZip(){
  const name = $("username").value.trim();
  if (!name){ alert("사용자 이름을 입력해주세요."); return; }
  window.open(`${API_BASE}/api/download/screenshots?user=${encodeURIComponent(name)}`, "_blank");
}

// 버튼 연결
window.addEventListener("load", () => {
  $("btnStart").addEventListener("click", onStart);
  $("btnStop").addEventListener("click", onStop);
  $("btnSnap").addEventListener("click", onSnapshot);
  $("btnLog").addEventListener("click", onDownloadLog);
  $("btnZip").addEventListener("click", onDownloadZip);
});
