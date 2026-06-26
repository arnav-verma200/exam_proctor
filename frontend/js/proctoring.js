import { getApiBase } from './api.js';
import { setCandidate, clearCandidate } from './auth.js';

const FLAGGED_KEYWORDS = [
  "answer","solution","help me","tell me","what is","copy","cheat",
  "share screen","hey google","hey siri","alexa","ok google",
  "pass","question","listen","hey","psst","whisper",
];

let currentAppNumber   = null;
let currentStudentName = null;
let faceApiReady       = false;
let webcamStream       = null;
let audioCtx           = null;
let audioAnalyser      = null;
let audioLevelTimer    = null;
let speechRec          = null;
let faceDetectTimer    = null;
let framePushTimer     = null;
let visibilityBound    = false;
let consecutiveNoFace  = 0;

const COOLDOWNS    = {};
const COOLDOWN_MS  = {
  "Face Not Detected": 8000,
  "Multiple Faces":    8000,
  "Audio Detected":    5000,
  "Tab Switch":        3000,
  "Browser Blur":      3000,
};

function pad(n) { return String(n).padStart(2, '0'); }

function canFire(eventType) {
  const now = Date.now();
  if ((COOLDOWNS[eventType] || 0) > now) return false;
  COOLDOWNS[eventType] = now + (COOLDOWN_MS[eventType] || 6000);
  return true;
}

async function loadFaceModels() {
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
    ]);
    faceApiReady = true;
    const el = document.getElementById("faceStatus");
    if (el) el.textContent = "Ready";
  } catch (e) {
    console.warn("[proctor] face model load:", e);
  }
}

async function flagEvent(eventType, confidence, note) {
  if (!currentAppNumber) return;
  try {
    await fetch(getApiBase() + "/api/proctor/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        app_number: currentAppNumber,
        event_type: eventType,
        confidence: confidence,
        note: note,
      }),
    });
  } catch {}
}

async function captureSnapshot(eventType) {
  const vid = document.getElementById("studentVideo");
  if (!vid || vid.readyState < 2) return;
  try {
    const c = document.createElement("canvas");
    c.width  = vid.videoWidth  || 320;
    c.height = vid.videoHeight || 240;
    c.getContext("2d").drawImage(vid, 0, 0, c.width, c.height);
    const dataUrl = c.toDataURL("image/jpeg", 0.65);
    await fetch(getApiBase() + "/api/proctor/snapshot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        app_number:   currentAppNumber,
        student_name: currentStudentName,
        event_type:   eventType,
        image:        dataUrl,
      }),
    });
  } catch {}
}

function startFramePusher(videoEl, appNo) {
  const cap = document.createElement("canvas");
  cap.width  = 320;
  cap.height = 240;
  const ctx = cap.getContext("2d");

  async function pushFrame() {
    if (!webcamStream || videoEl.readyState < 2) {
      framePushTimer = setTimeout(pushFrame, 400);
      return;
    }
    try {
      ctx.drawImage(videoEl, 0, 0, cap.width, cap.height);
      cap.toBlob(async blob => {
        if (!blob) return;
        try {
          await fetch(getApiBase() + "/proctor/frame/" + appNo, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: blob,
            credentials: "include",
          });
        } catch {}
      }, "image/jpeg", 0.6);
    } catch {}
    framePushTimer = setTimeout(pushFrame, 200);
  }
  pushFrame();
}

async function startFaceDetection() {
  const vid    = document.getElementById("studentVideo");
  const canvas = document.getElementById("faceCanvas");
  if (!vid || !canvas) return;
  const opts   = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 });

  async function loop() {
    if (!faceApiReady || vid.readyState < 2) {
      faceDetectTimer = setTimeout(loop, 800);
      return;
    }

    canvas.width  = vid.videoWidth  || 220;
    canvas.height = vid.videoHeight || 165;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const dets  = await faceapi.detectAllFaces(vid, opts);
      const count = dets.length;
      const fEl   = document.getElementById("faceStatus");

      if (count === 0) {
        consecutiveNoFace++;
        if (fEl) { fEl.textContent = "No Face"; fEl.style.color = "#e74c3c"; }
        if (consecutiveNoFace >= 2 && canFire("Face Not Detected")) {
          await flagEvent("Face Not Detected", 92, "Face absent");
          await captureSnapshot("Face Not Detected");
        }
      } else if (count > 1) {
        consecutiveNoFace = 0;
        if (fEl) { fEl.textContent = count + " Faces"; fEl.style.color = "#f0ad4e"; }
        dets.forEach(d => {
          const { x, y, width: w, height: h } = d.box;
          ctx.strokeStyle = "#f0ad4e";
          ctx.lineWidth   = 2;
          ctx.strokeRect(x, y, w, h);
        });
        if (canFire("Multiple Faces")) {
          await flagEvent("Multiple Faces", Math.min(99, 80 + count * 5), count + " faces");
          await captureSnapshot("Multiple Faces");
        }
      } else {
        consecutiveNoFace = 0;
        if (fEl) { fEl.textContent = "Face OK"; fEl.style.color = "#35ed7e"; }
        const { x, y, width: w, height: h } = dets[0].box;
        ctx.strokeStyle = "#35ed7e";
        ctx.lineWidth   = 2;
        ctx.strokeRect(x, y, w, h);
      }
    } catch {}

    faceDetectTimer = setTimeout(loop, 1000);
  }
  loop();
}

function startAudioMonitoring() {
  if (!webcamStream) return;
  try {
    audioCtx      = new (window.AudioContext || window.webkitAudioContext)();
    const src     = audioCtx.createMediaStreamSource(webcamStream);
    audioAnalyser = audioCtx.createAnalyser();
    audioAnalyser.fftSize = 512;
    src.connect(audioAnalyser);

    const data = new Uint8Array(audioAnalyser.frequencyBinCount);

    audioLevelTimer = setInterval(() => {
      audioAnalyser.getByteFrequencyData(data);
      const avg = data.reduce((a, b) => a + b, 0) / data.length;
      if (avg > 34 && canFire("Audio Detected")) {
        flagEvent("Audio Detected", Math.min(99, Math.round(avg * 2.8)), "Loud audio");
      }
    }, 1500);
  } catch {}
}

function startSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;

  speechRec = new SR();
  speechRec.continuous     = true;
  speechRec.interimResults = true;
  speechRec.lang           = "en-IN";

  speechRec.onresult = async ev => {
    const transcript = Array.from(ev.results)
      .map(r => r[0].transcript.toLowerCase())
      .join(" ");
    for (const kw of FLAGGED_KEYWORDS) {
      if (transcript.includes(kw) && canFire("Audio Detected")) {
        await flagEvent("Audio Detected", 91, `Keyword: "${kw}"`);
        break;
      }
    }
  };

  speechRec.onerror = () => {};
  speechRec.onend   = () => { try { speechRec.start(); } catch {} };

  try { speechRec.start(); } catch {}
}

function startTabMonitoring() {
  if (visibilityBound) return;
  visibilityBound = true;

  document.addEventListener("visibilitychange", async () => {
    if (document.hidden && canFire("Tab Switch")) {
      await flagEvent("Tab Switch", 95, "Tab switched");
    }
  });

  window.addEventListener("blur", async () => {
    if (canFire("Browser Blur")) {
      await flagEvent("Browser Blur", 85, "Window lost focus");
    }
  });
}

export async function initProctoring(appNo, name) {
  currentAppNumber   = appNo;
  currentStudentName = name;

  const webcamBox = document.getElementById("webcamBox");
  if (webcamBox) webcamBox.style.display = "block";

  loadFaceModels();

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: true,
    });

    const vid = document.getElementById("studentVideo");
    if (vid) {
      vid.srcObject = webcamStream;
      await vid.play();
    }

    const camDot  = document.getElementById("camDot");
    const camText = document.getElementById("camText");
    if (camDot) camDot.style.background = "#35ed7e";
    if (camText) camText.textContent = "AI Active · Webcam On";

    await setCandidate(appNo, name);

    if (vid) startFramePusher(vid, appNo);
    startFaceDetection();
    startAudioMonitoring();
    startSpeechRecognition();

  } catch (err) {
    const camDot  = document.getElementById("camDot");
    const camText = document.getElementById("camText");
    if (camDot) camDot.style.background = "#e74c3c";
    if (camText) camText.textContent = "Camera denied — limited proctoring";
  }

  startTabMonitoring();
}

export function getCurrentAppNumber() { return currentAppNumber; }

export function stopProctoring() {
  clearTimeout(faceDetectTimer);
  clearTimeout(framePushTimer);
  clearInterval(audioLevelTimer);
  if (speechRec) { try { speechRec.stop(); } catch {} }
  if (audioCtx)  { audioCtx.close(); audioCtx = null; }
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  clearCandidate();
  const webcamBox = document.getElementById("webcamBox");
  if (webcamBox) webcamBox.style.display = "none";
}

window.addEventListener("beforeunload", stopProctoring);
