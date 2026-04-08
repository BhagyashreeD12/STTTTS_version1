"""
Insurance Voice Agent
STT : faster-whisper  (swap to Kyutai STT — see KyutaiSTT section below)
LLM : OpenAI API      (modular brain — model controlled by LLM_MODEL in .env)
TTS : Kokoro-ONNX     (swap to Kyutai TTS — see KyutaiTTS section below)
Run : .venv\\Scripts\\python.exe scripts/voice_assistant.py
"""

from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

print("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))

import os
import ssl
import io
import wave
import tempfile
import base64
import asyncio
import threading
import queue
import json as _json
import traceback
import re
import time
import subprocess
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np
import av
import torch

# ── SSL / HuggingFace ──────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)
else:
    os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"

# ======================================================================
# DEBUG
# ======================================================================

DEBUG = True

# ======================================================================
# Optional backend globals (predeclared for Pylance / conditional imports)
# ======================================================================

# Kyutai STT globals
julius = None
_kyutai_info = None
_kyutai_mimi = None
_kyutai_tok = None
_kyutai_lm = None
_kyutai_lm_gen = None
_kyutai_pad_id = None
_kyutai_prefix = None
_kyutai_delay = None

# Kyutai TTS globals
_tts_cp = None
_tts_voice_path = None
_tts_cond = None


def dlog(*args):
    if DEBUG:
        print(*args)


# ======================================================================
# STT BACKEND — choose ONE block below and comment out the other
# ======================================================================

# ── Option A: faster-whisper (current, default) ───────────────────────
USE_KYUTAI_STT = False  # set True to switch to Kyutai

if not USE_KYUTAI_STT:
    print("Loading STT (Whisper base)...")
    from faster_whisper import WhisperModel

    stt_model = WhisperModel(
        "base", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8"
    )
    print("STT ready.")

# ── Option B: Kyutai STT ──────────────────────────────────────────────
# Uncomment & set USE_KYUTAI_STT = True to activate.
#
# if USE_KYUTAI_STT:
#     import math, itertools
#     import julius
#     import moshi.models
#     import sphn
#     print("Loading STT (Kyutai)...")
#     _kyutai_hf_repo = "kyutai/stt-2.6b-en"
#     _kyutai_info    = moshi.models.loaders.CheckpointInfo.from_hf_repo(_kyutai_hf_repo)
#     _kyutai_mimi    = _kyutai_info.get_mimi(device=DEVICE)
#     _kyutai_tok     = _kyutai_info.get_text_tokenizer()
#     _kyutai_lm      = _kyutai_info.get_moshi(device=DEVICE, dtype=torch.bfloat16)
#     _kyutai_lm_gen  = moshi.models.LMGen(_kyutai_lm, temp=0, temp_text=0.0)
#     _kyutai_pad_id  = _kyutai_info.raw_config.get("text_padding_token_id", 3)
#     _kyutai_prefix  = _kyutai_info.stt_config.get("audio_silence_prefix_seconds", 1.0)
#     _kyutai_delay   = _kyutai_info.stt_config.get("audio_delay_seconds", 5.0)
#     print("STT (Kyutai) ready.")

# ======================================================================
# TTS BACKEND — choose ONE block below
# ======================================================================

# ── Option A: Kokoro ONNX (current, default) ──────────────────────────
USE_KYUTAI_TTS = False  # set True to switch to Kyutai

if not USE_KYUTAI_TTS:
    print("Loading TTS (Kokoro)...")
    from kokoro_onnx import Kokoro
    from pathlib import Path
    import urllib.request

    _cache = Path.home() / ".cache" / "kokoro_onnx"
    _cache.mkdir(parents=True, exist_ok=True)
    _BASE_URL = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    )

    def _ensure(filename: str) -> str:
        dest = _cache / filename
        if not dest.exists():
            print(f"  Downloading {filename} ...")
            urllib.request.urlretrieve(f"{_BASE_URL}/{filename}", str(dest))
        return str(dest)

    tts_model = Kokoro(_ensure("kokoro-v1.0.onnx"), _ensure("voices-v1.0.bin"))
    print("TTS ready.")

    # Load the active voice profile (driven by VOICE_PROFILE env var, default: kokoro_heart)
    from voice_profiles import get_active_voice_profile

    _voice_profile = get_active_voice_profile()
    print(f"Voice profile: {_voice_profile.display_name}")

    try:

        async def _warmup():
            async for _ in tts_model.create_stream(
                "Hello.",
                voice=_voice_profile.voice_id,
                speed=_voice_profile.speed,
                lang=_voice_profile.language,
            ):
                break

        asyncio.run(_warmup())
        print("TTS warmed up.")
    except Exception:
        pass

# ── Option B: Kyutai TTS (streaming) ──────────────────────────────────
# Uncomment & set USE_KYUTAI_TTS = True to activate.
#
# if USE_KYUTAI_TTS:
#     from moshi.models.loaders import CheckpointInfo as _TTS_CPI
#     from moshi.models.tts import (
#         DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO,
#         TTSModel, script_to_entries, ConditionAttributes,
#     )
#     from moshi.conditioners import dropout_all_conditions
#     from moshi.models.lm import LMGen as _TTSLM
#     print("Loading TTS (Kyutai)...")
#     _tts_cp   = _TTS_CPI.from_hf_repo(DEFAULT_DSM_TTS_REPO)
#     tts_model = TTSModel.from_checkpoint_info(_tts_cp, n_q=32, temp=0.6, device=DEVICE)
#     _tts_voice_path = tts_model.get_voice_path(
#         "expresso/ex03-ex01_happy_001_channel1_334s.wav"
#     )
#     _tts_cond = tts_model.make_condition_attributes([_tts_voice_path], cfg_coef=2.0)
#     print("TTS (Kyutai) ready.")

# ======================================================================
# LLM: OpenAI (modular brain — model controlled by LLM_MODEL in .env)
# ======================================================================

from openai.types.chat import ChatCompletionMessageParam
from openai_brain import get_brain  # model, pricing, logging all in one place

# ======================================================================
# Insurance flow engine
# ======================================================================

from insurance_flow_engine import InsuranceSession

current_session: "InsuranceSession | None" = None

# ======================================================================
# CLEANUP / SAFETY
# ======================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MD_RE = re.compile(r"[*#`]+")
_ITEM_RE = re.compile(r"^\s*[A-Ca-c\d][.):]\s*", re.MULTILINE)
MULTISPACE_RE = re.compile(r"\s+")

# STT cleanup replacements — add insurance-specific mishearings here
STT_REPLACEMENTS: dict[str, str] = {
    # e.g. "teen" → "13" (handle common number mishears if found in testing)
}


def _clean_llm(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _MD_RE.sub("", text)
    text = _ITEM_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    # Keep voice-friendly length
    if " ... " not in text:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(sentences[:3]).strip()

    return text


def preprocess_transcript(text: str) -> str:
    """Basic STT normalization — whitespace cleanup + registered replacements."""
    if not text:
        return ""
    cleaned = MULTISPACE_RE.sub(" ", text.strip())
    low = cleaned.lower()
    for wrong, fixed in STT_REPLACEMENTS.items():
        if wrong in low:
            cleaned = re.sub(
                rf"\b{re.escape(wrong)}\b", fixed, cleaned, flags=re.IGNORECASE
            )
    return cleaned.strip()


# llm_fallback removed — naturalization is now handled inside InsuranceSession.
# See insurance_flow_engine.py → InsuranceSession.process_turn_naturalized()


conversation: list[ChatCompletionMessageParam] = []

# ======================================================================
# Flask / WebSocket
# ======================================================================

from flask import Flask, render_template_string
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

audio_queue = queue.Queue()
msg_queue = queue.Queue()

# HTML kept same as your current version
HTML = """<!DOCTYPE html>
<html>
<head>
<title>Insurance Voice Agent</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; }
#header { background: #fff; padding: 14px 20px; border-bottom: 1px solid #ddd;
          font-size: 18px; font-weight: 600; }
#chat { flex: 1; overflow-y: auto; padding: 20px;
        display: flex; flex-direction: column; gap: 12px; }
.bubble { max-width: 72%; padding: 10px 15px; border-radius: 18px;
          font-size: 15px; line-height: 1.5; word-wrap: break-word; }
.you  { align-self: flex-end; background: #0084ff; color: #fff;
        border-bottom-right-radius: 4px; }
.ai   { align-self: flex-start; background: #fff; color: #111;
        border: 1px solid #e0e0e0; border-bottom-left-radius: 4px; }
.label { font-size: 11px; color: #999; margin-bottom: 2px; }
.you-wrap { display: flex; flex-direction: column; align-items: flex-end; }
.ai-wrap  { display: flex; flex-direction: column; align-items: flex-start; }
#statusbar { background: #fff; border-top: 1px solid #ddd; padding: 10px 20px;
             display: flex; align-items: center; gap: 12px; }
#status { font-size: 14px; color: #555; flex: 1; }
#micBtn { background: #0084ff; color: #fff; border: none; border-radius: 50%;
          width: 44px; height: 44px; font-size: 20px; cursor: pointer; }
#micBtn:disabled { background: #ccc; }
#muteBtn { background: #fff; border: 2px solid #ccc; border-radius: 50%;
           width: 44px; height: 44px; cursor: pointer; display: none;
           align-items: center; justify-content: center; padding: 0; }
#muteBtn.muted { border-color: #e53935; }
#muteBtn svg { display: block; }
#muteBtnWrap { display: none; flex-direction: column; align-items: center; gap: 3px; }
#muteLabel { font-size: 11px; color: #888; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
       background: #0084ff; margin-right: 6px; }
.dot.red { background: #e53935; }
</style>
</head>
<body>
<div id="header">🎤 Insurance Voice Agent</div>
<div id="chat"></div>
<div id="statusbar">
  <span id="status">Click &#9654; to start</span>
  <div id="muteBtnWrap">
    <button id="muteBtn" onclick="toggleMute()" title="Mute/Unmute mic">
      <svg id="micIcon" width="22" height="22" viewBox="0 0 24 24" fill="none"
           stroke="#333" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="2" width="6" height="11" rx="3"/>
        <path d="M5 10a7 7 0 0 0 14 0"/>
        <line x1="12" y1="17" x2="12" y2="21"/>
        <line x1="9" y1="21" x2="15" y2="21"/>
        <line id="muteSlash" x1="3" y1="3" x2="21" y2="21"
              stroke="#e53935" stroke-width="2.5" style="display:none"/>
      </svg>
    </button>
    <span id="muteLabel">Mute</span>
  </div>
  <button id="micBtn" onclick="startAssistant()">&#9654;</button>
</div>
<script>
const SILENCE_THRESH = 0.02, SILENCE_MS = 700, MAX_RECORD_MS = 8000;
let ws, audioCtx, analyser, stream, recording=false, silenceTimer=null,
    mediaRecorder=null, recChunks=[], muted=false;

function toggleMute() {
    muted = !muted;
    document.getElementById('muteBtn').classList.toggle('muted', muted);
    document.getElementById('muteSlash').style.display = muted ? 'block' : 'none';
    document.getElementById('muteLabel').textContent = muted ? 'Unmute' : 'Mute';
    if (stream) stream.getAudioTracks().forEach(t => t.enabled = !muted);
    if (muted && recording) stopCapture();
    else if (!muted) detectVoice();
}

let audioQ=[], isPlaying=false, speechDone=false, aiBubble=null;

function addBubble(role, text) {
    const chat=document.getElementById('chat');
    const wrap=document.createElement('div'); wrap.className=role+'-wrap';
    const lbl=document.createElement('div'); lbl.className='label';
    lbl.textContent = role==='you' ? 'You' : 'Alex';
    const bub=document.createElement('div'); bub.className='bubble '+role;
    bub.textContent=text;
    wrap.appendChild(lbl); wrap.appendChild(bub);
    chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
    return bub;
}

function setStatus(text,dot) {
    const s=document.getElementById('status');
    s.innerHTML=dot?'<span class="dot '+dot+'"></span>'+text:text;
}

async function playNext() {
    if (!audioQ.length) {
        isPlaying=false;
        if (stream) stream.getAudioTracks().forEach(t => t.enabled=true);
        if (speechDone) { setStatus('Listening...',''); if (!muted) detectVoice(); }
        return;
    }
    isPlaying=true; setStatus('Speaking...','');
    if (stream) stream.getAudioTracks().forEach(t => t.enabled=false);
    const b64=audioQ.shift();
    const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
    try {
        const buf=await audioCtx.decodeAudioData(bytes.buffer.slice(0));
        const src=audioCtx.createBufferSource();
        src.buffer=buf; src.connect(audioCtx.destination);
        src.onended=playNext; src.start();
    } catch(err) { console.error('decode',err); setTimeout(playNext,0); }
}

async function startAssistant() {
    document.getElementById('micBtn').disabled=true;
    setStatus('Requesting mic...');
    try { stream=await navigator.mediaDevices.getUserMedia({audio:true}); }
    catch(e) { setStatus('Mic error: '+e.message); return; }
    audioCtx=new AudioContext(); await audioCtx.resume();
    analyser=audioCtx.createAnalyser(); analyser.fftSize=512;
    audioCtx.createMediaStreamSource(stream).connect(analyser);

    ws=new WebSocket('ws://'+location.host+'/audio');
    ws.onopen=()=>{ setStatus('Listening...','');
        document.getElementById('muteBtnWrap').style.display='flex';
        document.getElementById('muteBtn').style.display='inline-flex';
        detectVoice(); };
    ws.onmessage=async e=>{
        const msg=JSON.parse(e.data);
        if (msg.type==='you') { aiBubble=null; audioQ=[]; isPlaying=false; speechDone=false;
            addBubble('you',msg.text); setStatus('Thinking...',''); }
        else if (msg.type==='ai') {
            if (!aiBubble) { aiBubble=addBubble('ai',msg.text); }
            else { aiBubble.textContent+=' '+msg.text; } }
        else if (msg.type==='audio') { audioQ.push(msg.audio); if (!isPlaying) playNext(); }
        else if (msg.type==='audio_end') {
            speechDone=true;
            if (!isPlaying) {
                if (stream) stream.getAudioTracks().forEach(t=>t.enabled=!muted);
                setStatus('Listening...',''); if (!muted) detectVoice(); } }
        else if (msg.type==='status') { setStatus(msg.text,''); }
    };
    ws.onclose=()=>setStatus('Disconnected. Reload page.');
}

function detectVoice() {
    const buf=new Uint8Array(analyser.fftSize);
    (function loop() {
        if (recording||muted) return;
        analyser.getByteTimeDomainData(buf);
        let s=0; for(let i=0;i<buf.length;i++) s+=Math.abs(buf[i]-128);
        if (s/buf.length/128>SILENCE_THRESH) startCapture();
        else requestAnimationFrame(loop);
    })();
}

function startCapture() {
    recording=true; recChunks=[];
    setStatus('🔴 Recording...','red');
    mediaRecorder=new MediaRecorder(stream,{mimeType:'audio/webm'});
    mediaRecorder.ondataavailable=e=>{ if(e.data.size>0) recChunks.push(e.data); };
    mediaRecorder.start(200);
    silenceTimer=setTimeout(stopCapture,SILENCE_MS);
    setTimeout(stopCapture,MAX_RECORD_MS);
    const vbuf=new Uint8Array(analyser.fftSize);
    (function checkSilence() {
        if (!recording) return;
        analyser.getByteTimeDomainData(vbuf);
        let s=0; for(let i=0;i<vbuf.length;i++) s+=Math.abs(vbuf[i]-128);
        if (s/vbuf.length/128>SILENCE_THRESH) {
            clearTimeout(silenceTimer); silenceTimer=setTimeout(stopCapture,SILENCE_MS); }
        requestAnimationFrame(checkSilence);
    })();
}

function stopCapture() {
    if (!recording) return;
    recording=false; setStatus('⏳ Processing...');
    mediaRecorder.stop();
    mediaRecorder.onstop=()=>{
        const blob=new Blob(recChunks,{type:'audio/webm'});
        blob.arrayBuffer().then(buf=>{
            if (ws.readyState===1) { ws.send(buf); ws.send('END'); }
        });
    };
}
</script>
</body>
</html>"""


# ======================================================================
# WebSocket route
# ======================================================================


@app.route("/")
def index():
    return render_template_string(HTML)


@sock.route("/audio")
def audio_ws(ws):
    global current_session
    conversation.clear()
    current_session = InsuranceSession.from_json()
    dlog(f"[Insurance] New session started — step={current_session.current_step_id}")

    # Greeting sentinel
    audio_queue.put(None)

    # Send greeting first
    try:
        while True:
            item = msg_queue.get(timeout=30)
            if item == "DONE":
                break
            ws.send(item)
    except Exception as e:
        print(f"[Greeting WS error: {e}]")
        return

    # Main loop
    while True:
        try:
            chunks = []
            while True:
                data = ws.receive()
                if data is None:
                    return
                if data == "END":
                    audio_queue.put(b"".join(chunks))
                    break
                if isinstance(data, bytes):
                    chunks.append(data)

            while True:
                item = msg_queue.get(timeout=120)
                if item == "DONE":
                    break
                ws.send(item)

        except Exception as e:
            print(f"[WS error: {e}]")
            traceback.print_exc()
            return


# ======================================================================
# Audio helpers
# ======================================================================

_MIN_AUDIO_BYTES = 800


def get_audio():
    raw = audio_queue.get()

    if raw is None:
        return None

    dlog(f"\n[Received {len(raw)} bytes]")

    if len(raw) < _MIN_AUDIO_BYTES:
        return np.array([], dtype=np.float32)

    try:
        container = av.open(io.BytesIO(raw))
        sr = container.streams.audio[0].codec_context.sample_rate or 48000
        chunks = []

        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32))

        container.close()
        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    except Exception as e:
        print(f"[PyAV failed ({e}), falling back to ffmpeg]")
        tmp_in = os.path.join(tempfile.gettempdir(), "va_in.webm")
        tmp_out = os.path.join(tempfile.gettempdir(), "va_in.wav")

        with open(tmp_in, "wb") as f:
            f.write(raw)

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_out],
            capture_output=True,
        )

        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {r.stderr.decode()[-300:]}")

        with wave.open(tmp_out, "rb") as wf:
            audio = (
                np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(
                    np.float32
                )
                / 32768.0
            )

    dlog(f"[Audio: {len(audio) / 16000:.2f}s]")
    return audio


# ======================================================================
# ---------------------------------------------------------------------------
# STT context prompt — biases Whisper toward insurance intake vocabulary
# and structured spoken answers (numbers, dates, yes/no, phone numbers).
#
# Whisper uses initial_prompt as a "preceding transcript" hint; matching
# domain vocabulary here dramatically reduces hallucination into unrelated
# words.  Keep it ≤ 224 tokens (Whisper's context window).
# ---------------------------------------------------------------------------

_STT_INSURANCE_PROMPT = (
    # Domain context — tunes the language model prior
    "This is a phone call for an insurance intake. "
    "The agent is collecting information for a vehicle insurance quote. "
    # Structured answer types the caller is likely to say
    "The caller may say: yes, no, correct, that's right, I don't have any. "
    # Insurance-specific vocabulary
    "Common words: policy, coverage, insurance, broker, quote, premium, deductible, "
    "liability, collision, comprehensive, endorsement, certificate, household, "
    "registered owner, principal driver, licensed driver, marital status, "
    "single, married, common law, divorced, widowed, "
    "vehicle, financed, leased, owned, winter tires, modification, modifications. "
    # Phone numbers spoken digit by digit
    "Phone numbers: four one six five five five zero one nine nine, "
    "nine zero five two two two three three three three. "
    # Dates and years spoken aloud
    "Dates: March twenty-sixth twenty twenty-seven, January two thousand ten, "
    "twenty-sixth of March, April first twenty twenty-six. "
    # Numbers, counts, amounts
    "Numbers: one driver, two vehicles, three, four. "
    "Amounts: one hundred sixty thousand, two hundred thousand, fifty thousand. "
    # WhatsApp / contact
    "Contact: WhatsApp number, cell number, mobile number."
)


# Transcribe — faster-whisper (Option A) or Kyutai (Option B)
# ======================================================================


def transcribe(audio: np.ndarray) -> str:
    if len(audio) < 16000:
        return ""

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01:
        return ""

    if not USE_KYUTAI_STT:
        t0 = time.time()
        segs, info = stt_model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,
            initial_prompt=_STT_INSURANCE_PROMPT,
        )
        parts = [
            s.text.strip() for s in segs if getattr(s, "no_speech_prob", 0.0) < 0.6
        ]
        text = " ".join(parts).strip()
        dlog(f"[STT {time.time() - t0:.2f}s lang={info.language}: {repr(text)}]")
        return text

    else:
        # Kyutai STT path (kept for compatibility)
        import math
        import itertools

        t0 = time.time()

        audio_t = torch.from_numpy(audio).to(DEVICE).unsqueeze(0)
        sr = 16000
        target_sr = _kyutai_mimi.sample_rate

        if sr != target_sr:
            audio_t = julius.resample_frac(audio_t, sr, target_sr)

        if audio_t.shape[-1] % _kyutai_mimi.frame_size != 0:
            pad = _kyutai_mimi.frame_size - audio_t.shape[-1] % _kyutai_mimi.frame_size
            audio_t = torch.nn.functional.pad(audio_t, (0, pad))

        n_prefix = math.ceil(_kyutai_prefix * _kyutai_mimi.frame_rate)
        n_suffix = math.ceil(_kyutai_delay * _kyutai_mimi.frame_rate)
        silence = torch.zeros(
            (1, 1, _kyutai_mimi.frame_size), dtype=torch.float32, device=DEVICE
        )

        chunks = itertools.chain(
            itertools.repeat(silence, n_prefix),
            torch.split(audio_t[:, None], _kyutai_mimi.frame_size, dim=-1),
            itertools.repeat(silence, n_suffix),
        )

        tok_acc = []
        with _kyutai_mimi.streaming(1), _kyutai_lm_gen.streaming(1):
            for chunk in chunks:
                audio_tok = _kyutai_mimi.encode(chunk)
                text_tok = _kyutai_lm_gen.step(audio_tok)
                if text_tok is not None:
                    tok_acc.append(text_tok)

        all_tok = torch.concat(tok_acc, dim=-1).cpu().view(-1)
        all_tok = all_tok[all_tok > _kyutai_pad_id]
        text = _kyutai_tok.decode(all_tok.numpy().tolist())
        dlog(f"[Kyutai STT {time.time() - t0:.2f}s: {repr(text)}]")
        return text.strip()


# ======================================================================
# TTS stream — Kokoro (Option A) or Kyutai (Option B)
# ======================================================================


def tts_stream(text: str) -> None:
    if not text.strip():
        return

    if not USE_KYUTAI_TTS:

        async def _run():
            t0, n = time.time(), 0
            async for samples, sr in tts_model.create_stream(
                text,
                voice=_voice_profile.voice_id,
                speed=_voice_profile.speed,
                lang=_voice_profile.language,
            ):
                if len(samples) == 0:
                    continue

                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(
                        (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
                    )

                msg_queue.put(
                    _json.dumps(
                        {
                            "type": "audio",
                            "audio": base64.b64encode(buf.getvalue()).decode(),
                        }
                    )
                )
                n += 1

            dlog(f'[TTS {time.time() - t0:.2f}s  {n} chunks  "{text[:40]}"]')

        try:
            asyncio.run(_run())
        except Exception:
            traceback.print_exc()

    else:
        t0 = time.time()
        pcms = []

        entries = tts_model.prepare_script([text], padding_between=1)

        def _on_frame(frame):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))

        with tts_model.mimi.streaming(1):
            tts_model.generate([entries], [_tts_cond], on_frame=_on_frame)

        sr = tts_model.mimi.sample_rate
        chunk_size = sr // 12
        all_pcm = np.concatenate(pcms) if pcms else np.array([], dtype=np.float32)

        for i in range(0, len(all_pcm), chunk_size):
            chunk = all_pcm[i : i + chunk_size]
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(
                    (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                )

            msg_queue.put(
                _json.dumps(
                    {
                        "type": "audio",
                        "audio": base64.b64encode(buf.getvalue()).decode(),
                    }
                )
            )

        dlog(f'[Kyutai TTS {time.time() - t0:.2f}s  "{text[:40]}"]')


# ======================================================================
# Pipeline loop
# ======================================================================

import re


def chunk_reply(text, first_chunk_chars=70, later_chunk_chars=130):
    """
    Voice-agent-friendly chunking:
    - First chunk is smaller so speech starts quickly
    - Later chunks are a bit bigger for efficiency
    - Avoids too many tiny TTS calls
    """

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split by sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+|(?<=:)\s+|(?<=;)\s+", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks = []
    current = ""
    limit = first_chunk_chars

    for part in parts:
        # If a sentence is too long, split by commas
        subparts = [part]
        if len(part) > later_chunk_chars:
            subparts = re.split(r"(?<=,)\s+", part)
            subparts = [s.strip() for s in subparts if s.strip()]

        for sub in subparts:
            candidate = f"{current} {sub}".strip()

            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    limit = later_chunk_chars
                current = sub

    if current:
        chunks.append(current)

    # Merge awkward tiny chunks
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < 30:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)

    return merged


SENT_RE = re.compile(r"(?<=[.!?])\s+")


def pipeline_loop():
    print("\nReady! Open http://localhost:5000\n")

    while True:
        try:
            audio = get_audio()

            # ── Greeting sentinel ───────────────────────────────────────
            if audio is None:
                if current_session is None:
                    msg_queue.put(_json.dumps({"type": "audio_end"}))
                    msg_queue.put("DONE")
                    continue

                full_reply = current_session.get_greeting()
                dlog(f"[Insurance] Greeting: {full_reply[:80]}")
                msg_queue.put(_json.dumps({"type": "ai", "text": full_reply}))
                tts_stream(full_reply)

                conversation.append({"role": "assistant", "content": full_reply})

                msg_queue.put(_json.dumps({"type": "audio_end"}))
                msg_queue.put("DONE")
                continue

            text = transcribe(audio)

            if not text or not text.strip():
                msg_queue.put(
                    _json.dumps(
                        {"type": "status", "text": "Didn't catch that — try again."}
                    )
                )
                msg_queue.put(_json.dumps({"type": "audio_end"}))
                msg_queue.put("DONE")
                continue

            cleaned_text = preprocess_transcript(text)

            dlog("=" * 72)
            dlog(f"You (raw):      {text}")
            dlog(f"You (cleaned):  {cleaned_text}")
            if current_session:
                snap = current_session.get_state_summary()
                dlog(
                    f"State before:   step={snap['current_step']} | "
                    f"drivers={snap['driver_count']} | "
                    f"vehicles={snap['vehicle_count']} | "
                    f"answers={snap['answers_collected']}"
                )

            msg_queue.put(_json.dumps({"type": "you", "text": cleaned_text}))

            # ── Insurance flow engine (flow controls what, OpenAI controls how)
            if current_session is None:
                full_reply = "I'm sorry, there was a session error. Please reload."
            else:
                full_reply = current_session.process_turn_naturalized(
                    cleaned_text,
                    brain=get_brain(),
                    history=conversation[-6:],
                ).strip()

            dlog(f"Agent reply:    {full_reply[:80]}")
            if current_session:
                snap = current_session.get_state_summary()
                dlog(
                    f"State after:    step={snap['current_step']} | "
                    f"answers={snap['answers_collected']} | "
                    f"ended={snap['ended']}"
                )
            dlog("=" * 72)

            # Stream sentence by sentence
            for chunk in chunk_reply(
                full_reply, first_chunk_chars=70, later_chunk_chars=130
            ):
                msg_queue.put(_json.dumps({"type": "ai", "text": chunk}))
                tts_stream(chunk)

            conversation.append({"role": "user", "content": cleaned_text})
            conversation.append({"role": "assistant", "content": full_reply})

            msg_queue.put(_json.dumps({"type": "audio_end"}))
            msg_queue.put("DONE")

        except KeyboardInterrupt:
            break

        except Exception:
            traceback.print_exc()
            msg_queue.put(
                _json.dumps({"type": "status", "text": "Error — please try again."})
            )
            msg_queue.put(_json.dumps({"type": "audio_end"}))
            msg_queue.put("DONE")


if __name__ == "__main__":
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    threading.Thread(target=pipeline_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
