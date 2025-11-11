// frontend/src/components/LiveRecorder.jsx
import React, { useRef, useState, useEffect } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function LiveRecorder({ onPartial, language = "auto", chunkMs = 3000 }) {
  const [recording, setRecording] = useState(false);
  const mediaStreamRef = useRef(null);
  const recorderRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);
  const canvasRef = useRef(null);
  const meterRef = useRef(null);

  useEffect(() => {
    return () => {
      stopAll();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // MediaRecorder for chunking
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = mr;

      // Setup WebAudio for visualization
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtxRef.current.createMediaStreamSource(stream);
      const analyser = audioCtxRef.current.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      // start visualizer
      drawVisualizer();

      mr.ondataavailable = async (e) => {
        if (e.data && e.data.size > 0) {
          // send chunk to backend
          const fd = new FormData();
          fd.append("file", e.data, "chunk.webm");
          fd.append("language", language || "auto");
          try {
            const res = await axios.post(`${API}/api/transcribe_chunk`, fd, { timeout: 20000 });
            if (res?.data?.text) onPartial && onPartial(res.data.text);
          } catch (err) {
            // ignore network errors but you can log
            console.warn("chunk send failed", err);
          }
        }
      };

      mr.start(chunkMs); // collect chunks every chunkMs milliseconds
      setRecording(true);
    } catch (err) {
      alert("Microphone access is required.");
      console.error(err);
    }
  };

  const stopAll = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      try { recorderRef.current.stop(); } catch {}
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch {}
      audioCtxRef.current = null;
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setRecording(false);
  };

  const drawVisualizer = () => {
    const canvas = canvasRef.current;
    const meter = meterRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    const WIDTH = canvas.width = canvas.clientWidth;
    const HEIGHT = canvas.height = 80;
    const bufferLength = analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      rafRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "rgba(3,7,18,0.4)";
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#7c5cff";
      ctx.beginPath();

      let sliceWidth = WIDTH / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        let v = dataArray[i] / 128.0;
        let y = (v * HEIGHT) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(WIDTH, HEIGHT / 2);
      ctx.stroke();

      // VU meter
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        let val = (dataArray[i] - 128) / 128.0;
        sum += val * val;
      }
      let rms = Math.sqrt(sum / bufferLength);
      let level = Math.min(1, rms * 3); // scale

      if (meter) {
        meter.style.width = `${Math.round(level * 100)}%`;
        meter.style.background = `linear-gradient(90deg, #00e0b8, #7c5cff)`;
      }
    }
    draw();
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        {!recording ? (
          <button onClick={start} className="px-4 py-2 rounded bg-green-600">🎤 Start Live</button>
        ) : (
          <button onClick={stopAll} className="px-4 py-2 rounded bg-red-600">⏹ Stop</button>
        )}
        <div className="w-40 h-3 bg-gray-800 rounded overflow-hidden">
          <div ref={meterRef} className="h-3 w-0 rounded"></div>
        </div>
      </div>

      <div className="w-full">
        <canvas ref={canvasRef} className="w-full rounded bg-[#021026] border border-gray-800"></canvas>
      </div>

      <div className="text-xs text-gray-400">Sending {chunkMs/1000}s audio chunks for near-real-time transcription.</div>
    </div>
  );
}
