// import React, { useRef, useState } from 'react'

// export default function Recorder({ onComplete }){
//   const mediaRef = useRef(null)
//   const [recording, setRecording] = useState(false)

//   const start = async ()=>{
//     const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
//     const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
//     mediaRef.current = mr
//     const chunks = []
//     mr.ondataavailable = (e)=> chunks.push(e.data)
//     mr.onstop = ()=>{
//       const blob = new Blob(chunks, { type: 'audio/webm' })
//       onComplete && onComplete(blob)
//     }
//     mr.start()
//     setRecording(true)
//   }

//   const stop = ()=>{ mediaRef.current && mediaRef.current.stop(); setRecording(false) }

//   return (
//     <div className="flex gap-2">
//       {!recording ? <button onClick={start} className="px-3 py-2 rounded bg-red-600">Start</button> : <button onClick={stop} className="px-3 py-2 rounded bg-gray-600">Stop</button>}
//     </div>
//   )
// }
// import React, { useState, useRef, useEffect } from "react";

// export default function Recorder({ onComplete }) {
//   const [recording, setRecording] = useState(false);
//   const [level, setLevel] = useState(0);
//   const mediaRecorderRef = useRef(null);
//   const audioChunksRef = useRef([]);
//   const analyserRef = useRef(null);
//   const animationRef = useRef(null);

//   async function startRecording() {
//     const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

//     const audioContext = new AudioContext();
//     const source = audioContext.createMediaStreamSource(stream);
//     analyserRef.current = audioContext.createAnalyser();
//     analyserRef.current.fftSize = 256;

//     source.connect(analyserRef.current);

//     mediaRecorderRef.current = new MediaRecorder(stream);
//     audioChunksRef.current = [];

//     mediaRecorderRef.current.ondataavailable = (e) => {
//       if (e.data.size > 0) audioChunksRef.current.push(e.data);
//     };

//     mediaRecorderRef.current.onstop = () => {
//       const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
//       onComplete(blob);
//     };

//     mediaRecorderRef.current.start();
//     setRecording(true);

//     visualize();
//   }

//   function visualize() {
//     const bufferLength = analyserRef.current.frequencyBinCount;
//     const dataArray = new Uint8Array(bufferLength);

//     function draw() {
//       analyserRef.current.getByteFrequencyData(dataArray);
//       const avg = dataArray.reduce((a, b) => a + b) / bufferLength;
//       setLevel(avg);
//       animationRef.current = requestAnimationFrame(draw);
//     }

//     draw();
//   }

//   function stopRecording() {
//     mediaRecorderRef.current.stop();
//     setRecording(false);
//     cancelAnimationFrame(animationRef.current);
//   }

//   return (
//     <div className="mt-4">
//       <button
//         onClick={recording ? stopRecording : startRecording}
//         className="px-4 py-2 rounded bg-red-500"
//       >
//         {recording ? "Stop" : "Start Recording"}
//       </button>

//       <div className="mt-3 h-3 w-full bg-gray-700 rounded">
//         <div
//           className="h-3 bg-green-500 rounded"
//           style={{ width: `${level}%` }}
//         ></div>
//       </div>
//     </div>
//   );
// }

import React, { useState, useRef, useEffect } from "react";

export default function Recorder({ onComplete }) {
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);
  const animationRef = useRef(null);

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioContextRef.current = new AudioContext();
    const source = audioContextRef.current.createMediaStreamSource(stream);

    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 512;
    source.connect(analyserRef.current);

    mediaRecorderRef.current = new MediaRecorder(stream);
    audioChunksRef.current = [];

    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      onComplete(blob);
    };

    mediaRecorderRef.current.start();
    setRecording(true);

    visualize();
  }

  function visualize() {
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      analyserRef.current.getByteTimeDomainData(dataArray);

      // RMS volume calculation
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        let v = (dataArray[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / bufferLength);
      const volume = Math.min(100, rms * 200); // scale

      setLevel(volume);

      animationRef.current = requestAnimationFrame(draw);
    }

    draw();
  }

  function stopRecording() {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    }

    cancelAnimationFrame(animationRef.current);
    setRecording(false);

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  }

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationRef.current);
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="mt-4">
      <button
        onClick={recording ? stopRecording : startRecording}
        className="px-4 py-2 rounded bg-red-500"
      >
        {recording ? "Stop" : "Start Recording"}
      </button>

      <div className="mt-3 h-3 w-full bg-gray-700 rounded">
        <div
          className="h-3 bg-green-500 rounded"
          style={{ width: `${level}%` }}
        ></div>
      </div>
    </div>
  );
}
