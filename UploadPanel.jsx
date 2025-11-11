import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Recorder from "./Recorder";
import TranscriptionCard from "./TranscriptionCard";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

// Big language list for transcription + translation
const LANGUAGES = [
  { code: "auto", name: "Auto Detect" },
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi" },
  { code: "ur", name: "Urdu" },
  { code: "bn", name: "Bengali" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "ml", name: "Malayalam" },
  { code: "kn", name: "Kannada" },
  { code: "mr", name: "Marathi" },
  { code: "gu", name: "Gujarati" },
  { code: "pa", name: "Punjabi" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "ru", name: "Russian" },
  { code: "ar", name: "Arabic" },
  { code: "zh", name: "Chinese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
];

export default function UploadPanel() {
  const [file, setFile] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [translated, setTranslated] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const [language, setLanguage] = useState("auto"); // transcription language
  const [targetLang, setTargetLang] = useState("en"); // translation language

  const fileRef = useRef();

  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    try {
      const res = await axios.get(`${API}/api/transcriptions`);
      setHistory(res.data);
    } catch (e) {
      console.log("History load failed:", e);
    }
  }

  function onFile(e) {
    setFile(e.target.files[0]);
  }
  async function uploadFile(user_id = null) {
  if (!file) return alert("Choose a file first");
  setLoading(true);

  const fd = new FormData();
  fd.append("file", file);
  if (language !== "auto") fd.append("language", language);
  fd.append("target", targetLang); 
  //  send translation target language
  if (user_id) fd.append("user_id", user_id);

  try {
    const res = await axios.post(`${API}/api/transcribe`, fd);
    setTranscription(res.data.original_text || res.data.text || "");
    setTranslated(res.data.translated_text || "");
    await fetchHistory();
  } catch (e) {
    console.error("Upload failed:", e);
    alert("Upload failed");
  } finally {
    setLoading(false);
  }
}

  // Translate transcribed text
  async function translate() {
    if (!transcription) return alert("No transcription to translate");
    setTranslated("Translating...");

    try {
      const res = await axios.post(`${API}/api/translate`, {
        text: transcription,
        target: targetLang,
      });
      setTranslated(res.data.translated || "Translation failed");
    } catch (e) {
      setTranslated("Error during translation");
    }
  }

  // Download transcription as PDF
  async function downloadPdf() {
    try {
      const res = await axios.post(
        `${API}/api/download_pdf`,
        { text: transcription, translated },
        { responseType: "blob" }
      );

      const url = window.URL.createObjectURL(
        new Blob([res.data], { type: "application/pdf" })
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = "transcription.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("PDF generation failed");
    }
  }

async function transcribeBlob(blob) {
  if (!blob) return alert("No audio recorded");

  setLoading(true);
  const fd = new FormData();
  fd.append("file", blob, "recording.webm");
  fd.append("language", language);
  fd.append("target", targetLang); 
  //  send translation target language

  try {
    const res = await axios.post(`${API}/api/transcribe`, fd);
    setTranscription(res.data.original_text || res.data.text || "");
    setTranslated(res.data.translated_text || "");
    await fetchHistory();
  } catch (e) {
    console.error("Transcription failed:", e);
    alert("Recording upload failed");
  } finally {
    setLoading(false);
  }
}

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* LEFT SIDE */}
      <section className="glass p-6 rounded-2xl">
        <h1 className="text-2xl font-bold h1-neon">Transcribe • Translate</h1>
        <p className="text-gray-400 mt-2">
          Upload audio or record. Supports all languages for transcription &
          translation.
        </p>

        {/* LANGUAGE SELECTORS */}
        <div className="mt-4 flex gap-4 flex-wrap items-center">
          <div>
            <label className="text-sm text-gray-300">
              Transcription Language
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="px-2 py-2 bg-gray-900 border rounded ml-2"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm text-gray-300">Translate To</label>
            <select
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
              className="px-2 py-2 bg-gray-900 border rounded ml-2"
            >
              {LANGUAGES.filter((l) => l.code !== "auto").map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* UPLOAD SECTION */}
        <div className="mt-4">
          <input ref={fileRef} type="file" accept="audio/*" onChange={onFile} />

          <div className="mt-3 flex gap-2 flex-wrap">
            {/* Upload Button */}
            <button
              onClick={() => uploadFile()}
              className="px-3 py-2 bg-indigo-600 rounded"
            >
              {loading ? "Processing..." : "Upload & Transcribe"}
            </button>

            {/* Translate Button */}
            <button onClick={translate} className="px-3 py-2 border rounded">
              Translate
            </button>

            {/* Download PDF */}
            <button
              onClick={downloadPdf}
              className="px-3 py-2 border rounded"
            >
              Download PDF
            </button>

            {/* Clear Transcription */}
            <button
              onClick={() => {
                setTranscription("");
                setTranslated("");
              }}
              className="px-3 py-2 border rounded text-red-400 hover:bg-red-700 hover:text-white transition"
            >
              Clear Transcription
            </button>

            {/* Refresh History */}
            <button
              onClick={() => fetchHistory()}
              className="px-3 py-2 border rounded text-green-400 hover:bg-green-700 hover:text-white transition"
            >
              Refresh List
            </button>

            {/* Delete All History */}
            <button
              onClick={async () => {
                if (
                  window.confirm(
                    "Are you sure you want to delete all transcriptions?"
                  )
                ) {
                  try {
                    await axios.delete(`${API}/api/clear_transcriptions`);
                    setHistory([]);
                    await fetchHistory();
                  } catch (e) {
                    alert("Failed to clear history");
                  }
                }
              }}
              className="px-3 py-2 border rounded text-red-500 hover:bg-red-800 hover:text-white transition"
            >
              Delete All History
            </button>
          </div>
        </div>

        {/* RECORDER */}
        <div className="mt-6">
          <Recorder onComplete={transcribeBlob} />
        </div>

        {/* RESULTS */}
        <div className="mt-6">
          <TranscriptionCard
            transcription={transcription}
            translated={translated}
            onDownload={downloadPdf}
          />
        </div>
      </section>

      {/* RIGHT SIDE — HISTORY */}
      <aside className="glass p-6 rounded-2xl">
        <h3 className="text-lg font-semibold">Recent</h3>

        <div className="mt-4 space-y-3 max-h-[60vh] overflow-auto">
          {history.length === 0 && (
            <div className="text-gray-400">No transcriptions yet.</div>
          )}

          {history.map((h) => (
            <div key={h.id} className="p-3 rounded border border-gray-700">
              <div className="text-xs text-gray-400">{h.filename}</div>
              <div className="mt-1 text-sm">{h.text?.slice(0, 140)}</div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
