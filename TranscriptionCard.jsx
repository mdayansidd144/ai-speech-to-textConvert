import { motion } from "framer-motion";
export default function TranscriptionCard({ transcriptions = {}, translations = {}, onDownload }) {
  const hasTranscriptions = transcriptions && Object.keys(transcriptions).length > 0;
  if (!hasTranscriptions) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-[#061026] p-4 rounded-lg border border-gray-800 mt-6"
    >
      {/* ✅ MULTI-LANGUAGE TRANSCRIPTION */}
      <h3 className="text-lg font-semibold">Transcriptions</h3>

      {Object.entries(transcriptions).map(([lang, text]) => (
        <div key={lang} className="mt-3">
          <div className="text-xs text-indigo-400 font-semibold uppercase">{lang}</div>
          <pre className="whitespace-pre-wrap mt-1 text-sm text-gray-200">{text}</pre>
        </div>
      ))}

      {/* ✅ MULTI-LANGUAGE TRANSLATION */}
      {translations && Object.keys(translations).length > 0 && (
        <>
          <h4 className="mt-4 text-lg font-semibold">Translations</h4>

          {Object.entries(translations).map(([lang, text]) => (
            <div key={lang} className="mt-3">
              <div className="text-xs text-green-400 font-semibold uppercase">{lang}</div>
              <pre className="whitespace-pre-wrap mt-1 text-sm text-gray-200">{text}</pre>
            </div>
          ))}
        </>
      )}

      <div className="mt-4 flex gap-2">
        <button onClick={onDownload} className="px-3 py-2 bg-indigo-600 rounded">
          Download PDF
        </button>
      </div>
    </motion.div>
  );
}
