from deep_translator import GoogleTranslator
import os
import tempfile
import time
import librosa
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment, effects
from flask import Flask, request, jsonify, send_file,Response,json

from flask_cors import CORS
from dotenv import load_dotenv
from passlib.context import CryptContext
from faster_whisper import WhisperModel
import numpy as np 
#  Local modules
from pseudodb import (
    init_pseudodb,
    add_user,
    find_user_by_email_or_username,
    add_transcription,
    list_transcriptions,
)
from pdf_gen import create_pdf
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Initialize
load_dotenv()
init_pseudodb()

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
# Use Render’s PORT environment variable if available
PORT = int(os.environ.get("PORT", 5000))
#  Load Whisper model (choose: tiny, base, small, medium, large)
print("🎧 Loading Whisper 'small' model (multilingual, accurate)...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print(" Using Faster-Whisper (CPU optimized)")


# Health check
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "backend running"})


# Signup route
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not (username and email and password):
        return jsonify({"error": "missing fields"}), 400

    if find_user_by_email_or_username(email) or find_user_by_email_or_username(username):
        return jsonify({"error": "user exists"}), 400

    hashed = pwd_ctx.hash(password)
    user = add_user(username, email, hashed)
    return jsonify({
        "message": "created",
        "user": {"id": user["id"], "username": user["username"]}
    }), 201


#  Login route
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    identifier = data.get("username") or data.get("email")
    password = data.get("password")

    user = find_user_by_email_or_username(identifier)
    if not user or not pwd_ctx.verify(password, user["password_hash"]):
        return jsonify({"error": "invalid credentials"}), 401

    return jsonify({
        "message": "ok",
        "user": {"id": user["id"], "username": user["username"]}
    })

def transcribe_long_audio(file_path, language=None, chunk_length_s=45):
    """
    Improved long-audio transcription:
    - Adds overlap between chunks
    - Reduces background noise
    - Produces smoother, clearer text
    """
    import tempfile, soundfile as sf
    print(f"Starting transcription for {file_path}")

    # Load audio
    y, sr = librosa.load(file_path, sr=16000, mono=True, res_type="kaiser_fast")
    if len(y) == 0:
        raise ValueError("Empty or invalid audio file")

    # Optional noise reduction
    try:
        y = nr.reduce_noise(y=y, sr=sr)
    except Exception as e:
        print("Noise reduction skipped:", e)

    total_duration = librosa.get_duration(y=y, sr=sr)
    print(f"Total duration: {total_duration:.1f}s")

    # Process in chunks with small overlap
    overlap = 3  # seconds
    texts = []

    for start in np.arange(0, total_duration, chunk_length_s - overlap):
        end = min(start + chunk_length_s, total_duration)
        chunk = y[int(start * sr): int(end * sr)]

        tmp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_chunk.name, chunk, sr)

        try:
            segments, _ = whisper_model.transcribe(
                tmp_chunk.name,
                language=language or None,
                beam_size=5,
                best_of=5,
                vad_filter=True
            )
            chunk_text = " ".join(s.text.strip() for s in segments)
            if chunk_text:
                texts.append(chunk_text)
            print(f"Chunk {start:.0f}-{end:.0f}s done")
        except Exception as e:
            print(f"Chunk {start:.0f}-{end:.0f}s failed:", e)
        finally:
            tmp_chunk.close()
            os.remove(tmp_chunk.name)

    full_text = " ".join(texts).strip()
    print(" Long transcription complete")
    return full_text

@app.route("/api/transcribe", methods=["POST"])
def transcribe_route():
    print("FILE RECEIVED:", request.files.keys(), "FORM:", request.form)
    fileobj = request.files.get("file")
    language = request.form.get("language")
    user_id = request.form.get("user_id")

    if not fileobj:
        return jsonify({"error": "no file uploaded"}), 400

    filename = fileobj.filename or "recording.webm"
    ext = os.path.splitext(filename)[1] or ".webm"

    save_path = os.path.join(UPLOAD_DIR, filename)
    fileobj.save(save_path)
    tmp_path = save_path

    try:
        print(" Preprocessing audio...")
        y, sr = librosa.load(tmp_path, sr=16000, mono=True, res_type="kaiser_fast")
        if len(y) == 0:
            return jsonify({"error": "Empty or invalid audio"}), 400

        # Trim silence, normalize
        y, _ = librosa.effects.trim(y, top_db=25)
        y = y / (np.max(np.abs(y)) + 1e-6)

        # Save preprocessed file
        clean_path = tmp_path.replace(ext, "_clean.wav")
        sf.write(clean_path, y.astype(np.float32), sr)

        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Duration: {duration:.2f}s")

        # Auto-detect language once
        if language == "auto" or not language:
            _, info = whisper_model.transcribe(clean_path, beam_size=1)
            language = info.language or "en"
            print(f"Detected language: {language}")

        #  Chunking setup
        CHUNK_LEN = 60  # seconds
        texts = []
        print(" Transcribing in chunks...")

        for start in np.arange(0, duration, CHUNK_LEN):
            end = min(start + CHUNK_LEN, duration)
            start_samp = int(start * sr)
            end_samp = int(end * sr)
            chunk = y[start_samp:end_samp]

            temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_chunk.name, chunk, sr)

            try:
                segments, _ = whisper_model.transcribe(
                    temp_chunk.name,
                    language=language,
                    beam_size=5
                )
                chunk_text = " ".join([s.text.strip() for s in segments])
                if chunk_text:
                    texts.append(chunk_text)
                    print(f"Chunk {start:.0f}-{end:.0f}s done.")
            except Exception as e:
                print(f" Chunk {start:.0f}-{end:.0f}s failed:", e)
            finally:
                temp_chunk.close()
                os.remove(temp_chunk.name)

        full_text = " ".join(texts).strip()
        print("Long transcription complete.")

        if not full_text:
            return jsonify({"error": "No speech detected"}), 400

        # Save to pseudo-DB
        entry = add_transcription(
            user_id=int(user_id) if user_id else None,
            filename=filename,
            text=full_text,
            language=language,
        )

        return jsonify({
            "id": entry["id"],
            "text": full_text,
            "detected_language": language,
        })

    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        # save_path.close()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(clean_path):
                os.remove(clean_path)
        except Exception as e:
            print("Cleanup failed:", e)

@app.route("/api/translate", methods=["POST"])
def translate_route():
    data = request.json or {}
    text = data.get("text", "").strip()
    target = data.get("target", "en").lower()

    if not text:
        return jsonify({"translated": "", "error": "No text provided"}), 400

    try:
        print(f"Translating text → {target.upper()} using Deep Translator...")

        # Split large text into chunks to handle API limits safely (4k chars per chunk)
        chunk_size = 4000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        translated_parts = []
        for idx, chunk in enumerate(chunks, 1):
            chunk = chunk.strip()
            if not chunk:
                continue

            try:
                translated = GoogleTranslator(source="auto", target=target).translate(chunk)
                translated_parts.append(translated)
                print(f"Chunk {idx}/{len(chunks)} translated successfully.")
            except Exception as chunk_err:
                print(f"Failed to translate chunk {idx}: {chunk_err}")
                translated_parts.append("[Translation failed for part]")

        translated_text = " ".join(translated_parts).strip()

        # Basic sanity check — if still empty, fallback message
        if not translated_text:
            translated_text = "[No translation produced]"

        print("Translation complete.")
        return jsonify({"translated": translated_text})

    except Exception as e:
        print("Translation error:", e)
        return jsonify({"translated": "", "error": str(e)}), 500

#Download PDF
@app.route("/api/download_pdf", methods=["POST"])
def download_pdf():
    data = request.json or {}
    text = data.get("text", "")
    translated = data.get("translated", "")

    content = text or ""
    if translated:
        content += "\n\n--- Translation ---\n\n" + translated

    path = create_pdf(content, None)
    return send_file(path, as_attachment=True, download_name="transcription.pdf")

# List transcriptions
@app.route("/api/transcriptions", methods=["GET"])
def transcriptions_route():
    items = list_transcriptions()
    return jsonify(items)

# Clear transcriptions
@app.route("/api/clear_transcriptions", methods=["DELETE"])
def clear_transcriptions():
    try:
        pseudodb_path = os.path.join(os.path.dirname(__file__), "pseudodb.json")
        if os.path.exists(pseudodb_path):
            os.remove(pseudodb_path)
            print(" pseudodb.json deleted.")
        else:
            print(" pseudodb.json not found.")
        init_pseudodb()
        print(" Empty pseudodb recreated.")
        return jsonify({"message": "All transcriptions cleared."}), 200
    except Exception as e:
        print("Error clearing history:", e)
        return jsonify({"error": str(e)}), 500

#  Run app
if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 500* 1024 * 1024  # 100 MB limit
    app.run(host="0.0.0.0", port=5000, debug=True)
