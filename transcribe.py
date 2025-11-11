import os
import tempfile
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_fileobj(fileobj, language=None):
    """
    Accepts an audio FileStorage object and returns transcription using
    OpenAI Whisper v3.1 (gpt-4o-audio-transcribe).
    """

    # ✅ Fix filename extension
    filename = getattr(fileobj, "filename", "recording.webm")
    ext = os.path.splitext(filename)[1].lower()

    if ext == "":
        ext = ".webm"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    fileobj.seek(0)
    tmp.write(fileobj.read())
    tmp.close()
    tmp_path = tmp.name

    try:
        resp = client.audio.transcriptions.create(
            file=open(tmp_path, "rb"),
            model="gpt-4o-audio-transcribe",
            language=language if language else None
        )
        text = resp.text

    except Exception as e:
        text = f"(transcription failed: {e})"

    # Cleanup
    try:
        os.remove(tmp_path)
    except:
        pass

    return text
