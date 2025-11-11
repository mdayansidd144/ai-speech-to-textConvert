from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import os

def create_pdf(text, translated=None, title="Transcription"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    path = tmp.name

    #  Try to find a multilingual font
    possible_fonts = [
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",     
        "C:\\Windows\\Fonts\\ARIALUNI.TTF",         
        "C:\\Windows\\Fonts\\NotoSans-Regular.ttf", 
    ]
    font_found = None

    for f in possible_fonts:
        if os.path.exists(f):
            pdfmetrics.registerFont(TTFont("CustomFont", f))
            font_found = "CustomFont"
            print(f" Using font: {f}")
            break

    if not font_found:
        print("⚠️ No custom font found. Using default Helvetica (limited Unicode support).")
        font_found = "Helvetica"

    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont(font_found, 14)
    c.drawString(50, y, title)
    y -= 28
    c.setFont(font_found, 11)

    #  Draw transcription text
    for line in str(text).splitlines():
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont(font_found, 11)
        c.drawString(50, y, line[:1000])
        y -= 16

    #  Add translation section
    if translated:
        y -= 12
        c.setFont(font_found, 12)
        c.drawString(50, y, "--- Translation ---")
        y -= 20
        c.setFont(font_found, 11)
        for line in str(translated).splitlines():
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont(font_found, 11)
            c.drawString(50, y, line[:1000])
            y -= 16

    c.save()
    return path

