"""
PDF-to-Speech Gradio web app.

Converts a PDF to text using Unlimited-OCR, then reads it aloud using edge-tts.

Usage:
    python src/tts/app.py
    # or with Gradio hot-reload:
    gradio src/tts/app.py
"""

import asyncio
import ssl
import sys
import tempfile
from pathlib import Path

import edge_tts
import edge_tts.communicate
import gradio as gr

# edge-tts ships its own certifi CA bundle which may lack certs for
# speech.platform.bing.com on macOS. Use system-default certs instead.
edge_tts.communicate._SSL_CTX = ssl.create_default_context()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VOICES = {
    "Francisca (PT-BR, female)": "pt-BR-FranciscaNeural",
    "Antonio (PT-BR, male)": "pt-BR-AntonioNeural",
    "Raquel (PT-PT, female)": "pt-PT-RaquelNeural",
    "Duarte (PT-PT, male)": "pt-PT-DuarteNeural",
    "Aria (EN-US, female)": "en-US-AriaNeural",
    "Guy (EN-US, male)": "en-US-GuyNeural",
    "Sonia (EN-GB, female)": "en-GB-SoniaNeural",
    "Ryan (EN-GB, male)": "en-GB-RyanNeural",
}

_ocr = None


def _get_ocr():
    """Lazy-load the OCR module to avoid triggering transformers version checks at startup."""
    global _ocr
    if _ocr is None:
        import importlib
        _ocr = importlib.import_module("ocr.ocr")
    return _ocr


_model = None
_tokenizer = None


def _ensure_model():
    global _model, _tokenizer
    if _model is None:
        ocr = _get_ocr()
        _model, _tokenizer = ocr.load_model("./model")


def extract_text_from_pdf(pdf_path: str, pages: str | None = None) -> str:
    _ensure_model()
    ocr = _get_ocr()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir = tmpdir / "pages"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        import fitz

        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()

        page_indices = None
        if pages and pages.strip():
            page_indices = ocr.parse_page_range(pages, total)

        image_files = ocr.pdf_to_images(pdf_path, images_dir, pages=page_indices)

        if len(image_files) == 1:
            ocr.run_single(_model, _tokenizer, image_files[0], str(output_dir))
        else:
            ocr.run_multi(_model, _tokenizer, image_files, str(output_dir))

        md_path = output_dir / "result.md"
        if not md_path.exists():
            return ""
        md_content = md_path.read_text(encoding="utf-8")
        return ocr.md_to_txt(md_content)


async def text_to_speech(text: str, voice: str, speed: float) -> str:
    rate = f"{int((speed - 1) * 100):+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    output_path = tempfile.mktemp(suffix=".mp3")
    await communicate.save(output_path)
    return output_path


def process_pdf(pdf_file, voice_name, speed, page_range):
    if pdf_file is None:
        raise gr.Error("Please upload a PDF file.")

    yield "Extracting text from PDF (this may take a while)...", None

    text = extract_text_from_pdf(pdf_file, page_range)
    if not text.strip():
        raise gr.Error("No text could be extracted from the PDF.")

    yield text, None

    voice_id = VOICES[voice_name]
    audio_path = asyncio.run(text_to_speech(text, voice_id, speed))

    yield text, audio_path


def process_text(text, voice_name, speed):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")

    voice_id = VOICES[voice_name]
    audio_path = asyncio.run(text_to_speech(text, voice_id, speed))

    return audio_path


with gr.Blocks(title="PDF to Speech") as demo:
    gr.Markdown("# PDF to Speech\nUpload a PDF to extract text via OCR, then listen to it.")

    with gr.Tab("PDF to Speech"):
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                page_range = gr.Textbox(
                    label="Page range (optional)",
                    placeholder="e.g. 1-3,5,8-10",
                )
                voice_select = gr.Dropdown(
                    choices=list(VOICES.keys()),
                    value="Francisca (PT-BR, female)",
                    label="Voice",
                )
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="Speed",
                )
                pdf_btn = gr.Button("Extract & Speak", variant="primary")

            with gr.Column():
                text_output = gr.Textbox(label="Extracted text", lines=15, interactive=True)
                audio_output = gr.Audio(label="Audio", type="filepath")

        pdf_btn.click(
            fn=process_pdf,
            inputs=[pdf_input, voice_select, speed_slider, page_range],
            outputs=[text_output, audio_output],
        )

    with gr.Tab("Text to Speech"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text",
                    lines=10,
                    placeholder="Type or paste text here...",
                )
                voice_select_tts = gr.Dropdown(
                    choices=list(VOICES.keys()),
                    value="Francisca (PT-BR, female)",
                    label="Voice",
                )
                speed_slider_tts = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="Speed",
                )
                tts_btn = gr.Button("Speak", variant="primary")

            with gr.Column():
                audio_output_tts = gr.Audio(label="Audio", type="filepath")

        tts_btn.click(
            fn=process_text,
            inputs=[text_input, voice_select_tts, speed_slider_tts],
            outputs=[audio_output_tts],
        )


if __name__ == "__main__":
    demo.launch()
