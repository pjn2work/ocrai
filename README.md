# ocrai — OCR & PDF-to-Speech

A local setup for running [Baidu's Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) model on your machine, with a **Gradio web app** that converts PDFs to text and reads them aloud using text-to-speech.

## Features

- **OCR** — Extract text from images and PDFs using Unlimited-OCR (3B parameter vision-language model)
- **PDF to Speech** — Upload a PDF, extract text via OCR, and listen to it with natural-sounding voices
- **Text to Speech** — Type or paste any text and convert it to audio
- **Multilingual voices** — 8 voices across Portuguese (BR & PT) and English (US & GB)
- **Gradio web UI** — Browser-based interface with audio playback, download, and speed controls

### Unlimited-OCR

Unlimited-OCR is a 3B parameter vision-language model developed by Baidu for document parsing. It extracts text from images and PDFs in a single inference pass, handling complex layouts, tables, and multi-page documents.

- Model size: ~6GB (BF16)
- License: MIT
- Source: [baidu/Unlimited-OCR on Hugging Face](https://huggingface.co/baidu/Unlimited-OCR)

### Text-to-Speech

TTS is powered by [edge-tts](https://github.com/rany2/edge-tts), which uses Microsoft Edge's free online TTS API. Requires an internet connection.

Available voices:

| Voice | Locale | Gender |
|-------|--------|--------|
| Francisca | PT-BR | Female |
| Antonio | PT-BR | Male |
| Raquel | PT-PT | Female |
| Duarte | PT-PT | Male |
| Aria | EN-US | Female |
| Guy | EN-US | Male |
| Sonia | EN-GB | Female |
| Ryan | EN-GB | Male |

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support
- 8–12GB VRAM for single image inference
- 16GB+ VRAM for multi-page or PDF inference

### Software
- Python 3.12+
- CUDA 12.9+

---

## Setup

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model

```bash
python download_model.py
```

This downloads the model from Hugging Face (~6GB) and saves it to the `./model/` directory. You only need to do this once. After that, everything runs offline.

---

## Usage

### Web App (PDF to Speech)

```bash
python src/tts/app.py
```

Then open http://127.0.0.1:7860 in your browser. The app has two tabs:

- **PDF to Speech** — Upload a PDF, optionally select a page range, pick a voice and speed, then click "Extract & Speak". The OCR model extracts text from the PDF, then edge-tts converts it to audio.
- **Text to Speech** — Type or paste text directly, pick a voice, and click "Speak".

The OCR model (~6GB) is loaded on demand when the first PDF is uploaded. The Text to Speech tab works instantly without it.

---

### CLI (OCR only)

#### Single image

```bash
python src/ocr/ocr.py -i photo.jpg
```

#### Multiple images (treated as a multi-page document)

```bash
python src/ocr/ocr.py -i page1.png page2.png page3.png
```

#### PDF file

```bash
python src/ocr/ocr.py -i document.pdf --pdf
```

The PDF is automatically converted to images page by page, then passed to the model.

#### PDF with page range

Use `-P` to select specific pages instead of processing the whole document:

```bash
# Pages 2 to 5 only
python src/ocr/ocr.py -i document.pdf --pdf -P '2-5'

# Pages 2-5, page 8, and pages 11 to 21
python src/ocr/ocr.py -i document.pdf --pdf -P '2-5,8,11-21'

# A mix of individual pages and ranges
python src/ocr/ocr.py -i document.pdf --pdf -P '1,3,5-10,15'
```

- Page numbers are **1-based** (matching your PDF viewer)
- Ranges are inclusive on both ends
- Out-of-range numbers are ignored with a warning
- Without `-P`, all pages are processed

#### Custom output directory

```bash
python src/ocr/ocr.py -i photo.jpg -o my_results/
```

Results are saved to `./output/` by default.

#### Also export plain text

```bash
python src/ocr/ocr.py -i document.pdf --pdf --txt
```

By default only `result.md` is saved. Pass `--txt` to also produce `result.txt`.

#### Limit output length (useful on CPU)

```bash
python src/ocr/ocr.py -i document.pdf --pdf -L 4096
```

Caps the number of generated tokens. Lower values finish faster. Default is `32768`.

#### Force a specific device

```bash
# Force CPU
python src/ocr/ocr.py -i photo.jpg -d cpu

# Force CUDA
python src/ocr/ocr.py -i photo.jpg -d cuda
```

By default the script auto-detects the best device. On Apple Silicon, MPS is skipped because this model architecture produces incorrect results on it — CPU is used instead.

#### All options

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-i` | `--input` | *(required)* | One or more image files, or a single PDF |
| `-o` | `--output` | `./output` | Output directory |
| | `--pdf` | off | Treat input as a PDF file |
| `-P` | `--pages` | all | Page range for PDFs, e.g. `2-5,8,11-21` |
| | `--txt` | off | Also save a plain-text `result.txt` |
| `-L` | `--max_length` | `32768` | Max tokens to generate (lower = faster on CPU) |
| `-d` | `--device` | auto | Force device: `cpu`, `mps`, or `cuda` |
| `-m` | `--model` | `./model` | Path to local model directory |

---

## Output files

After each run the following files are written to the output directory:

| File | Description |
|------|-------------|
| `result.md` | Full OCR result in Markdown format (always saved) |
| `result.txt` | Plain text with Markdown stripped (only with `--txt`) |

---

## File Structure

```
ocrai/
├── .venv/                      # Python virtual environment
├── model/                      # Downloaded model files (created by download_model.py)
├── output/                     # OCR results (created on first run)
├── src/
│   ├── chats/
│   │   ├── chat_hf.py          # Chat via Hugging Face Inference API
│   │   └── chat_ollama.py      # Chat via local Ollama
│   ├── ocr/
│   │   └── ocr.py              # OCR inference script (CLI)
│   └── tts/
│       └── app.py              # Gradio web app (PDF to Speech)
├── download_model.py           # Downloads the model from Hugging Face
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.10.0 | Deep learning runtime |
| torchvision | 0.25.0 | Image utilities for PyTorch |
| transformers | 4.57.1 | Model loading and inference |
| Pillow | 12.1.1 | Image file handling |
| matplotlib | 3.10.8 | Visualization support |
| einops | 0.8.2 | Tensor operations |
| addict | 2.4.0 | Dict utilities used by the model |
| easydict | 1.13 | Dict utilities used by the model |
| pymupdf | 1.27.2.2 | PDF to image conversion |
| psutil | 7.2.2 | System memory monitoring |
| huggingface_hub | latest | Model download from Hugging Face |
| gradio | latest | Web UI framework |
| edge-tts | latest | Text-to-speech via Microsoft Edge API |
