"""
Unlimited-OCR — local inference script.

Usage:
    # Single image
    python ocr.py -i photo.jpg

    # Multiple images (multi-page)
    python ocr.py -i page1.png page2.png page3.png

    # PDF file
    python ocr.py -i document.pdf --pdf

    # Custom output directory
    python ocr.py -i photo.jpg -o results/
"""

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

transformers.logging.set_verbosity_error()


@contextlib.contextmanager
def suppress_stdout():
    """Silence stdout during model inference (streamer token output, model print statements)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

MODEL_PATH = "./model"  # local path set by download_model.py
MODEL_ID = "baidu/Unlimited-OCR"  # fallback: download on first run
PDF_DPI = 300  # matches official infer.py; higher = better quality, larger images


def get_device(force: str | None = None) -> tuple[torch.device, torch.dtype]:
    if force:
        device = torch.device(force)
        dtype = torch.bfloat16 if force == "cuda" else torch.float32
        print(f"Device: {force} (forced)")
        return device, dtype
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda"), torch.bfloat16
    # MPS (Apple Silicon) produces incorrect outputs for this model architecture.
    # Fall back to CPU. Use -d mps to override.
    if torch.backends.mps.is_available():
        print("Device: CPU (MPS detected but not compatible with this model — use -d mps to override)")
    else:
        print("Device: CPU (inference will be slow)")
    return torch.device("cpu"), torch.float32


def patch_cuda_to_device(device: torch.device):
    """Redirect hardcoded .cuda() calls in the model source to the actual device."""
    if device.type == "cuda":
        return

    def tensor_cuda(self, *args, **kwargs):
        return self.to(device)

    def module_cuda(self, *args, **kwargs):
        return self.to(device)

    torch.Tensor.cuda = tensor_cuda  # type: ignore[method-assign]
    torch.nn.Module.cuda = module_cuda  # type: ignore[method-assign]
    print(f"Patched .cuda() -> .to({device})\n")


def patch_dtype_mismatch(model: torch.nn.Module):
    """
    Register forward pre-hooks on Conv2d layers to auto-cast inputs to the
    layer's weight dtype. Needed because the model hardcodes .to(bfloat16)
    on image tensors internally, which conflicts with non-CUDA dtypes.
    """
    def cast_input(module, args):
        x = args[0]
        if x.dtype != module.weight.dtype:
            return (x.to(module.weight.dtype),) + args[1:]

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_pre_hook(cast_input)


def patch_generate(model: torch.nn.Module):
    """
    Wrap model.generate() to always inject an attention_mask.
    Without it, pad_token_id == eos_token_id causes the model to stop
    generating after the very first token.
    """
    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if "attention_mask" not in kwargs and input_ids is not None:
            kwargs["attention_mask"] = torch.ones_like(input_ids)
        return original_generate(*args, **kwargs)

    model.generate = patched_generate


def load_model(model_path: str, force_device: str | None = None):
    print(f"Loading model from: {model_path}")
    device, dtype = get_device(force_device)
    patch_cuda_to_device(device)
    source = model_path if Path(model_path).exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        source,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=dtype,
    )
    model = model.eval().to(device)
    patch_dtype_mismatch(model)
    patch_generate(model)
    print("Model loaded.\n")
    return model, tokenizer


def md_to_txt(md_text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", md_text)       # remove images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # links -> label only
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold / italic
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)    # underscore bold/italic
    text = re.sub(r"`{3}.*?`{3}", "", text, flags=re.DOTALL)  # fenced code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)               # inline code
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # unordered list markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # ordered list markers
    text = re.sub(r"^\s*[-|]+\s*$", "", text, flags=re.MULTILINE) # table dividers
    text = re.sub(r"\|", " ", text)                        # table columns
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # horizontal rules
    text = re.sub(r"\n{3,}", "\n\n", text)                 # collapse blank lines
    return text.strip()


def save_txt(output_dir: str):
    md_path = Path(output_dir) / "result.md"
    txt_path = Path(output_dir) / "result.txt"
    if not md_path.exists():
        print(f"Warning: {md_path} not found, skipping TXT export.")
        return
    content = md_path.read_text(encoding="utf-8")
    txt_path.write_text(md_to_txt(content), encoding="utf-8")
    print(f"Saved: {txt_path}")


def parse_page_range(spec: str, total_pages: int) -> list[int]:
    """Parse a page range string like '2-5,8,11-21' into a sorted list of 0-based indices."""
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start) - 1, int(end)))
        else:
            indices.add(int(part) - 1)
    valid = sorted(i for i in indices if 0 <= i < total_pages)
    if not valid:
        print(f"Warning: page range '{spec}' matched no pages in a {total_pages}-page document.", file=sys.stderr)
    return valid


def pdf_to_images(pdf_path: str, output_dir: Path, pages: list[int] | None = None) -> list[str]:
    import fitz  # pymupdf

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = len(doc)
    page_indices = pages if pages is not None else list(range(total))
    image_paths = []

    zoom = PDF_DPI / 72  # 72 is PyMuPDF's base DPI
    mat = fitz.Matrix(zoom, zoom)
    for i in page_indices:
        pix = doc[i].get_pixmap(matrix=mat)
        img_path = str(output_dir / f"page_{i + 1:03d}.png")
        pix.save(img_path)
        image_paths.append(img_path)
        print(f"  Converted page {i + 1}/{total} -> {img_path}")

    doc.close()
    return image_paths


def cleanup_boxes(output_dir: str):
    for f in Path(output_dir).glob("result_with_boxes*.jpg"):
        f.unlink()


def run_single(model, tokenizer, image_file: str, output_dir: str, max_length: int = 32768, save_txt_file: bool = False):
    print(f"Running OCR on: {image_file}")
    with suppress_stdout():
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=image_file,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )
    cleanup_boxes(output_dir)
    if save_txt_file:
        save_txt(output_dir)
    print(f"Done. Results saved to: {output_dir}\n")


def run_multi(model, tokenizer, image_files: list[str], output_dir: str, max_length: int = 32768, save_txt_file: bool = False):
    print(f"Running multi-page OCR on {len(image_files)} image(s)...")
    with suppress_stdout():
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_files,
            output_path=output_dir,
            image_size=1024,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
    cleanup_boxes(output_dir)
    if save_txt_file:
        save_txt(output_dir)
    print(f"Done. Results saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Unlimited-OCR local inference")
    parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="Input file(s): one or more images, or a single PDF (use --pdf)",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Treat input as a PDF file and convert pages to images first",
    )
    parser.add_argument(
        "-P", "--pages", default=None,
        help="Page range to parse, e.g. '2-5,8,11-21' (PDF only, default: all pages)",
    )
    parser.add_argument(
        "--txt", action="store_true",
        help="Also save a plain-text version of the result (result.txt)",
    )
    parser.add_argument(
        "-L", "--max_length", type=int, default=32768,
        help="Max token length for generation (default: 32768). Lower values are faster on CPU.",
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="Force a specific device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "-m", "--model", default=MODEL_PATH,
        help=f"Path to local model directory (default: {MODEL_PATH})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, force_device=args.device)

    if args.pdf:
        if len(args.input) != 1:
            print("Error: --pdf expects exactly one PDF file.", file=sys.stderr)
            sys.exit(1)
        pdf_path = args.input[0]
        print(f"Converting PDF to images: {pdf_path}")
        pages_dir = output_dir / "pages" / Path(pdf_path).stem
        page_indices = None
        if args.pages:
            import fitz
            doc = fitz.open(pdf_path)
            total = len(doc)
            doc.close()
            page_indices = parse_page_range(args.pages, total)
            print(f"  Page range '{args.pages}' -> {len(page_indices)} page(s) selected")
        image_files = pdf_to_images(pdf_path, pages_dir, pages=page_indices)
        if len(image_files) == 1:
            run_single(model, tokenizer, image_files[0], str(output_dir), args.max_length, args.txt)
        else:
            run_multi(model, tokenizer, image_files, str(output_dir), args.max_length, args.txt)
    elif len(args.input) == 1:
        run_single(model, tokenizer, args.input[0], str(output_dir), args.max_length, args.txt)
    else:
        run_multi(model, tokenizer, args.input, str(output_dir), args.max_length, args.txt)


if __name__ == "__main__":
    main()
