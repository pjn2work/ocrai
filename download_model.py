"""
Download the Unlimited-OCR model from Hugging Face to local cache.
Run this once before using ocr.py.
"""

from huggingface_hub import snapshot_download

MODEL_ID = "baidu/Unlimited-OCR"


def main():
    print(f"Downloading model: {MODEL_ID}")
    print("This may take a while depending on your connection (~6GB in BF16)...\n")

    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir="./model",
    )

    print(f"\nModel downloaded to: {local_dir}")
    print("You can now run: python ocr.py --input <image_or_pdf>")


if __name__ == "__main__":
    main()
