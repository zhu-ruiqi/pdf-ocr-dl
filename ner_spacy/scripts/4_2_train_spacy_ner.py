import subprocess
from pathlib import Path

def train_spacy_model():
    ner_root = Path(__file__).resolve().parent.parent

    config_path = ner_root / "config.cfg"
    train_data_path = ner_root / "data" / "train.spacy"
    dev_data_path = ner_root / "data" / "dev.spacy"
    output_dir = ner_root / "model"

    cmd = [
        "python", "-m", "spacy", "train", str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(train_data_path),
        "--paths.dev", str(dev_data_path)
    ]

    print("[INFO] Running spaCy training...")
    subprocess.run(cmd, check=True)
    print(f"[INFO] Model saved to: {output_dir}")

if __name__ == "__main__":
    train_spacy_model()



