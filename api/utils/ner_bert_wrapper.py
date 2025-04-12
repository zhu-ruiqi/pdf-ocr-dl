# api/utils/ner_bert_wrapper.py

import subprocess
import json
import os

def run_bert_ner_in_subprocess(text: str) -> dict:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    input_path = os.path.join(temp_dir, "ner_input.txt")
    output_path = os.path.join(temp_dir, "ner_output.json")

    with open(input_path, "w", encoding="utf-8") as f:
        f.write(text)

    command = [
        "conda", "run", "-n", "bert-env", "python",
        "pdf-ocr-dl/ner_bert/scripts/run_bert_ner.py",
        "--input", input_path,
        "--output", output_path
    ]

    subprocess.run(command, check=True)

    with open(output_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result
