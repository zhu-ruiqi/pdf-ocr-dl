import argparse
import json
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import numpy as np

def load_bert_model():
    # model_path = "ner_bert/model"
    model_path = "pdf-ocr-dl/ner_bert/model"
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def predict_entities(text: str):
    ner_pipeline = load_bert_model()
    return ner_pipeline(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    # predictions = predict_entities(text)

    # with open(args.output, "w", encoding="utf-8") as f:
    #     json.dump(predictions, f, ensure_ascii=False, indent=2)

    predictions = predict_entities(text)

    # 解决 float32 报错问题
    def convert(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2, default=convert)

    print(f"[✅ DONE] Output saved to {args.output}")

if __name__ == "__main__":
    main()
