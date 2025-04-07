import json
import spacy
from spacy.tokens import DocBin
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convert_doccano_to_spacy(jsonl_path, output_path, nlp):
    doc_bin = DocBin()
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            example = json.loads(line)
            text = example["text"]
            ents = []
            for label in example.get("labels", []):
                start, end, ent_label = label
                span = nlp.make_doc(text).char_span(start, end, label=ent_label)
                if span:
                    ents.append(span)
            doc = nlp.make_doc(text)
            doc.ents = ents
            doc_bin.add(doc)
    doc_bin.to_disk(output_path)

if __name__ == "__main__":
    nlp = spacy.blank("en")
    # Convert train set
    train_input_path = "pdf-ocr-dl/ner_spacy/data/doccano_train.jsonl"
    train_output_path = "pdf-ocr-dl/ner_spacy/data/train.spacy"
    convert_doccano_to_spacy(train_input_path, train_output_path, nlp)

    # Convert dev set
    dev_input_path = "pdf-ocr-dl/ner_spacy/data/doccano_dev.jsonl"
    dev_output_path = "pdf-ocr-dl/ner_spacy/data/dev.spacy"
    convert_doccano_to_spacy(dev_input_path, dev_output_path, nlp)
    print("[INFO] Conversion done.")