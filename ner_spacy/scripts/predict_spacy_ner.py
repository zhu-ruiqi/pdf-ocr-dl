import spacy

def load_model(model_dir):
    print(f"[INFO] Loading model from: {model_dir}")
    return spacy.load(model_dir)

def predict_entities(nlp, text):
    doc = nlp(text)
    print("\n📍 Named Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

if __name__ == "__main__":
    model_path = "pdf-ocr-dl/ner_spacy/model/model-best"
    nlp = load_model(model_path)

    # 🔁 测试样例
    sample_text = "Elon Musk founded SpaceX in 2002 and lives in California."
    print(f"\n📝 Input: {sample_text}")
    predict_entities(nlp, sample_text)
