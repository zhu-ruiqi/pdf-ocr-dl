# pdf-ocr-dl/ner_bert/scripts/predict_bert_ner.py

from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import pipeline
import os

# Load model and tokenizer
model_dir = "pdf-ocr-dl/ner_bert/model"
model = BertForTokenClassification.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)

# Build NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example text
text = "Barack Obama was born in Hawaii in 1961 and worked at the White House."

print("\n Input:", text)
print("\n Named Entities:")

# Predict
entities = ner_pipeline(text)
for ent in entities:
    print(f" - {ent['word']} ({ent['entity_group']})")
