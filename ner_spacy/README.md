# NER with spaCy

## 1. Convert annotated data
```bash
python scripts/convert_doccano.py
```

## 2. Train model
```bash
python -m spacy train config.cfg --output ./model --paths.train ./data/train.spacy --paths.dev ./data/dev.spacy
```

## 3. Inference
```bash
python -m spacy evaluate ./model/model-best ./data/dev.spacy