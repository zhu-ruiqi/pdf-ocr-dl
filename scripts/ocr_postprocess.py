import re
import jieba

import re
import jieba

def clean_ocr_text(raw_text, language="ch"):
    """
    Clean and enhance OCR raw text output.
    - Remove extra whitespace
    - Chinese segmentation (optional)
    - Reformat paragraphs
    """
    # Remove excessive whitespace and line breaks
    text = re.sub(r"[ \t]+", " ", raw_text)              # Convert multiple spaces/tabs to single space
    text = re.sub(r"\n{2,}", "\n", text)                 # Convert multiple newlines to single newline

    if language == "ch":
        # Split by Chinese sentence-ending punctuation
        lines = re.split(r"(。|！|\!|？|\?)", text)
        lines = [line.strip() for line in lines if line.strip()]
        
        # Rejoin with punctuation and newlines
        sentences = []
        for i in range(0, len(lines) - 1, 2):
            sentence = lines[i] + lines[i + 1]
            sentence = " ".join(jieba.cut(sentence))  # Chinese word segmentation
            sentences.append(sentence)
        text = "\n".join(sentences)

    return text
