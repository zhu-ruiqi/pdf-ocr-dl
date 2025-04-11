import fitz  # PyMuPDF

# 提取整个PDF文档的纯文本内容，并在每页之间添加页码标识。
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += f"\n--- Page {page_num} ---\n{text}"
    doc.close()
    return full_text

# 提取主标题
def extract_titles(pdf_path):
    # tmp=0
    doc = fitz.open(pdf_path)
    titles = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font = span["font"].lower()
                    size = span["size"]
                    color = span["color"]  # RGB整数值
                    text = span["text"].strip()

                    if (
                        len(text) > 5 and
                        size > 16 and
                        any(w in font for w in ["bold", "medi", "medium"]) and
                        color == 0  # 黑色
                    ):
                        titles.append(text)
                        
                    # tmp += 1   
                    # print(span["text"], "| font:", span["font"], "| size:", span["size"], "| color:", span["color"])
                    # if tmp > 3: 
                    #     return titles
    doc.close()
    return titles

# 提取小节标题
def extract_section_titles(pdf_path):
    doc = fitz.open(pdf_path)
    section_titles = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                # 将本行所有 span 的 text 拼成一句完整的 line_text
                line_text = " ".join([span["text"].strip() for span in line.get("spans", [])])
                fonts = [span["font"].lower() for span in line["spans"]]
                sizes = [span["size"] for span in line["spans"]]

                # 判断这行是否是小标题
                if (
                    len(line_text) > 5 and
                    (
                        any(line_text.strip().startswith(str(n)) for n in range(1, 10)) or
                        line_text.lower().strip() in [
                            "introduction", "background", "model architecture", "training", "conclusion"
                        ]
                    ) and
                    any("bold" in f or "medi" in f for f in fonts) and
                    any(11 < s < 17 for s in sizes)
                ):
                    section_titles.append(line_text.strip())

    doc.close()
    return list(set(section_titles))

# 从全文中抽取关键词和摘要
def extract_keywords_and_abstract(text):
    lines = text.lower().split("\n")
    keywords = []
    abstract = ""
    for i, line in enumerate(lines):
        if "keywords" in line:
            keywords = line.replace("keywords", "").replace(":", "").strip().split(",")
        if "abstract" in line:
            abstract = "\n".join(lines[i+1:i+5])  
            break
    return keywords, abstract

# 提取段落
def extract_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    return paragraphs

if __name__ == "__main__":
    pdf_path = "pdf-ocr-dl/data/raw_pdfs/paper_sample_en.pdf"  
    text = extract_text_from_pdf(pdf_path)
    
    titles = extract_titles(pdf_path)
    section_titles = extract_section_titles(pdf_path)
    keywords, abstract = extract_keywords_and_abstract(text)
    paragraphs = extract_paragraphs(text)

    print("\nTitle: ")
    for t in titles:
        print(" -", t)

    print("\nSection Titles: ")
    for s in section_titles:
        print(" -", s)

    print("\nKeywords:", keywords)
    print("\nAbstract:\n", abstract)
    print("\nFirst Paragraph:\n", paragraphs[0])

