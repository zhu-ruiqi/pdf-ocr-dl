接口路径 功能描述 要上传的内容
/text/extract 提取 PDF 中的纯文本内容（无 OCR） 📄 文本型 PDF（非扫描图）
/ocr/paddle 使用 PaddleOCR 提取图片文字 📄 PDF （含扫描图、截图等）
/ocr/tesseract 使用 Tesseract OCR 提取文字 📄 PDF
/layout/analyze 使用 PaddleOCR 版面布局分析 📄 图文混排的 PDF（如报告、书籍）
/table/extract 使用 Camelot 提取表格数据 📄 包含结构化表格的 PDF
/images/extract 提取 PDF 内嵌图像 📄 PDF （含嵌入图片）
/table/detect-yolo 使用 YOLOv5 检测图像中表格区域 🖼️ 上传图片（JPG/PNG）
/ner/bert 使用 BERT 模型 做命名实体识别（NER） 📝 纯文本文件或 JSON（实体抽取用）

/text/extract 提取 PDF 文字，速度快但对扫描图无效
/ocr/paddle & /ocr/tesseract 都是 OCR，区别在于底层引擎不同（PaddleOCR 更强）
/layout/analyze 会输出段落、图片、标题区域的结构信息
/ner/bert 你训练的 BERT 模型接口，文本实体识别（如人名、机构等）

haha
huhu
