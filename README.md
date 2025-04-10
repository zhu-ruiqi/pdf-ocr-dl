# pdf-ocr-dl
Deep Learning-based PDF Document Parsing and OCR Project

            用户上传 PDF
               │
               ▼
            【判断 PDF 类型】
               │
    ┌───────────────┬───────────────┐
    │               │               │
    │               │               │    
【纯文本 PDF】   【扫描 PDF】   【混合结构 PDF】
    │               │               │
    │               │               │
文本提取       图像转文本（OCR）  ➤ Layout 分析
+ 标题关键词提取    + 去噪/分词         │
+ 表格解析（Camelot）  + NER           ├── 文本块 ➝ 【纯文本 PDF】
+ NER                           ├── 表格块 ➝ Camelot / YOLO
                                └── 图片块 ➝ 图片提取 ➝ 分类
              
               ▼
     【聚合所有信息生成结果】
               │
               ▼
        Web界面展示 & 下载