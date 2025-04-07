import camelot
import os

import camelot
import os

output_dir = "pdf-ocr-dl/data/outputs/tables"

def extract_tables_from_pdf(pdf_path, output_dir="table_outputs", flavor="lattice"):
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return

    print(f"[INFO] Extracting tables from: {pdf_path}")
    tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)

    print(f"[INFO] Detected {tables.n} tables")

    os.makedirs(output_dir, exist_ok=True)

    for i, table in enumerate(tables):
        csv_path = os.path.join(output_dir, f"table_{i+1}.csv")
        table.df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved table {i+1} to: {csv_path}")

    return tables

if __name__ == "__main__":
    pdf_path = "pdf-ocr-dl/data/raw_pdfs/sample_table.pdf"  # 
    extract_tables_from_pdf(pdf_path, output_dir=output_dir, flavor="lattice")
  # or 'stream'
