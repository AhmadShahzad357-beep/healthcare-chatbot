import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PDFS_DIR

import fitz  # PyMuPDF

def clean_and_convert_to_md(pdf_path: Path, output_dir: Path):
    doc = fitz.open(pdf_path)
    md_content = f"# Source: {pdf_path.name}\n\n"
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        # Cleaning
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # hyphenation
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n+', ' ', text).strip()
        
        # Markdown formatting
        lines = text.split('. ')
        formatted_text = '. '.join(lines)
        
        md_content += f"## Page {page_num}\n\n{formatted_text}\n\n"
    
    doc.close()
    
    md_filename = pdf_path.stem + ".md"
    md_path = output_dir / md_filename
    md_path.write_text(md_content, encoding="utf-8")
    print(f"✅ Converted: {pdf_path.name} -> {md_path.name}")

def main():
    output_dir = Path("data/md_from_pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found in data/pdfs/")
        return
    
    for pdf in pdf_files:
        clean_and_convert_to_md(pdf, output_dir)
    
    print(f"🎉 All PDFs converted to Markdown in {output_dir}")

if __name__ == "__main__":
    main()
