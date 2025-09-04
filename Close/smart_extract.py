from pathlib import Path 
from pdf2image import convert_from_path
import pdfplumber 
import pytesseract
import pandas as pd


def extract_single_pdf(pdf_file: Path, output_dir: Path) -> Path:
    """Extract text from a single PDF file and save to individual text file."""
    print(f"Extracting: {pdf_file.name}...")
    
    pdf_name = pdf_file.stem
    individual_file = output_dir / f"{pdf_name}.txt"
    
    pdf_content = []

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            tables = page.extract_tables()
            table_md_blocks = []
            for tbl in tables:
                df = pd.DataFrame(tbl).fillna("")
                table_md_blocks.append(df.to_markdown(index=False))

            if not text.strip():
                image = convert_from_path(pdf_file, dpi=300,
                                          first_page=i, last_page=i)[0]
                text = pytesseract.image_to_string(image, lang="eng")

            page_content = f"\n\n--- Page {i} ---\n"
            for block in table_md_blocks:
                page_content += block + "\n\n"
            page_content += text
            pdf_content.append(page_content)

    # Write individual file for this PDF
    individual_content = f"--- {pdf_file.name} ---\n" + "".join(pdf_content)
    individual_file.write_text(individual_content, encoding="utf-8")
    print(f"  → Saved to: {individual_file}")
    
    return individual_file

def check_and_extract_missing(pdf_dir: Path = None, output_dir: Path = None) -> list:
    """
    Check which PDFs need extraction and extract only missing ones.
    Returns list of extracted file paths.
    """
    if pdf_dir is None:
        pdf_dir = Path("data")
    if output_dir is None:
        output_dir = Path("extracted")
    
    output_dir.mkdir(exist_ok=True)
    
    extracted_files = []
    missing_pdfs = []
    
    # Check which PDFs need extraction
    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_name = pdf_file.stem
        expected_txt = output_dir / f"{pdf_name}.txt"
        
        if not expected_txt.exists():
            missing_pdfs.append(pdf_file)
        else:
            print(f"Already extracted: {pdf_name}.txt")
    
    # Extract only missing files
    if missing_pdfs:
        print(f"\nFound {len(missing_pdfs)} PDF(s) that need extraction:")
        for pdf_file in missing_pdfs:
            try:
                extracted_file = extract_single_pdf(pdf_file, output_dir)
                extracted_files.append(extracted_file)
                update_combined_file(output_dir)
            except Exception as e:
                print(f"Error extracting {pdf_file.name}: {e}")
    else:
        print("All PDFs already extracted!")
    
    return extracted_files

def update_combined_file(output_dir: Path) -> Path:
    """Recreate the combined documents.txt file from all individual files for LangChain."""
    combined_file = output_dir / "documents.txt"
    
    print("\nUpdating combined documents file...")
    with combined_file.open("w", encoding="utf-8") as combined:
        for txt_file in sorted(output_dir.glob("*.txt")):
            if txt_file.name == "documents.txt": 
                continue
            
            content = txt_file.read_text(encoding="utf-8")
            combined.write(content + "\n\n")
    
    print(f"Combined file updated: {combined_file}")
    return combined_file

def force_extract_all(pdf_dir: Path = None, output_dir: Path = None) -> list:
    """Force re-extraction of all PDFs (for when you want to rebuild everything)."""
    if pdf_dir is None:
        pdf_dir = Path("data")
    if output_dir is None:
        output_dir = Path("extracted")
    
    output_dir.mkdir(exist_ok=True)
    
    # Remove existing text files
    for txt_file in output_dir.glob("*.txt"):
        txt_file.unlink()
        print(f"Removed: {txt_file.name}")
    
    # Extract all PDFs
    extracted_files = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            extracted_file = extract_single_pdf(pdf_file, output_dir)
            extracted_files.append(extracted_file)
        except Exception as e:
            print(f"Error extracting {pdf_file.name}: {e}")
    
    update_combined_file(output_dir)
    
    return extracted_files

if __name__ == "__main__":
    check_and_extract_missing()
