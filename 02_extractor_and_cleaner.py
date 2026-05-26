import fitz  # PyMuPDF
import re
import os
from config import RAW_PDF_DIR, CLEANED_TEXT_DIR


def clean_text(text):
    """Aplica expresiones regulares para limpiar ruido del texto."""
    # 1. Unir oraciones cortadas por saltos de línea (que no sean saltos de párrafo)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 2. Eliminar espacios múltiples
    text = re.sub(r"[ \t]+", " ", text)
    # 3. Eliminar caracteres nulos o corruptos
    text = text.replace("\x00", "")
    return text.strip()


def process_pdfs():
    """Lee PDFs raw, extrae, limpia y guarda como .txt."""
    archivos = [f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")]

    if not archivos:
        print("No hay PDFs para procesar. Ejecuta 01_downloader.py primero.")
        return

    for filename in archivos:
        pdf_path = os.path.join(RAW_PDF_DIR, filename)
        print(f"Procesando: {filename}...")

        doc = fitz.open(pdf_path)
        full_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text = page.get_text("text")

            cleaned_page_text = clean_text(raw_text)

            full_text += f"\n\n--- PÁGINA {page_num + 1} ---\n\n"
            full_text += cleaned_page_text

        doc.close()

        # Guardar en txt
        output_txt_path = os.path.join(
            CLEANED_TEXT_DIR, filename.replace(".pdf", ".txt")
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text.strip())

        print(f"  -> Guardado limpio en: {output_txt_path}")


if __name__ == "__main__":
    process_pdfs()
