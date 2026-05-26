import os
from config import CLEANED_TEXT_DIR


def simple_chunking(text, chunk_size=1000, overlap=200):
    """
    Función de prueba para entender el concepto de solapamiento.
    Más adelante, reemplazaremos esto por LangChain (RecursiveCharacterTextSplitter).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    print("Módulo de chunking preparado.")
    print(
        "Próximo paso: Leer los archivos .txt, aplicar LangChain para dividir el texto y agregar metadatos."
    )
