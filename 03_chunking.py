import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CLEANED_TEXT_DIR

# Carpeta donde guardaremos los chunks listos para ChromaDB
CHUNKS_DIR = "chunks_ready"
os.makedirs(CHUNKS_DIR, exist_ok=True)


def process_and_chunk():
    """Lee los TXT limpios, aplica chunking estructurado y guarda en JSON."""

    # Configuramos el divisor de LangChain
    # El orden es CLAVE: intentará cortar primero por página, luego por artículo,
    # luego por párrafos (\n\n), y así sucesivamente.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n--- PÁGINA", "ARTÍCULO", "\n\n", "\n", " "],
        chunk_size=1200,  # Límite máximo de caracteres por bloque (Ideal para OpenAI)
        chunk_overlap=200,  # Solapamiento para no perder el hilo entre bloque y bloque
        length_function=len,
        is_separator_regex=False,
    )

    archivos = [f for f in os.listdir(CLEANED_TEXT_DIR) if f.lower().endswith(".txt")]

    if not archivos:
        print("No hay archivos .txt para procesar en", CLEANED_TEXT_DIR)
        return

    total_chunks_generados = 0

    for filename in archivos:
        file_path = os.path.join(CLEANED_TEXT_DIR, filename)
        print(f"Dividiendo documento: {filename}...")

        with open(file_path, "r", encoding="utf-8") as f:
            texto_completo = f.read()

        # Generar los chunks usando LangChain
        chunks_texto = text_splitter.split_text(texto_completo)

        # Armar la estructura de datos con Metadatos
        documentos_procesados = []
        for i, chunk in enumerate(chunks_texto):
            doc = {
                "id": f"{filename}_chunk_{i}",
                "text": chunk.strip(),
                "metadata": {"source": filename, "chunk_index": i},
            }
            documentos_procesados.append(doc)

        total_chunks_generados += len(documentos_procesados)

        # Guardar el resultado en un JSON para que el equipo de DB lo levante fácil
        output_json_path = os.path.join(CHUNKS_DIR, filename.replace(".txt", ".json"))
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(documentos_procesados, json_file, ensure_ascii=False, indent=4)

        print(
            f"  -> {len(documentos_procesados)} chunks guardados en {output_json_path}"
        )

    print("\n" + "=" * 50)
    print(f"PROCESO TERMINADO. Se generaron {total_chunks_generados} chunks en total.")
    print("=" * 50)


if __name__ == "__main__":
    process_and_chunk()
