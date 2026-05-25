# anyoneai_sprint-final_project

Pipeline base para pólizas de QuePlan.cl:

- **Extracción + limpieza** con PyMuPDF (`extract_and_clean_pdf_text`) para remover ruido común (watermarks, headers/footers repetidos, paginación e índices mal formateados).
- **Chunking + embeddings** (`chunk_text`, `SimpleHashEmbedder`, `build_chroma_payload`) para dejar los datos listos para carga en ChromaDB.

## Uso rápido

```python
from policy_data_pipeline import process_policy_pdf_for_chromadb

payload = process_policy_pdf_for_chromadb(
    pdf_path="./poliza.pdf",
    output_directory="./processed",
    source_id="queplan-poliza-001",
)
```

Esto genera:

- `processed/queplan-poliza-001_cleaned.txt`
- `processed/queplan-poliza-001_chroma_payload.json`

## Tests

```bash
python -m unittest discover -s tests -v
```
