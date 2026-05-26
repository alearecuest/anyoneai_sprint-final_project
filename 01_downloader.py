import boto3
import os
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME, PREFIX, RAW_PDF_DIR


def download_pdfs(limit=5):
    """
    Descarga una cantidad limitada de PDFs desde S3 para hacer pruebas.
    """
    print("Conectando a AWS S3...")
    s3 = boto3.client(
        "s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
    )

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)

    if "Contents" not in response:
        print("No se encontraron archivos en el bucket.")
        return

    count = 0
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.lower().endswith(".pdf"):
            local_filename = os.path.join(RAW_PDF_DIR, os.path.basename(key))

            # Solo descargar si el archivo no existe localmente
            if not os.path.exists(local_filename):
                print(f"Descargando: {os.path.basename(key)}")
                s3.download_file(BUCKET_NAME, key, local_filename)
            else:
                print(f"El archivo {os.path.basename(key)} ya existe. Omitiendo.")

            count += 1
            if count >= limit:
                break

    print(f"\nDescarga finalizada. {count} archivos listos en '{RAW_PDF_DIR}/'.")


if __name__ == "__main__":
    download_pdfs(limit=9)
