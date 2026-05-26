# config.example.py (ESTE SÍ SE SUBE A GIT)
import os

AWS_ACCESS_KEY = "TU_ACCESS_KEY_AQUI"
AWS_SECRET_KEY = "TU_SECRET_KEY_AQUI"
BUCKET_NAME = "anyoneai-datasets"
PREFIX = "queplan_insurance/"

RAW_PDF_DIR = "pdfs_raw"
CLEANED_TEXT_DIR = "text_cleaned"

os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(CLEANED_TEXT_DIR, exist_ok=True)
