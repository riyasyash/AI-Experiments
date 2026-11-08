import os, re
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFLoader,
)

PDF_PAGES = 18


def load_pdf(path, pages=PDF_PAGES):
    try:
        docs = PyMuPDFLoader(path).load()
    except Exception as e:
        print(
            f"  (PyMuPDFLoader failed on {os.path.basename(path)}: {e} -> PyPDFLoader fallback)"
        )
        docs = PyPDFLoader(path).load()
    return docs[:pages]


def clean_ws(text):
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()
