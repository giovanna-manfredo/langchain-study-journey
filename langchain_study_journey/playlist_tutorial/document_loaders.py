from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

loader = PyPDFLoader(Path(__file__).parent.parent / "docs" / "layout-parser-paper.pdf")

docs = loader.load()

for elemento in docs:
    print("content:" + elemento.page_content)
    print("metadata:" + str(elemento.metadata))