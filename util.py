from streamlit.runtime.uploaded_file_manager import UploadedFile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing import List


def get_raw_text_from_pdf(pdf: UploadedFile):
    pdfreader = PdfReader(pdf)
    raw_text = ""
    for page in pdfreader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


# We need to split the text into smaller chunks, so that it shouldn't increase the token size
def split_raw_text(raw_text, separator="\n", chunck_size=800, chunck_overlap=200) -> List[str] | None:
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunck_size,
        chunk_overlap=chunck_overlap,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks
