from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
import os
import shutil

openai.api_key = # chave api do openai

CHROMA_PATH = "" # path do db
DATA_PATH = r"" # path dos documentos

def main():
    generate_data_store()

def generate_data_store(): 
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents(): # carrega os documentos que irao para o chunking
    loader = DirectoryLoader(DATA_PATH, glob="*.txt") # tipo do documento
    documents = loader.load()
    return documents

def split_text(documents: list[Document]): # faz o chunk do documento, imprime o resultado
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[1]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]): # faz o embedding dos chunks
    # apaga o chroma se ele existe
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # cria o chroma e faz o embedding dos chunks
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
