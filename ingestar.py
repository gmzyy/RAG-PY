import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Cargar las variables de entorno
load_dotenv()

def ingestar_pdf(ruta_archivo: str):
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo '{ruta_archivo}'.")
        return

    # 2. Cargar el PDF
    print(f"Cargando el documento: {ruta_archivo}...")
    loader = PyPDFLoader(ruta_archivo)
    documentos = loader.load()
    print(f"Documento cargado (Páginas: {len(documentos)})")

    # 3. Dividir el texto en trozos (chunks)
    print("Dividiendo el texto en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Se han generado {len(chunks)} trozos de texto.")

    # 4. Generar los embeddings (LÍNEA CORREGIDA AQUÍ)
    print("Configurando el modelo de Embeddings de Google...")
    # Cambiamos nuevamente al modelo global embedding-001 por compatibilidad
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # 5. Guardar en la base de datos vectorial
    directorio_db = "./db_vectorial"
    print(f"Guardando la información en '{directorio_db}'...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=directorio_db
    )
    
    print("¡Proceso de ingesta finalizado con éxito!")

if __name__ == "__main__":
    ruta_mi_pdf = "documento-prueba.pdf"
    ingestar_pdf(ruta_mi_pdf)