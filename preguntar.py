import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar variables de entorno
load_dotenv()

def consultar_documento(pregunta: str):
    """
    Ejecuta una consulta contra el documento utilizando RAG con modelo Gemini.
    """
    # 2. MODELO DE EMBEDDINGS (Debe ser el mismo que usaste en ingestar.py)
    # Usamos el modelo global compatible embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")    
    
    # 3. Conexión a la base de datos
    directorio_db = "./db_vectorial"
    
    if not os.path.exists(directorio_db):
        print(f"El directorio '{directorio_db}' no existe. Ejecuta 'ingestar.py' primero.")
        return

    print("Conectando con la Base de Datos Vectorial...")
    vectorstore = Chroma(
        persist_directory=directorio_db,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever()

    # 4. MODELO DE LENGUAJE (Actualizado a Gemini 3 Flash)
    print("Cargando el modelo Gemini 3...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Actualizado para máxima potencia y velocidad
        temperature=0 
    )

    # 5. El Prompt (Las instrucciones para la IA)
    prompt_template = """
    Eres un asistente corporativo experto de Staicka. 
    Usa estrictamente los siguientes fragmentos de contexto para responder la pregunta.
    Si la respuesta no está en el contexto, di: "La información no se encuentra en el documento."
    
    Contexto:
    {context}

    Pregunta: {question}

    Respuesta:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # 6. Función para limpiar los documentos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. La Cadena (Chain) de procesamiento
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # 8. Ejecución
    print(f"\nGenerando respuesta para: '{pregunta}'...")
    respuesta = qa_chain.invoke(pregunta)
    
    print("\n" + "="*50)
    print("✨ RESPUESTA DE TU IA:")
    print(respuesta)
    print("="*50)

if __name__ == "__main__":
    mi_pregunta = "¿Cuáles son los puntos más importantes expuestos en el documento?"
    consultar_documento(mi_pregunta)