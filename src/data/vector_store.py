import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    """
    Clase para la gestión de la base de datos vectorial (ChromaDB).
    Encapsula la creación, persistencia y búsqueda de vectores.
    """
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Inicializa el gestor con el directorio de persistencia.
        """
        self.persist_directory = persist_directory
        # Usamos el modelo small que es eficiente y barato
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None

    def create_vector_store(self, documents: list[Document], reset: bool = True) -> Chroma:
        """
        Crea la base de datos vectorial a partir de documentos.
        
        Args:
            documents (list[Document]): Lista de chunks procesados.
            reset (bool): Si es True, borra la DB existente antes de crearla.
        """
        if reset and os.path.exists(self.persist_directory):
            print(f"🧹 Eliminando base de datos antigua en: {self.persist_directory}")
            shutil.rmtree(self.persist_directory)

        print("🧠 Generando embeddings y almacenando en ChromaDB...")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Base de datos vectorial lista en: {self.persist_directory}")
        return self.vector_store

    def get_retriever(self, k: int = 4):
        """
        Devuelve un objeto 'retriever' para usar en cadenas de LangChain.
        """
        if not self.vector_store:
            # Si no está cargada, intentamos cargarla del disco
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError("La base de datos vectorial no existe. Ejecuta create_vector_store primero.")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def search(self, query: str, k: int = 3) -> list[Document]:
        """
        Realiza una búsqueda de similitud directa para pruebas.
        """
        if not self.vector_store:
             self.get_retriever() # Intenta cargarla
             
        print(f"🔍 Buscando: '{query}'")
        return self.vector_store.similarity_search(query, k=k)
