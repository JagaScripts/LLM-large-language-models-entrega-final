import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DataLoader:
    """
    Clase encargada de cargar y procesar documentos PDF.
    Sigue el Principio de Responsabilidad Única (SRP) de SOLID.
    """
    
    def __init__(self, file_path: str):
        """
        Inicializa el cargador con la ruta del archivo.
        
        Args:
            file_path (str): Ruta al archivo PDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo no existe: {file_path}")
        self.file_path = file_path

    def load(self) -> list[Document]:
        """
        Carga el PDF y devuelve una lista de documentos (paginas).
        """
        print(f"📄 Cargando PDF desde: {self.file_path}")
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        print(f"✅ PDF cargado con {len(docs)} páginas.")
        return docs

    def split(self, documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
        """
        Divide los documentos en fragmentos (chunks) más pequeños.
        
        Args:
            documents (list[Document]): Lista de documentos a dividir.
            chunk_size (int): Tamaño máximo del fragmento.
            chunk_overlap (int): Solapamiento entre fragmentos.
            
        Returns:
            list[Document]: Lista de fragmentos.
        """
        print(f"✂️  Dividiendo {len(documents)} documentos en chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Se generaron {len(chunks)} fragmentos.")
        return chunks
