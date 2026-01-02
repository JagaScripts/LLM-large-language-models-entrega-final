from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.data.vector_store import VectorStoreManager

class RagEngine:
    """
    Motor RAG que orquesta la recuperación de contexto y la generación de respuestas.
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, model_name: str = "gpt-3.5-turbo"):
        """
        Inicializa el motor RAG.
        
        Args:
            vector_store_manager (VectorStoreManager): Gestor de la base de datos vectorial ya inicializado.
            model_name (str): Nombre del modelo LLM a usar.
        """
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.retriever = self.vector_store_manager.get_retriever(k=4)
        
        # Prompt template definido en la fase de prototipado
        self.template = """
        Eres un guía turístico experto en Tenerife, amigable y conocedor.
        Usa la siguiente información de contexto para responder a la pregunta del turista.
        Si no encuentras la respuesta en el contexto, di amablemente que no tienes esa información, no inventes nada.

        Contexto:
        {context}

        Pregunta del Turista: {question}

        Respuesta:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        
        # Construimos la cadena (Chain)
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        """Formatea los documentos recuperados en un solo string."""
        return "\n\n".join([d.page_content for d in docs])

    def _build_chain(self):
        """Construye la LCEL Chain."""
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, question: str) -> str:
        """
        Genera una respuesta para la pregunta dada.
        
        Args:
            question (str): Pregunta del usuario.
            
        Returns:
            str: Respuesta generada por el LLM.
        """
        print(f"🤖 Procesando pregunta: {question}")
        return self.chain.invoke(question)
