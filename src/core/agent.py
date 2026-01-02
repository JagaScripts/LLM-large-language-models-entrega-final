from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.data.vector_store import VectorStoreManager

class TouristAgent:
    """
    Agente turístico con memoria conversacional y RAG.
    """

    def __init__(self, vector_store_manager: VectorStoreManager, model_name: str = "gpt-3.5-turbo"):
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.retriever = self.vector_store_manager.get_retriever(k=4)
        
        # Historial de chats en memoria (simple para demo)
        self.store = {}
        
        self.prompt = self._create_prompt()
        self.chain = self._build_chain()
        self.conversational_chain = self._build_history_chain()

    def _create_prompt(self):
        template = """
        Eres un guía turístico experto en Tenerife, amigable y conocedor.
        Usa la siguiente información de contexto para responder a la pregunta del turista.
        Si no encuentras la respuesta en el contexto, di amablemente que no tienes esa información, no inventes nada.

        Contexto:
        {context}

        Historial de Chat:
        {chat_history}

        Pregunta del Turista: {question}

        Respuesta:
        """
        # Usamos MessagesPlaceholder si quisieramos usar lista de mensajes, 
        # pero aquí formatearemos el historial como string manualmente en get_session_history si fuera necesario,
        # o confiaremos en que LangChain inyecte el historial en 'chat_history'.
        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def _build_chain(self):
        # Cadena base con pipes
        chain = (
            {
                "context": itemgetter("question") | self.retriever | self._format_docs,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _build_history_chain(self):
        # Encapulamos la cadena base con gestión de historial
        return RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

    def get_response(self, question: str, session_id: str = "default") -> str:
        """
        Obtiene respuesta manteniendo el historial de la sesión.
        """
        print(f"🤖 Agente procesando: {question} (Sesión: {session_id})")
        return self.conversational_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
