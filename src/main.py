import os
import sys
from dotenv import load_dotenv

# Asegurar que el directorio raíz está en el path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.loader import DataLoader
from src.data.vector_store import VectorStoreManager
from src.core.agent import TouristAgent

def main():
    # 1. Cargar Entorno
    load_dotenv(os.path.join(project_root, '.env'))
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY no encontrada.")
        return

    print("🌴 Iniciando Asistente Turístico de Tenerife... 🌴")
    print("--------------------------------------------------")

    # 2. Rutas
    pdf_path = os.path.join(project_root, "data", "raw", "TENERIFE.pdf")
    chroma_path = os.path.join(project_root, "chroma_db")

    # 3. Inicializar Componentes
    agent = None
    try:
        # Vector Store
        # Si no existe la DB, la creamos al vuelo. Si existe, la cargamos.
        vector_manager = VectorStoreManager(persist_directory=chroma_path)
        
        if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
            print("📦 Creando base de datos por primera vez...")
            if os.path.exists(pdf_path):
                loader = DataLoader(pdf_path)
                docs = loader.load()
                chunks = loader.split(docs)
                vector_manager.create_vector_store(chunks)
            else:
                print(f"❌ No se encuentra el PDF en: {pdf_path}")
                return

        # Agente
        agent = TouristAgent(vector_manager)
        print("✅ Sistema listo. ¡Pregúntame algo sobre Tenerife! (Escribe 'salir' para terminar)")

    except Exception as e:
        print(f"❌ Error crítico inicializando el sistema: {e}")
        return

    # 4. Bucle de Chat
    session_id = "cli_user_session"
    while True:
        try:
            user_input = input("\n👤 Tú: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("👋 ¡Hasta luego! Disfruta de Tenerife.")
                break
            
            if not user_input.strip():
                continue

            response = agent.get_response(user_input, session_id=session_id)
            print(f"\n🤖 Guía: {response}")

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error procesando respuesta: {e}")

if __name__ == "__main__":
    main()
