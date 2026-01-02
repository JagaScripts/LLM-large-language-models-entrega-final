# Asistente Turístico Inteligente de Tenerife 🌴

Este proyecto implementa un sistema de **Retrieval Augmented Generation (RAG)** para actuar como guía turístico experto en Tenerife, basándose exclusivamente en la documentación oficial proporcionada.

## 📋 Características
*   **RAG (Recuperación Aumentada)**: Busca información relevante en el documento PDF `TENERIFE.pdf`.
*   **Memoria Conversacional**: Mantiene el contexto de la conversación (diálogo multiturno).
*   **Doble Interfaz**:
    *   📘 **Notebook (`notebooks/main_demo.ipynb`)**: Para demostración académica y análisis paso a paso.
    *   💻 **CLI (`src/main.py`)**: Para uso interactivo en consola.

## 🚀 Instalación y Configuración

### 1. Clonar y Entorno
```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows)
.\.venv\Scripts\activate

# Activar (Mac/Linux)
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar API Key
Crea un archivo `.env` en la raíz del proyecto con tu clave de OpenAI:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

## 🛠️ Uso

### Opción A: Ejecutar Demo (Notebook)
Abre Jupyter Lab o Notebook:
```bash
jupyter notebook notebooks/main_demo.ipynb
```
Ejecuta las celdas secuencialmente para ver la ingesta de datos y el chat de prueba.

### Opción B: Ejecutar App (CLI)
Para chatear directamente en la terminal:
```bash
python src/main.py
```
> **Nota**: La primera vez que se ejecute, el sistema procesará el PDF y creará la base de datos vectorial en la carpeta `chroma_db`. Esto puede tardar unos segundos.

## 📂 Estructura del Proyecto
```
├── notebooks/          # Notebooks de Jupyter
│   └── main_demo.ipynb # Entregable principal
├── src/                # Código fuente modular
│   ├── core/           # Lógica del RAG y Agente
│   └── data/           # Carga de PDF y Vector Store
├── requirements.txt    # Dependencias
├── .env                # Variables de entorno (NO subir al repo)
└── README.md           # Documentación
```

## 👤 Autor
Desarrollado para la Entrega Final del Master en Inteligencia Artificial.
