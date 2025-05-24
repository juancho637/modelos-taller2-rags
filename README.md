# Chatbot RAG con Streamlit y LangGraph

Este proyecto implementa un **chatbot RAG** (Retrieval-Augmented Generation) usando **LangChain**, **Ollama embeddings**, **Chroma**, **Streamlit** y **LangGraph**.  
Permite cargar PDFs, indexar su contenido, invocar herramientas de recuperación de fragmentos y generar respuestas enriquecidas en una interfaz web.

👨🏽‍💻 **Estudiantes:** Alejandra Garces, Victor Silva, Juan David Garcia

---

## 📂 Estructura del proyecto

```text
taller2-rags/
├── README.md               # Documentación del proyecto
├── pyproject.toml          # Configuración de paquete y dependencias
├── run.sh                  # Script de arranque (usa UV)
├── .env.example            # Ejemplo de variables de entorno
├── .python-version         # Versión de Python (>=3.13)
├── data/
│   ├── constitucion.pdf    # PDF de la Constitución para indexar
│   └── historia.pdf        # PDF de historia para indexar
└── src/
    ├── app.py              # Interfaz Streamlit y lógica principal
    ├── graph.py            # Definición del grafo de estados con LangGraph
    └── rag.py              # Carga de PDFs y herramienta de retrieval
````

---

## 📋 Requisitos

* **Python** 3.13 o superior (definido en `.python-version`)
* **UV**: runner principal para instalar y ejecutar el proyecto
* **Dependencias** (definidas en `pyproject.toml`):

  * chromadb
  * dotenv
  * langchain
  * langchain-chroma
  * langchain-community
  * langchain-ollama
  * langgraph
  * ollama
  * pypdf
  * streamlit
  * watchdog

---

## ⚙️ Variables de entorno

Copia el archivo de ejemplo y renómbralo a `.env`:

```bash
cp .env.example .env
```

Luego edita `.env` y completa las variables con tus credenciales:

```ini
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<tu_api_key>
USER_AGENT=<tu_user_agent>
```

---

## 🚀 Instalación y ejecución con UV

1. Clona el repositorio:

   ```bash
   git clone https://github.com/juancho637/taller2-rags.git
   cd taller2-rags
   ```
2. Prepara las variables de entorno:

   ```bash
   cp .env.example .env
   # Edita .env con tus credenciales
   ```
3. (Opcional) Da permisos de ejecución al script:

   ```bash
   chmod +x run.sh
   ```
4. Ejecuta la aplicación:

   ```bash
   ./run.sh
   ```

   Esto equivale a:

   ```bash
   uv run streamlit run ./src/app.py
   ```
5. Abre tu navegador en [http://localhost:8501](http://localhost:8501) para interactuar con el chatbot.

---

## 🔐 Permisos

* Asegúrate de que `run.sh` tenga permisos de ejecución:

  ```bash
  chmod +x run.sh
  ```
* Verifica permisos de lectura/escritura en:

  * `./data/` (para cargar los PDFs que quieras indexar)
  * `./chroma_langchain_db` (para persistencia de índices)

---

## 💬 Ejemplos de preguntas

Después de indexar `constitucion.pdf` e `historia.pdf`, prueba preguntas como:

* ¿Cuál es la fecha de promulgación de la Constitución de 1991?
* ¿Qué derechos fundamentales aparecen en el Capítulo I de la Constitución?
* ¿Quiénes fueron los protagonistas del proceso independentista descrito en historia.pdf?
* ¿Qué eventos clave se mencionan para el año 1810?

¡Listo! Ahora puedes levantar tu chatbot RAG y explorar documentos PDF de forma interactiva.
