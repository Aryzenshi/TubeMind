# 🧠 TubeMind – AI Video Analysis & Chat

**TubeMind** is an intelligent video analysis platform that allows users to *talk to their videos*. It leverages **local AI models** to transcribe audio, analyze visual content, and answer contextual questions about specific moments in a video.

---

## 🚀 Features

### 🎥 Video Ingestion
- Upload **MP4, AVI, or MOV** files
- Maximum file size: **500MB**

### 📝 Audio Transcription
- High-accuracy speech-to-text using **Faster-Whisper**

### 👀 Visual Understanding
- Powered by **LLaVA (Large Language-and-Vision Assistant)**
- Automatically detects scene changes and captures screenshots
- Understands:
  - Code displayed on screen
  - Graphs and charts
  - Visual actions not described verbally

### 💬 Contextual Chat
Ask questions such as:
- *“What error did he fix at 5:30?”*
- *“Summarize the code snippet shown.”*

Responses are generated using **both audio and visual context**.

### 🛡️ Admin Dashboard
- Role-based access control (**Admin / User**)
- IP-based rate limiting
- User ban functionality

### 📂 My Videos History
- View and manage previously uploaded videos

### 🔒 Security
- Password hashing
- JWT authentication
- Secure session management

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- JavaScript (Vanilla)
- CSS3

### Backend
- Python
- FastAPI

### AI Models
- **Speech**: faster-whisper (medium)
- **Vision**: LLaVA (via Ollama)
- **Chat LLM**: Llama 3 (via Ollama)
- **Embeddings**: nomic-embed-text

### Infrastructure
- **Database**: PostgreSQL + pgvector
- **Task Queue**: Celery & Redis
- **Containerization**: Docker (PostgreSQL & Redis)

---

## ⚙️ Installation & Setup

### Prerequisites
Ensure the following are installed:
- **Python 3.10+**
- **Docker Desktop**
- **Ollama** with required models:
  ```bash
  ollama pull llama3
  ollama pull llava
  ollama pull nomic-embed-text
  ```
- **FFmpeg** (added to system PATH)

---

### 1️⃣ Start Infrastructure

```bash
# Start PostgreSQL with vector support
docker run -d --name tm-db -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres ankane/pgvector:latest

# Start Redis
docker run -d --name tm-redis -p 6379:6379 redis
```

---

### 2️⃣ Install Dependencies

Create and activate a virtual environment:

```bash
python -m venv venv
```

**Windows**
```bash
.\venv\Scripts\activate
```

**Mac / Linux**
```bash
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---
**⚠️ Important**
After creating the venv, some required DLL files may not install automatically.
1. Download: **cuBLAS.and.cuDNN_CUDA12_win_v3**
2. Extract the files.
3. Copy/Paste (replace/add) the files into: `venv\Lib\site-packages\ctranslate2`

---
### 3️⃣ Initialize Database

Run this **one-time** command to create tables and the initial admin account:

```bash
python -c "from models import SessionLocal, User, engine, Base; \
from main import get_password_hash; import uuid; \
Base.metadata.create_all(bind=engine); \
db=SessionLocal(); \
db.add(User(id=str(uuid.uuid4()), username='admin', \
hashed_password=get_password_hash('admin123'), \
role='admin', ip_address='127.0.0.1')); \
db.commit(); print('Admin Created')"
```

**Default Login**
- **Username:** admin  
- **Password:** admin123  

---

### 4️⃣ Run the Application

You need **three terminal windows**.

**Terminal 1 – API Server**
```bash
uvicorn main:app --reload
```

**Terminal 2 – AI Worker**
```bash
celery -A worker.celery_app worker --loglevel=info --pool=solo
```

**Terminal 3 - Start Docker + Ollama**
If running ollama or docker containers gives error run these commands in terminal 3 and restart terminal 1 and 2
```bash
docker start tm-db tm-queue
```
```bash
ollama serve
```
---

## 📖 Usage

1. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
2. Log in with your credentials
3. Upload a video file
4. Wait for the status to change to **COMPLETED**
5. Click **View** to see:
   - AI-generated summary
   - Chat interface for asking questions about the video

---

## ⚖️ License

Distributed under the **MIT License**.  
See the `LICENSE` file for more information.

---

## 👨‍💻 Author

**Created by Aaryav Rastogi (Aryzenshi)**
