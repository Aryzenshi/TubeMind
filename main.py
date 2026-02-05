import shutil
import os
import redis
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import text
from celery import Celery
from celery.result import AsyncResult
from jose import JWTError, jwt
from passlib.context import CryptContext
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import uuid
from datetime import datetime

# Import your database models
from models import SessionLocal, Video, TranscriptChunk, User, ChatMessage

app = FastAPI()

# --- SECURITY CONFIGURATION ---
SECRET_KEY = "supersecretkey"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
MAX_ACCOUNTS_PER_IP = 2
RATE_LIMIT_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- APP CONFIGURATION ---

# 1. Define Celery (Lightweight mode)
celery_app = Celery(
    "tubemind", 
    broker="redis://localhost:6379/0", 
    backend="redis://localhost:6379/0"
)

# 2. Mount Static Files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Create Temp Folder
os.makedirs("temp", exist_ok=True)

# 4. Database Dependency
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- AUTH HELPER FUNCTIONS ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- AUTH ENDPOINTS ---

@app.post("/register")
def register(
    request: Request, 
    user_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    # 1. IP Check (Anti-Spam)
    client_ip = request.client.host
    r = redis.Redis(host='localhost', port=6379, db=0)
    limit_key = f"reg_limit:{client_ip}"
    current_count = r.get(limit_key)
    
    if current_count and int(current_count) >= MAX_ACCOUNTS_PER_IP:
        raise HTTPException(status_code=429, detail="Too many accounts from this IP.")

    # 2. Check Username
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username taken")
    
    # 3. Create User (With IP and UUID)
    new_user = User(
        id=str(uuid.uuid4()),  # Generate Random ID
        username=user_data.username, 
        hashed_password=get_password_hash(user_data.password), 
        role="user",
        ip_address=client_ip,  # Save Location/IP
        created_at=datetime.utcnow()
    )
    db.add(new_user)
    db.commit()
    
    # 4. Update Limits
    r.incr(limit_key)
    r.expire(limit_key, RATE_LIMIT_HOURS * 3600)
    
    return {"msg": "User created successfully"}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create Token with Role info inside it
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

# --- CORE ENDPOINTS ---
@app.get("/admin/users")
def get_all_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Returns list of users for the Admin Dashboard.
    Hides passwords, shows Account Age & IP.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access only")
    
    users = db.query(User).all()
    
    user_list = []
    for u in users:
        # Calculate Account Age (e.g., "2 days")
        age = (datetime.utcnow() - u.created_at).days
        age_str = f"{age} days" if age > 0 else "Today"

        user_list.append({
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "ip": u.ip_address,
            "created_at": u.created_at.strftime("%Y-%m-%d"),
            "account_age": age_str
        })
    
    return user_list

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/my_videos")
def get_my_videos(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return all videos uploaded by the logged-in user."""
    videos = db.query(Video).filter(Video.user_id == current_user.id).order_by(Video.id.desc()).all()
    return videos

@app.delete("/clear_data")
def clear_data(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Smart Reset: 
    1. Wipes all Videos/Transcripts.
    2. Deletes all Users EXCEPT Admins.
    3. Physically deletes files in 'temp/' folder.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only Admins can reset!")

    try:
        # 1. Delete ALL Videos (Database)
        db.query(Video).delete()
        
        # 2. Delete ALL Users who are NOT admins
        # This keeps YOU and any other admin safe.
        db.query(User).filter(User.role != "admin").delete()
        
        db.commit()
        
        # 3. Clear Redis
        redis.Redis(host='localhost', port=6379, db=0).flushall()
        
        # 4. PHYSICAL CLEANUP: Wipe 'temp' folder
        # This removes any leftover videos, audio, or frame folders
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) # Delete file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) # Delete folder (frames)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    return {"status": "System Wiped! Non-admin users deleted. Temp folder cleaned."}

@app.delete("/admin/user/{user_id}")
def delete_user(user_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Admin Action: Delete a specific user and all their videos/data.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access only")
    
    # Prevent Admin from deleting themselves
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own admin account.")

    user_to_delete = db.query(User).filter(User.id == user_id).first()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Cascade delete should handle videos, but let's be safe and cleanup files
    user_videos = db.query(Video).filter(Video.id.in_(
        db.query(Video.id).join(User, User.id == user_id)
    )).all()
    
    db.delete(user_to_delete)
    db.commit()
    
    return {"status": f"User {user_to_delete.username} deleted"}

@app.delete("/video/{video_id}")
def delete_video(video_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # SECURITY: Only Owner or Admin can delete
    if video.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not your video!")
    
    db.delete(video)
    db.commit()
    
    # Try deleting file
    file_path = f"temp/{video.filename}"
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"status": "Video deleted"}

@app.post("/upload/")
async def upload_video(
    request: Request, 
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # (Keep your file size check here)
    MAX_FILE_SIZE = 500 * 1024 * 1024 
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")

    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Save with user_id
    new_video = Video(filename=file.filename, user_id=current_user.id)
    db.add(new_video)
    db.commit()
    db.refresh(new_video)
    
    # FIX: Pass file.filename as the 3rd argument
    task = celery_app.send_task("worker.process_video_task", args=[new_video.id, file_location, file.filename])
    
    return {"status": "Processing started", "video_id": new_video.id}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state == 'PENDING':
        return {"status": "Processing", "progress": "Waiting in queue or running..."}
    elif task_result.state == 'FAILURE':
        return {"status": "Failed", "error": str(task_result.result)}
    elif task_result.state == 'SUCCESS':
        return {"status": "Completed", "data": task_result.result}
    return {"status": task_result.state}

@app.get("/video/{video_id}")
def get_video_db_status(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"status": video.status, "summary": video.summary}

@app.get("/chat_history/{video_id}")
def get_chat_history(video_id: int, db: Session = Depends(get_db)):
    """Fetch previous conversation for this video."""
    return db.query(ChatMessage).filter(ChatMessage.video_id == video_id).order_by(ChatMessage.created_at).all()

@app.post("/chat/{video_id}")
def chat(video_id: int, query: str, db: Session = Depends(get_db)):
    # 1. GET GLOBAL CONTEXT (The Summary)
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    user_msg = ChatMessage(video_id=video_id, is_user=1, text=query) # SAVE USER PROMPT
    db.add(user_msg)
    db.commit()

    global_context = video.summary if video.summary else "No summary available."

    # 2. EMBED QUERY
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    query_vector = embed_model.embed_query(query)
    
    # 3. RETRIEVE & EXPAND (Fetch 30 chunks instead of 5)
    results = db.query(TranscriptChunk).filter(TranscriptChunk.video_id == video_id)\
        .order_by(TranscriptChunk.embedding.cosine_distance(query_vector))\
        .limit(30).all()

    if not results and not video.summary:
        return {"answer": "I couldn't find anything relevant in the video."}

    # 4. SORT BY TIME
    results.sort(key=lambda x: x.start_time)

    # 5. BUILD CONTEXT STRING
    transcript_text = "\n".join([f"[{r.start_time:.0f}s]: {r.text}" for r in results])
    
    llm = OllamaLLM(model="llama3")
    
    # 6. FINAL PROMPT
    prompt = f"""
    You are an expert video assistant. Use the context below to answer the user's question.
    
    --- GLOBAL VIDEO SUMMARY ---
    {global_context}
    
    --- TRANSCRIPT SEGMENTS (Chronological) ---
    {transcript_text}
    
    --- USER QUESTION ---
    {query}
    
    RULES:
    - If the answer is in the Global Summary, use it.
    - If the user asks for a specific detail, check the Transcript Segments.
    - Keep answers concise and helpful.
    """
    
    answer = llm.invoke(prompt)
    
    ai_msg = ChatMessage(video_id=video_id, is_user=0, text=answer) # SAVE AI ANSWER
    db.add(ai_msg)
    db.commit()

    return {"answer": answer, "sources": transcript_text}