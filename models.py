import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, DateTime, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pgvector.sqlalchemy import Vector

# --- DATABASE CONNECTION ---
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enable Vector Extension
with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    connection.commit()

# --- USER TABLE ---
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    ip_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- VIDEO TABLE (Updated) ---
class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    # NEW: Link video to the User who uploaded it
    user_id = Column(String, ForeignKey("users.id")) 
    filename = Column(String)
    status = Column(String, default="PENDING")
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class TranscriptChunk(Base):
    __tablename__ = "transcript_chunks"
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"))
    start_time = Column(Float)
    end_time = Column(Float)
    text = Column(Text)
    embedding = Column(Vector(768))

Base.metadata.create_all(bind=engine)