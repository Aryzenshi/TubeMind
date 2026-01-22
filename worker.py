import os
import glob
import numpy as np
from celery import Celery
from faster_whisper import WhisperModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from models import SessionLocal, Video, TranscriptChunk
import ffmpeg
from PIL import Image
import ollama

# 1. Connect to Queue (Redis)
celery_app = Celery("tubemind", broker="redis://localhost:6379/0")

# 2. Load AI Models (The "Heavy" Start)
print("Loading Whisper Model... (Uses VRAM)")
whisper = WhisperModel("medium", device="cuda", compute_type="float16")

print("Loading Embedding Model...")
embed_model = OllamaEmbeddings(model="nomic-embed-text") 

print("Loading LLM...")
llm = OllamaLLM(model="llama3")

# --- HELPER FUNCTIONS ---

def has_screen_changed(prev_img_path, curr_img_path, threshold=5.0):
    """
    Returns True if the screen has changed significantly.
    Uses simple Mean Squared Error (MSE) on grayscale thumbnails.
    """
    if not prev_img_path: return True # Always process the first frame
    
    try:
        # 1. Open and convert to grayscale small thumbnails for speed
        img1 = Image.open(prev_img_path).convert("L").resize((64, 64))
        img2 = Image.open(curr_img_path).convert("L").resize((64, 64))
        
        # 2. Convert to numbers (arrays)
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # 3. Calculate difference (Mean Squared Error)
        mse = np.mean((arr1 - arr2) ** 2)
        
        # If difference is high, the screen changed!
        return mse > threshold
    except Exception as e:
        print(f"Error comparing frames: {e}")
        return True # Default to processing if comparison fails

def extract_frames(video_path, output_folder, interval):
    """
    Extracts frames based on a dynamic interval using FFMPEG.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"--- Extracting frames from {video_path} every {interval:.2f}s ---")
    try:
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1/interval)
            .output(f"{output_folder}/frame_%04d.jpg", qscale=2)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"Frame extraction error: {e.stderr.decode()}")

# --- MAIN TASK ---

@celery_app.task
def process_video_task(video_id: int, file_path: str):
    print(f"--- STARTED PROCESSING VIDEO {video_id} ---")
    
    db = SessionLocal()
    video = db.query(Video).filter(Video.id == video_id).first()
    
    # Update status so user knows we started
    video.status = "PROCESSING_AUDIO"
    db.commit()

    temp_frame_dir = f"temp/frames_{video_id}"
    full_text = "" # Combine audio + visual text for the final summary
    video_duration = 0.0 # Track total length

    try:
        # --- STEP 1: AUDIO PROCESSING (Fast) ---
        audio_path = file_path.replace(".mp4", ".wav")
        
        # Extract audio if it doesn't exist
        if not os.path.exists(audio_path):
            print("1. Extracting Audio...")
            ffmpeg.input(file_path).output(audio_path, ac=1, ar=16000, loglevel="quiet").run(overwrite_output=True)

        print("2. Transcribing Audio...")
        segments, _ = whisper.transcribe(audio_path)
        
        for segment in segments:
            text = segment.text.strip()
            if not text: continue
            
            # Label audio text clearly
            labeled_text = f"[AUDIO] {text}"
            full_text += labeled_text + " "
            
            # Update Duration Tracker
            if segment.end > video_duration:
                video_duration = segment.end
            
            # Save Audio Chunk to DB
            vector = embed_model.embed_query(labeled_text)
            chunk = TranscriptChunk(
                video_id=video_id,
                start_time=segment.start,
                end_time=segment.end,
                text=labeled_text,
                embedding=vector
            )
            db.add(chunk)
            
        # --- STEP 2: VISUAL PROCESSING (DYNAMIC BUDGET) ---
        video.status = "PROCESSING_VISUALS"
        db.commit()

        # === NEW LOGIC: Calculate Interval based on Duration ===
        # We want roughly 20 visual checks MAX per video to keep it under ~4 mins.
        target_frame_count = 20
        
        if video_duration > 0:
            smart_interval = max(15.0, video_duration / target_frame_count)
        else:
            smart_interval = 15.0 # Fallback default
            
        print(f"3. Analyzing Visuals (Budget: ~{target_frame_count} frames, Interval: {smart_interval:.1f}s)...")
        
        # Extract frames using the smart interval
        extract_frames(file_path, temp_frame_dir, interval=smart_interval)
        
        frame_files = sorted(glob.glob(f"{temp_frame_dir}/*.jpg"))
        last_processed_frame = None
        
        for i, frame_file in enumerate(frame_files):
            timestamp = i * smart_interval
            
            # SKIP if the screen hasn't changed (The "Data Saver")
            if not has_screen_changed(last_processed_frame, frame_file, threshold=15.0):
                print(f"   Skipping frame at {timestamp:.1f}s (No change)")
                continue
            
            # If changed, analyze it!
            print(f"   Analyzing frame at {timestamp:.1f}s...")
            try:
                response = ollama.chat(model='llava', messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this screen in 1 sentence. Focus on code, buttons, or headlines.',
                        'images': [frame_file]
                    }
                ])
                
                visual_description = response['message']['content']
                last_processed_frame = frame_file # Update "last seen" frame

                # Save Visual Chunk to DB
                visual_text = f"[VISUAL] At {timestamp:.1f}s: {visual_description}"
                full_text += visual_text + " "
                
                vector = embed_model.embed_query(visual_text)
                chunk = TranscriptChunk(
                    video_id=video_id,
                    start_time=timestamp,
                    end_time=timestamp + smart_interval,
                    text=visual_text,
                    embedding=vector
                )
                db.add(chunk)
            except Exception as e:
                print(f"   Error analyzing frame {frame_file}: {e}")

        # --- STEP 3: SUMMARY ---
        print("4. Generating Final Summary...")
        summary_prompt = f"Summarize this video tutorial based on the audio and visual actions:\n\n{full_text[:6000]}"
        video.summary = llm.invoke(summary_prompt)
        
        video.status = "COMPLETED"
        print(f"--- COMPLETED VIDEO {video_id} ---")
        
        # Cleanup files to save space
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(file_path): os.remove(file_path)
        
        if os.path.exists(temp_frame_dir):
            for f in glob.glob(f"{temp_frame_dir}/*"): os.remove(f)
            os.rmdir(temp_frame_dir)

    except Exception as e:
        print(f"!!! ERROR: {e}")
        video.status = f"FAILED: {str(e)}"
    finally:
        db.commit()
        db.close()