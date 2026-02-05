import os
import glob
import numpy as np
import gc
from celery import Celery
from faster_whisper import WhisperModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from models import SessionLocal, Video, TranscriptChunk
import ffmpeg
from PIL import Image
import ollama

# 1. Connect to Queue
celery_app = Celery("tubemind", broker="redis://localhost:6379/0")

print("Worker Ready. Waiting for tasks...")

# --- HELPER FUNCTIONS ---

def has_screen_changed(prev_img_path, curr_img_path, threshold=25.0):
    """
    TUNED FOR PERFORMANCE:
    - Resizes to 32x32 (Blurs details so 'talking heads' don't trigger change)
    - Higher default threshold
    """
    if not prev_img_path: return True
    try:
        # Resize to 32x32 to ignore small movements (like hands waving)
        img1 = Image.open(prev_img_path).convert("L").resize((32, 32))
        img2 = Image.open(curr_img_path).convert("L").resize((32, 32))
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        mse = np.mean((arr1 - arr2) ** 2)
        return mse > threshold
    except Exception as e:
        print(f"Error comparing frames: {e}")
        return True

def extract_frames(video_path, output_folder, interval):
    if os.path.exists(output_folder):
        for f in glob.glob(f"{output_folder}/*"): os.remove(f)
    else:
        os.makedirs(output_folder)
        
    print(f"--- Extracting frames every {interval:.2f}s ---")
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
def process_video_task(video_id: int, file_path: str, original_filename: str = "Video"):
    print(f"--- STARTED PROCESSING VIDEO {video_id} ---")
    db = SessionLocal()
    video = db.query(Video).filter(Video.id == video_id).first()
    
    video.status = "PROCESSING_AUDIO"
    db.commit()

    temp_frame_dir = f"temp/frames_{video_id}"
    full_text = ""
    video_duration = 0.0
    
    embed_model = OllamaEmbeddings(model="nomic-embed-text") 

    try:
        # ====================================================
        # STEP 1: AUDIO
        # ====================================================
        print("1. Loading Whisper (GPU)...")
        whisper = WhisperModel("medium", device="cuda", compute_type="float16")
        
        audio_path = file_path.replace(".mp4", ".wav")
        if not os.path.exists(audio_path):
            ffmpeg.input(file_path).output(audio_path, ac=1, ar=16000, loglevel="quiet").run(overwrite_output=True)

        print("   Transcribing...")
        segments, _ = whisper.transcribe(audio_path)
        
        for segment in segments:
            text = segment.text.strip()
            if not text: continue
            labeled_text = f"[AUDIO] {text}"
            full_text += labeled_text + " "
            if segment.end > video_duration: video_duration = segment.end
            
            vector = embed_model.embed_query(labeled_text)
            db.add(TranscriptChunk(video_id=video_id, start_time=segment.start, end_time=segment.end, text=labeled_text, embedding=vector))
        
        print("   Freeing Whisper from Memory...")
        del whisper
        gc.collect() 
        
        # ====================================================
        # STEP 2: VISUALS (PERFORMANCE TUNED)
        # ====================================================
        video.status = "PROCESSING_VISUALS"
        db.commit()

        # FIX: 4.0s Interval (Good balance for slides vs speed)
        smart_interval = 4.0 
        
        print(f"2. Extracting Frames (Interval: {smart_interval:.1f}s)...")
        extract_frames(file_path, temp_frame_dir, interval=smart_interval)
        
        frame_files = sorted(glob.glob(f"{temp_frame_dir}/*.jpg"))
        print(f"   Found {len(frame_files)} frames. Starting Similarity Check...")

        last_processed_frame = None
        
        for i, frame_file in enumerate(frame_files):
            timestamp = i * smart_interval
            
            # THE FILTER: Threshold raised to 25.0 to ignore "guy talking"
            if not has_screen_changed(last_processed_frame, frame_file, threshold=25.0): 
                print(f"   [Skip] Frame at {timestamp:.1f}s (No major change)")
                continue
            
            print(f"   [Analyze] Visual change detected at {timestamp:.1f}s...")
            try:
                # Optimized Prompt for Brevity
                response = ollama.chat(model='llava', messages=[{
                    'role': 'user',
                    'content': 'Describe this image in 1 sentence. Focus on objects shown or text displayed.',
                    'images': [frame_file]
                }])
                
                visual_description = response['message']['content']
                last_processed_frame = frame_file
                
                visual_text = f"[VISUAL SCENE AT {timestamp:.1f}s] {visual_description}"
                full_text += visual_text + " "
                
                vector = embed_model.embed_query(visual_text)
                db.add(TranscriptChunk(video_id=video_id, start_time=timestamp, end_time=timestamp + smart_interval, text=visual_text, embedding=vector))
            except Exception as e:
                print(f"   !!! Visual Error on frame {i}: {e}")

        # ====================================================
        # STEP 3: SUMMARY
        # ====================================================
        print(f"3. Generating Summary...")
        
        summary_prompt = f"""
        You are an intelligent video analyzer. 
        
        Video Title: {original_filename}

        ⚠️ CRITICAL INSTRUCTION:
        - Incorporate BOTH [AUDIO] and [VISUAL SCENE] data.
        - The visual data contains timestamps. Use them to describe the flow.
        - DO NOT mention video quality, resolution, or microphone quality.
        - DO NOT act like a reviewer.

        FORMAT:
        # 📌 Video Title & Subject
        (1 sentence)

        # 🎯 Core Purpose
        (Why this video exists)

        # 💡 Key Concepts
        (Bullet points)

        # 🛠️ Timeline Walkthrough
        (Chronological breakdown)

        # 🏁 Conclusion
        (Final verdict)

        DATA:
        {full_text[:32000]}
        """
        
        llm = OllamaLLM(model="llama3")
        video.summary = llm.invoke(summary_prompt)
        video.status = "COMPLETED"
        db.commit()
        
        # Cleanup
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(file_path): os.remove(file_path)
        if os.path.exists(temp_frame_dir):
            for f in glob.glob(f"{temp_frame_dir}/*"): os.remove(f)
            os.rmdir(temp_frame_dir)

    except Exception as e:
        print(f"!!! WORKER ERROR: {e}")
        video.status = f"FAILED: {str(e)}"
    finally:
        db.commit()
        db.close()