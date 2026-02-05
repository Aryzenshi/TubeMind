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

# 2. Load AI Models
print("Loading Whisper Model...")
whisper = WhisperModel("medium", device="cuda", compute_type="float16")

print("Loading Embedding Model...")
embed_model = OllamaEmbeddings(model="nomic-embed-text") 

print("Loading LLM...")
llm = OllamaLLM(model="llama3")

# --- HELPER FUNCTIONS ---

def has_screen_changed(prev_img_path, curr_img_path, threshold=5.0):
    if not prev_img_path: return True
    try:
        img1 = Image.open(prev_img_path).convert("L").resize((64, 64))
        img2 = Image.open(curr_img_path).convert("L").resize((64, 64))
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        mse = np.mean((arr1 - arr2) ** 2)
        return mse > threshold
    except Exception as e:
        print(f"Error comparing frames: {e}")
        return True

def extract_frames(video_path, output_folder, interval):
    if not os.path.exists(output_folder):
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

    try:
        # --- STEP 1: AUDIO PROCESSING ---
        audio_path = file_path.replace(".mp4", ".wav")
        if not os.path.exists(audio_path):
            ffmpeg.input(file_path).output(audio_path, ac=1, ar=16000, loglevel="quiet").run(overwrite_output=True)

        segments, _ = whisper.transcribe(audio_path)
        for segment in segments:
            text = segment.text.strip()
            if not text: continue
            labeled_text = f"[AUDIO] {text}"
            full_text += labeled_text + " "
            if segment.end > video_duration: video_duration = segment.end
            
            vector = embed_model.embed_query(labeled_text)
            db.add(TranscriptChunk(video_id=video_id, start_time=segment.start, end_time=segment.end, text=labeled_text, embedding=vector))
            
        # --- STEP 2: VISUAL PROCESSING (GENERAL CONTEXT) ---
        video.status = "PROCESSING_VISUALS"
        db.commit()

        target_frame_count = 20
        smart_interval = max(15.0, video_duration / target_frame_count) if video_duration > 0 else 15.0
        
        extract_frames(file_path, temp_frame_dir, interval=smart_interval)
        frame_files = sorted(glob.glob(f"{temp_frame_dir}/*.jpg"))
        last_processed_frame = None
        
        for i, frame_file in enumerate(frame_files):
            timestamp = i * smart_interval
            if not has_screen_changed(last_processed_frame, frame_file, threshold=15.0): continue
            
            try:
                # GENERAL PURPOSE VISUAL PROMPT: Works for Math, Cooking, Tech, etc.
                response = ollama.chat(model='llava', messages=[{
                    'role': 'user',
                    'content': 'Analyze this frame. If it contains text, equations, or code, transcribe the most important parts. If it shows an object or action, describe it. Focus on details relevant to a tutorial or presentation.',
                    'images': [frame_file]
                }])
                
                visual_description = response['message']['content']
                last_processed_frame = frame_file
                visual_text = f"[VISUAL AT {timestamp:.1f}s] {visual_description}"
                full_text += visual_text + " "
                
                vector = embed_model.embed_query(visual_text)
                db.add(TranscriptChunk(video_id=video_id, start_time=timestamp, end_time=timestamp + smart_interval, text=visual_text, embedding=vector))
            except Exception as e:
                print(f"Error analyzing frame: {e}")

        # --- STEP 3: THE UNIVERSAL SUMMARY ENGINE ---
        print(f"4. Generating Universal Summary...")
        
        summary_prompt = f"""
        You are an intelligent video analyzer. Analyze the provided [AUDIO] and [VISUAL] data from a video.
        
        Video Title/Label: {original_filename}

        INSTRUCTIONS:
        1. Determine the CATEGORY of the video (e.g., Academic Lecture, Software Tutorial, Product Review, Vlog).
        2. Create a structured summary using the following sections:

        # 📌 Video Title & Subject
        (Clearly state what this video is about)

        # 🎯 Core Purpose / Goal
        (What is the main thing the creator wants the viewer to learn or understand?)

        # 💡 Key Concepts & Takeaways
        (List the 5 most important points, formulas, steps, or insights.)

        # 🛠️ Detailed Walkthrough
        (A chronological breakdown of the video's flow. Include timestamps where possible.)

        # 🏁 Final Conclusion & Perspective
        (Summarize the creator's final thoughts, the solution to the problem, or the verdict on the topic.)

        DATA:
        {full_text[:32768]}
        """
        
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
        print(f"!!! ERROR: {e}")
        video.status = f"FAILED: {str(e)}"
    finally:
        db.commit()
        db.close()