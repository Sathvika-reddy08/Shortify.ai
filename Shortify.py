import os
import yt_dlp
import whisper
import subprocess
import heapq
import nltk
from dataclasses import dataclass
from typing import List, Tuple
from transformers import pipeline
import cv2
from moviepy import VideoFileClip
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Helper Functions
def setup_nltk():
    """Download required NLTK resources."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download NLTK resources. Using basic sentence splitting.")

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK or basic rules."""
    try:
        return nltk.sent_tokenize(text)
    except:
        sentences = []
        current = []
        for chunk in text.split():
            current.append(chunk)
            if chunk.endswith(('.', '!', '?')) and len(chunk) > 1:
                sentences.append(' '.join(current))
                current = []
        if current:
            sentences.append(' '.join(current))
        return sentences

# Data Class for Clip Timestamps
@dataclass
class ClipTimestamp:
    start: float
    end: float
    score: float
    full_text: str
    highlight_text: str

    def _lt_(self, other):
        return self.score < other.score

# Main Class
class YoutubeShortsMaker:
    def _init_(self, video_path: str):
        setup_nltk()
        self.video_path = video_path
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.whisper_model = whisper.load_model("base")
        os.makedirs("output", exist_ok=True)
    
    

    def generate_transcript(self) -> List[dict]:
        """Generate transcript with timestamps."""
        try:
            result = self.whisper_model.transcribe(self.video_path)
            return result["segments"]
        except Exception as e:
            raise Exception(f"Failed to generate transcript: {str(e)}")

    def analyze_segment(self, text: str) -> float:
        """Analyze segment content and return a score."""
        categories = ["key point", "complete thought", "engaging content", "meaningful message", "clear audio"]
        results = self.classifier(text, categories)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        score = sum(score * weight for score, weight in zip(results['scores'], weights))
        return score

    def extract_clips(self, transcript: List[dict]) -> List[ClipTimestamp]:
        """Extract clips with natural sentence boundaries."""
        clips_heap = []
        for segment in transcript:
            start_time = segment['start']
            end_time = start_time + 30  # Ensure 30-second clips
            text = segment['text']
            score = self.analyze_segment(text)
            highlight = text[:10]  # Extract first 10 characters as highlight
            clip = ClipTimestamp(start=start_time, end=end_time, score=score, full_text=text, highlight_text=highlight)
            heapq.heappush(clips_heap, (-score, clip))
        return clips_heap

    def select_top_clips(self, clips_heap: List[Tuple[float, ClipTimestamp]], num_clips: int) -> List[ClipTimestamp]:
        """Select top non-overlapping clips."""
        selected_clips = []
        while clips_heap and len(selected_clips) < num_clips:
            _, clip = heapq.heappop(clips_heap)
            selected_clips.append(clip)
        return selected_clips

    def cut_clip(self, clip: ClipTimestamp, idx: int):
        """Cut video and generate output."""
        output_path = f"output/clip_{idx}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip.start),
            "-i", self.video_path,
            "-t", str(clip.end - clip.start),
            "-c:v", "libx264", "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
        print(f"✅ Created clip {idx}")

    def convert_to_vertical(self, input_video: str, output_video: str):
        """Convert video to vertical format."""
        subprocess.run([
            "ffmpeg", "-i", input_video,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", output_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"📏 Converted {input_video} to vertical format.")

    def extract_audio(self, video_path: str, audio_path: str):
        """Extract audio from a video file."""
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le", ffmpeg_params=["-ac", "1"])
        print("✅ Audio extracted successfully!")

    def generate_captions(self, audio_path: str):
        """Generate captions using Whisper."""
        model = whisper.load_model("small")
        result = model.transcribe(audio_path, word_timestamps=True)
        captions = []
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            words = segment.get("words", [])
            captions.append((start_time, end_time, words))
        return captions

    def overlay_captions(self, video_path: str, captions: List[Tuple[float, float, List[dict]]], output_path: str):
        """Overlay captions on the video with word highlighting."""
        # First, create a video with captions (without audio)
        temp_output = output_path.replace(".mp4", "_temp.mp4")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened():
            print("❌ Error: Could not open video file.")
            return

        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        if not out.isOpened():
            print("❌ Error: Could not initialize VideoWriter.")
            return

        frame_num = 0
        caption_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps
            if caption_index < len(captions):
                start_time, end_time, words = captions[caption_index]
                if start_time <= timestamp <= end_time:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.8
                    thickness = 6
                    line_spacing = 60
                    min_x = int(width * 0.10)
                    max_x = int(width * 0.90)
                    max_text_width = max_x - min_x

                    text_rows = []
                    current_line = ""
                    for word in words:
                        word_text = word.get("text", word.get("word", ""))
                        if not word_text:
                            continue
                        next_line = current_line + " " + word_text if current_line else word_text
                        text_size = cv2.getTextSize(next_line, font, font_scale, thickness)[0]
                        if text_size[0] > max_text_width:
                            text_rows.append(current_line)
                            current_line = word_text
                        else:
                            current_line = next_line
                    if current_line:
                        text_rows.append(current_line)

                    min_y = int(height * 0.65)
                    max_y = int(height * 0.85)
                    line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + line_spacing
                    total_text_height = len(text_rows) * line_height
                    start_y = (min_y + max_y - total_text_height) // 2
                    start_y = max(min_y, min(start_y, max_y - total_text_height))

                    for i, line in enumerate(text_rows):
                        words_in_line = line.split()
                        word_positions = []
                        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                        text_x = min_x + (max_text_width - text_size[0]) // 2
                        text_y = start_y + i * line_height

                        word_x = text_x
                        for word in words_in_line:
                            word_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
                            word_positions.append((word_x, text_y, word))
                            word_x += word_size[0] + 10

                        for word_x, word_y, word_text in word_positions:
                            is_highlighted = False
                            for word in words:
                                word_actual = word.get("text", word.get("word", ""))
                                if word_actual.strip() == word_text.strip() and word["start"] <= timestamp <= word["end"]:
                                    is_highlighted = True
                                    break

                            highlight_color = (0, 215, 255) if is_highlighted else (255, 255, 255)
                            shadow_thickness = 8 if is_highlighted else 6

                            cv2.putText(frame, word_text, (word_x, word_y), font, font_scale, (0, 0, 0), shadow_thickness, cv2.LINE_AA)
                            cv2.putText(frame, word_text, (word_x, word_y), font, font_scale, highlight_color, thickness, cv2.LINE_AA)

                elif timestamp > end_time:
                    caption_index += 1

            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()

        # Merge the captioned video with the original audio
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_output,
            "-i", video_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
        os.remove(temp_output)  # Clean up temporary file

        if os.path.exists(output_path):
            print(f"✅ Captions added to video! Saved at: {os.path.abspath(output_path)}")
        else:
            print("❌ Error: Output file was not generated.")

    def process_video(self, num_clips: int = 3):
        """Main processing pipeline."""
        print("🎯 Generating transcript...")
        transcript = self.generate_transcript()

        print("📊 Analyzing content...")
        clips_heap = self.extract_clips(transcript)

        print("✂ Selecting best clips...")
        selected_clips = self.select_top_clips(clips_heap, num_clips)

        print("🎬 Creating final clips...")
        for idx, clip in enumerate(selected_clips, 1):
            output_clip_path = f"output/clip_{idx}.mp4"
            self.cut_clip(clip, idx)
            vertical_output = f"output/clip_{idx}_vertical.mp4"
            self.convert_to_vertical(output_clip_path, vertical_output)

            # Add captions to the vertical clip
            audio_path = "temp_audio.wav"
            self.extract_audio(vertical_output, audio_path)
            captions = self.generate_captions(audio_path)
            final_output = f"output/clip_{idx}_final.mp4"
            self.overlay_captions(vertical_output, captions, final_output)
            os.remove(audio_path)

        print("🎉 Processing complete! Check the 'output' folder.")

# Entry Point
if _name_ == "_main_":
    def download_youtube_video(url, output_path="video.mp4"):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_path
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    url=input("enter the url of youtube viedo")
    video_path = download_youtube_video(link)
    shorts_maker = YoutubeShortsMaker(video_path)
    shorts_maker.process_video()
