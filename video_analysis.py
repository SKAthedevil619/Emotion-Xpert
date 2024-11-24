import cv2
from fer import FER
from collections import Counter
import os
import tempfile
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np
import audio_analysis
import text_analysis

class EnhancedVideoAnalyzer:
    def __init__(self, max_duration=45):
        self.max_duration = max_duration
        self.detector = FER()
        self.recognizer = sr.Recognizer()

    def extract_audio(self, video_path):
        """Extract audio from video file and save it temporarily"""
        video = VideoFileClip(video_path)
        
        # Create temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
        video.close()
        
        return temp_audio.name

    def transcribe_audio(self, audio_path):
        """Convert speech to text using speech recognition"""
        # Convert audio to compatible format
        audio = AudioSegment.from_wav(audio_path)
        
        # Split audio into chunks to handle long files
        chunk_length_ms = 30000  # 30 seconds
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        full_text = []
        
        for chunk in chunks:
            # Export chunk to temporary file
            chunk_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            chunk.export(chunk_file.name, format="wav")
            
            # Perform recognition
            with sr.AudioFile(chunk_file.name) as source:
                audio_data = self.recognizer.record(source)
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    full_text.append(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    print("API unavailable")
                    
            os.unlink(chunk_file.name)
            
        return " ".join(full_text)

    def analyze_video(self, video_path):
        """Perform comprehensive video analysis including facial, audio, and speech emotions"""
        results = {
            'facial_emotions': {},
            'audio_emotions': {},
            'text_emotions': {},
            'combined_emotions': {}
        }

        # 1. Facial Emotion Analysis
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1 / fps
        emotion_counter = Counter()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = (frame_count * frame_duration)
            if elapsed_time > self.max_duration:
                break

            emotions = self.detector.detect_emotions(frame)
            for emotion in emotions:
                emotion_label = emotion['emotions']
                for key in emotion_label:
                    emotion_counter[key] += emotion_label[key]

            frame_count += 1

        cap.release()

        total_emotions = sum(emotion_counter.values())
        if total_emotions > 0:
            for emotion, count in emotion_counter.items():
                score = count / total_emotions
                if score >= 0.01:  # Filter out emotions with score below 0.01
                    results['facial_emotions'][emotion] = score

        # 2. Audio Emotion Analysis
        try:
            audio_path = self.extract_audio(video_path)
            results['audio_emotions'] = audio_analysis.recognize_emotion(audio_path)
            
            # 3. Speech-to-Text and Text Emotion Analysis
            transcript = self.transcribe_audio(audio_path)
            if transcript:
                results['text_emotions'] = text_analysis.perform_analysis(input_text=transcript)
            
            # Cleanup temporary audio file
            os.unlink(audio_path)
        except Exception as e:
            print(f"Error in audio processing: {str(e)}")

        # 4. Combine all emotions with weighted average
        all_emotions = set()
        for emotion_dict in results.values():
            all_emotions.update(emotion_dict.keys())

        # Define weights for each modality
        weights = {
            'facial_emotions': 0.4,
            'audio_emotions': 0.3,
            'text_emotions': 0.3
        }

        # Calculate weighted average for each emotion
        for emotion in all_emotions:
            weighted_sum = 0
            weight_sum = 0
            
            for modality, weight in weights.items():
                if emotion in results[modality]:
                    weighted_sum += results[modality][emotion] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                combined_score = weighted_sum / weight_sum
                if combined_score >= 0.01:  # Filter combined emotions with score below 0.01
                    results['combined_emotions'][emotion] = combined_score

        return results

def analyze_video_emotions(video_path, max_duration=45):
    """Main function to be called from Flask routes"""
    analyzer = EnhancedVideoAnalyzer(max_duration)
    try:
        return analyzer.analyze_video(video_path)
    except Exception as e:
        print(f"Error in video analysis: {str(e)}")
        return {}
