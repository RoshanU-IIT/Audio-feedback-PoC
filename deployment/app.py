# app.py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

class AudioFeedbackApp:
    def __init__(self):
        # Initialize the whisper model for transcription
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load models
        self.model_id = "openai/whisper-base"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device
        )

    # [Rest of the AudioFeedbackApp class remains the same as in previous code]
    # ... [Previous class methods remain unchanged]

# Create and launch the interface
app = AudioFeedbackApp()
iface = gr.Interface(
    fn=app.process_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Textbox(label="Feedback Response (Optional)")
    ],
    outputs=gr.Textbox(label="Results"),
    title="Audio Conversation Feedback Tool",
    description="Upload an audio file to receive speaker-labeled transcription and feedback."
)

# For Hugging Face Spaces deploymenntt
iface.launch()
