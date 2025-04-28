# SwimVisionLM: Automated Swimming Technique Analysis

## Description

**SwimVision LM** is a cutting-edge, AI-powered web application that helps swimmers and coaches analyze swimming technique through automated, intelligent video analysis. Leveraging state-of-the-art Vision-Language Models (VLMs) like GPT-4o, SwimVision can break down swimming movements, identify key technique issues, and deliver tailored improvement suggestions—all by simply uploading a video. The platform is designed to democratize expert coaching, enabling swimmers of all levels to benefit from professional-level feedback, anytime and anywhere.

## Features

- **Video Upload & Frame Sampling**  
  Upload swimming videos directly through the web interface. The system automatically samples relevant frames for analysis.

- **AI-Powered Technique Analysis**  
  Utilizes advanced VLMs to recognize swim strokes, body posture, and movement phases. Provides detailed, actionable feedback in natural language.

- **Key Frame Preview**  
  Visualize and browse the automatically selected key video frames used for analysis.

- **Actionable Improvement Suggestions**  
  Get customized recommendations for refining swimming technique, based on industry standards and best practices.

- **Session History Management**  
  Efficiently manages analysis results and previously uploaded videos for progress tracking.

- **Simple Web Demo via Gradio**  
  User-friendly web interface powered by Gradio—no installation or technical skills required.

- **Extensible & Modular Codebase**  
  Well-structured, modular Python code for easy development, scaling, and integration with additional AI or computer vision components.

## Getting Started

1. Upload a swimming video from your device.
2. The system will process the video, extract key frames, and analyze them with an AI model.
3. Receive detailed technique analysis and practical training suggestions.
4. Explore your session history and follow your progress over time.

## Launch
```
conda create -n swim-vision-lm python=3.11.7 -y
conda activate swim-vision-lm
pip install -r requirements.txt
gradio main.py
```

