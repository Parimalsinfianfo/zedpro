import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import uuid
import json
import base64
import re
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import tempfile
import subprocess
import requests
from PIL import Image
import torch
from transformers import pipeline
import ffmpeg
from groq import Groq
import base64
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pymupdf  # PDF processing
from typing import Optional
from typing import Dict, Any
import wave
import contextlib
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import soundfile as sf

# Configuration
st.set_page_config(
    page_title="AI Apprentice Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    :root {
        --primary: #1e3c72;
        --secondary: #2a5298;
        --accent: #ff6b6b;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
    }
    
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .video-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        background: #f8f9fa;
    }
    
    .metric-card {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        text-align: center;
        border-top: 3px solid var(--primary);
    }
    
    .personality-radar {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .profile-card {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid var(--primary);
    }
    
    .skill-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        display: inline-block;
        font-size: 12px;
    }
    
    .record-button {
        background-color: var(--accent);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .record-button:hover {
        background-color: #ff4444;
        transform: scale(1.05);
    }
    
    .subscription-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 10px;
        text-align: center;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .subscription-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .subscription-card.basic {
        border-top: 4px solid #4caf50;
    }
    
    .subscription-card.standard {
        border-top: 4px solid #2196f3;
    }
    
    .subscription-card.professional {
        border-top: 4px solid #ff9800;
    }
    
    .language-selector {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 100;
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 10px;
    }
    
    .notification-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background-color: var(--accent);
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
    }
    
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        background: #e9ecef;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .chat-message {
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #d1e7ff;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .other-message {
        background-color: #f1f1f1;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
    
    .tab-content {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 10px;
    }
    
    .ai-insight-section {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #4caf50;
    }
    
    .radar-chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .step-container {
        background: #e8f4ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .step-timer {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff6b6b;
        margin: 0.5rem 0;
    }
    .question-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .psychometric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Advanced Profiler Classes
class GroqLLM:
    """Custom wrapper for Groq API"""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        self.client = Groq(api_key=groq_api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class ApprenticeProfiler:
    def __init__(self):
        self.models = {}
        self.setup_models()
        self.recognizer = sr.Recognizer()
        
    def setup_models(self):
        """Initialize all AI models with proper error handling"""
        self.models = {}
        
        try:
            # Only load models if transformers is available
            with st.spinner("Loading AI models..."):
                # Try to load Whisper for transcription
                try:
                    self.models['whisper'] = pipeline(
                        "automatic-speech-recognition",
                        model="openai/whisper-large-v2",  # Using larger model for better accuracy
                        device=-1  # Force CPU usage for compatibility
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not load Whisper model: {str(e)}")
                
                # Try to load NER model
                try:
                    self.models['ner'] = pipeline(
                        "ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple",
                        device=-1
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not load NER model: {str(e)}")
                
                # Try to load sentiment model for personality analysis
                try:
                    self.models['sentiment'] = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=-1
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not load sentiment model: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error setting up models: {str(e)}")
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video using ffmpeg with robust error handling"""
        try:
            audio_path = video_path.replace('.mp4', '.wav').replace('.mov', '.wav').replace('.webm', '.wav')
            
            # Robust ffmpeg command
            command = [
                'ffmpeg', 
                '-y',  # Overwrite output file without asking
                '-i', video_path, 
                '-vn',  # Disable video recording
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ar', '16000',  # Audio sample rate
                '-ac', '1',  # Audio channels (mono)
                '-loglevel', 'error',  # Only show errors
                audio_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                st.error(f"FFmpeg error: {result.stderr}")
                return None
                
            if not os.path.exists(audio_path):
                st.error("Audio extraction failed: output file not created")
                return None
                
            return audio_path
                
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return None
    
    def transcribe_long_audio(self, audio_path: str) -> str:
        """Transcribe long audio files with optimized progress tracking"""
        try:
            # Split audio into chunks based on silence
            audio = AudioSegment.from_wav(audio_path)
            chunks = split_on_silence(
                audio,
                min_silence_len=1000,  # Increased silence length for fewer chunks
                silence_thresh=-40,
                keep_silence=250
            )
            
            transcript = ""
            total_chunks = len(chunks)
            
            # Create progress bar in status
            with st.status(f"Transcribing {total_chunks} audio chunks...", expanded=True) as status:
                for i, chunk in enumerate(chunks):
                    # Export chunk
                    chunk_path = f"chunk_{i}.wav"
                    chunk.export(chunk_path, format="wav")
                    
                    try:
                        # Transcribe chunk with timeout
                        if 'whisper' in self.models:
                            result = self.models['whisper'](chunk_path)
                            transcript += result['text'] + " "
                        else:
                            # Fallback to speech_recognition
                            with sr.AudioFile(chunk_path) as source:
                                audio_data = self.recognizer.record(source)
                                text = self.recognizer.recognize_google(audio_data)
                                transcript += text + " "
                    except Exception as e:
                        st.warning(f"Error processing chunk {i+1}: {str(e)}")
                        continue
                    finally:
                        # Clean up chunk file
                        try:
                            os.unlink(chunk_path)
                        except:
                            pass
                    
                    # Update progress
                    progress = (i + 1) / total_chunks
                    status.update(label=f"Processing chunk {i+1}/{total_chunks} ({int(progress*100)}%)")
            
            return transcript
                
        except Exception as e:
            st.error(f"Long audio transcription error: {str(e)}")
            return ""
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using best available method"""
        try:
            # Check audio duration
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            
            # For short audio (< 2 min), use Whisper directly
            if duration < 120 and 'whisper' in self.models:
                with st.spinner("Transcribing audio with AI model..."):
                    result = self.models['whisper'](audio_path)
                    return result['text']
            else:
                # For long audio, use chunking method
                return self.transcribe_long_audio(audio_path)
                
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return f"Transcription failed: {str(e)}"
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF or fallback"""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def extract_candidate_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information using NER or regex fallback"""
        try:
            if 'ner' in self.models:
                # Use NER model
                entities = self.models['ner'](text)
                
                # Process entities
                info = {
                    'name': '',
                    'email': '',
                    'phone': '',
                    'skills': [],
                    'education': '',
                    'experience': '',
                    'goals': '',
                    'organizations': []
                }
                
                # Extract entities
                for entity in entities:
                    if entity['entity_group'] == 'PER' and not info['name']:
                        info['name'] = entity['word']
                    elif entity['entity_group'] == 'ORG':
                        info['organizations'].append(entity['word'])
                
                # Combine with regex extraction
                regex_info = self.extract_info_with_regex(text)
                
                # Merge information
                for key in ['email', 'phone']:
                    if regex_info[key] and not info[key]:
                        info[key] = regex_info[key]
                
                info['skills'] = regex_info['skills']
                if regex_info['education']:
                    info['education'] = regex_info['education']
                
                return info
            else:
                return self.extract_info_with_regex(text)
                
        except Exception as e:
            st.error(f"Information extraction error: {str(e)}")
            return self.extract_info_with_regex(text)
    
    def extract_info_with_regex(self, text: str) -> Dict[str, Any]:
        """Fallback method using regex patterns"""
        info = {
            'name': '',
            'email': '',
            'phone': '',
            'skills': [],
            'education': '',
            'experience': '',
            'goals': '',
            'organizations': []
        }
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            info['email'] = emails[0]
        
        # Phone extraction (improved pattern)
        phone_patterns = [
            r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(\+\d{1,3}[-.\s]?)?\d{10}',
            r'(\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                info['phone'] = phones[0] if isinstance(phones[0], str) else ''.join(phones[0])
                break
        
        # Name extraction (improved)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and not any(char.isdigit() for char in line) and not '@' in line:
                words = re.findall(r'\b[A-Z][a-z]+\b', line)
                if len(words) >= 2:
                    info['name'] = ' '.join(words[:3])  # Take up to 3 names
                    break
        
        # Skills extraction (expanded list)
        skill_keywords = [
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            # Web technologies
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
            # AI/ML
            'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch', 'scikit-learn',
            # Soft skills
            'leadership', 'communication', 'teamwork', 'problem solving', 'project management',
            'agile', 'scrum', 'analytical thinking', 'creativity'
        ]
        
        text_lower = text.lower()
        found_skills = []
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        info['skills'] = list(set(found_skills))  # Remove duplicates
        
        # Education extraction
        education_keywords = ['degree', 'bachelor', 'master', 'phd', 'university', 'college', 'diploma', 'certification']
        education_sentences = []
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in education_keywords):
                education_sentences.append(sentence.strip())
        
        if education_sentences:
            info['education'] = '. '.join(education_sentences[:2])  # Take first 2 relevant sentences
        
        # Experience extraction
        experience_keywords = ['experience', 'worked', 'position', 'role', 'job', 'company', 'years']
        experience_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in experience_keywords):
                experience_sentences.append(sentence.strip())
        
        if experience_sentences:
            info['experience'] = '. '.join(experience_sentences[:3])  # Take first 3 relevant sentences
        
        return info
    
    def analyze_personality(self, text: str) -> Dict[str, float]:
        """Analyze personality traits from text"""
        try:
            if 'sentiment' in self.models:
                # Use sentiment analysis as a proxy for personality traits
                with st.spinner("Analyzing personality traits..."):
                    # Split text into chunks for better analysis
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    
                    sentiment_scores = []
                    for chunk in chunks[:5]:  # Analyze first 5 chunks
                        if chunk.strip():
                            result = self.models['sentiment'](chunk)
                            sentiment_scores.append(result[0])
                    
                    # Convert sentiment to personality traits (simplified mapping)
                    personality = self.sentiment_to_personality(sentiment_scores, text)
                    return personality
            else:
                return self.rule_based_personality(text)
                
        except Exception as e:
            st.error(f"Personality analysis error: {str(e)}")
            return self.rule_based_personality(text)
    
    def sentiment_to_personality(self, sentiment_scores: list, text: str) -> Dict[str, float]:
        """Convert sentiment analysis to OCEAN personality traits"""
        
        # Initialize personality traits
        personality = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        
        if not sentiment_scores:
            return personality
        
        # Calculate average sentiment
        positive_count = sum(1 for score in sentiment_scores if score['label'] == 'LABEL_2')  # Positive
        negative_count = sum(1 for score in sentiment_scores if score['label'] == 'LABEL_0')  # Negative
        
        total_scores = len(sentiment_scores)
        if total_scores > 0:
            positive_ratio = positive_count / total_scores
            negative_ratio = negative_count / total_scores
            
            # Map sentiment to personality traits
            personality['extraversion'] = min(0.9, 0.3 + positive_ratio * 0.6)
            personality['agreeableness'] = min(0.9, 0.3 + positive_ratio * 0.5)
            personality['neuroticism'] = min(0.9, 0.2 + negative_ratio * 0.6)
        
        # Combine with rule-based analysis
        rule_based = self.rule_based_personality(text)
        
        # Average the two approaches
        for trait in personality:
            personality[trait] = (personality[trait] + rule_based[trait]) / 2
        
        return personality
    
    def rule_based_personality(self, text: str) -> Dict[str, float]:
        """Rule-based personality analysis as fallback"""
        text_lower = text.lower()
        
        # Enhanced keywords for each trait
        trait_keywords = {
            'openness': ['creative', 'innovative', 'curious', 'imaginative', 'artistic', 'original', 'inventive', 'experimental'],
            'conscientiousness': ['organized', 'responsible', 'detail', 'planning', 'systematic', 'thorough', 'careful', 'disciplined'],
            'extraversion': ['outgoing', 'social', 'energetic', 'talkative', 'assertive', 'enthusiastic', 'active', 'gregarious'],
            'agreeableness': ['cooperative', 'helpful', 'kind', 'supportive', 'friendly', 'empathetic', 'considerate', 'compassionate'],
            'neuroticism': ['anxious', 'worried', 'stressed', 'emotional', 'sensitive', 'nervous', 'tense', 'moody']
        }
        
        personality = {}
        
        for trait, words in trait_keywords.items():
            # Count occurrences and calculate score
            word_count = sum(1 for word in words if word in text_lower)
            word_frequency = word_count / len(words)
            
            # Normalize score between 0.1 and 0.9
            base_score = 0.5
            adjustment = (word_frequency - 0.1) * 0.4  # Scale adjustment
            
            personality[trait] = max(0.1, min(0.9, base_score + adjustment))
        
        return personality
    
    def generate_insights(self, candidate_data: Dict, transcript: str, 
                         personality: Dict, groq_api_key: str) -> str:
        """Generate AI insights using Groq or fallback method"""
        try:
            if groq_api_key:
                # Use Groq API for insights
                llm = GroqLLM(groq_api_key=groq_api_key)
                
                prompt = f"""
You are an expert AI career counselor and talent assessment specialist.

Analyze the following candidate information and provide comprehensive insights:

CANDIDATE INFORMATION:
Name: {candidate_data.get('name', 'Not provided')}
Email: {candidate_data.get('email', 'Not provided')}
Phone: {candidate_data.get('phone', 'Not provided')}
Skills: {', '.join(candidate_data.get('skills', []))}
Education: {candidate_data.get('education', 'Not provided')}
Experience: {candidate_data.get('experience', 'Not provided')}

PERSONALITY TRAITS (0-1 scale):
Openness: {personality.get('openness', 0.5):.2f}
Conscientiousness: {personality.get('conscientiousness', 0.5):.2f}
Extraversion: {personality.get('extraversion', 0.5):.2f}
Agreeableness: {personality.get('agreeableness', 0.5):.2f}
Neuroticism: {personality.get('neuroticism', 0.5):.2f}

TRANSCRIPT SAMPLE: {transcript[:500]}...

Please provide a comprehensive analysis with the following sections:
1. Professional Background Summary
2. Key Strengths and Skills Assessment
3. Personality Profile Analysis
4. Career Path Recommendations
5. Development Areas and Suggestions

Format your response clearly with headers and bullet points where appropriate.
"""
                
                return llm.generate(prompt)
            else:
                return self.generate_rule_based_insights(candidate_data, personality)
                
        except Exception as e:
            st.error(f"Insight generation error: {str(e)}")
            return self.generate_rule_based_insights(candidate_data, personality)
    
    def generate_rule_based_insights(self, candidate_data: Dict, personality: Dict) -> str:
        """Generate insights using rule-based approach as fallback"""
        
        insights = []
        name = candidate_data.get('name', 'The candidate')
        
        # Professional Background Summary
        insights.append("## 📌 Professional Background Summary")
        insights.append(f"{name} presents a profile with the following characteristics:")
        
        if candidate_data.get('skills'):
            skills_count = len(candidate_data['skills'])
            insights.append(f"- Demonstrates {skills_count} identified technical and soft skills")
        
        if candidate_data.get('education'):
            insights.append(f"- Educational background: {candidate_data['education'][:100]}...")
            
        if candidate_data.get('experience'):
            insights.append(f"- Professional experience: {candidate_data['experience'][:100]}...")
        
        insights.append("")
        
        # Key Strengths Assessment
        insights.append("## 💡 Key Strengths and Skills Assessment")
        skills = candidate_data.get('skills', [])
        
        if skills:
            # Categorize skills
            technical_skills = [s for s in skills if any(tech in s.lower() for tech in ['python', 'java', 'sql', 'react', 'aws', 'docker'])]
            soft_skills = [s for s in skills if any(soft in s.lower() for soft in ['leadership', 'communication', 'teamwork', 'problem'])]
            
            if technical_skills:
                insights.append(f"- **Technical Skills**: {', '.join(technical_skills[:5])}")
            if soft_skills:
                insights.append(f"- **Soft Skills**: {', '.join(soft_skills[:5])}")
            if len(skills) > 10:
                insights.append(f"- **Skill Diversity**: Demonstrates broad skill set with {len(skills)} identified competencies")
        else:
            insights.append("- Skills assessment requires more detailed information from candidate")
        
        insights.append("")
        
        # Personality Profile Analysis
        insights.append("## 🧠 Personality Profile Analysis (OCEAN Model)")
        
        # Find dominant traits
        sorted_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)
        top_traits = [trait for trait, score in sorted_traits[:2] if score > 0.6]
        low_traits = [trait for trait, score in sorted_traits if score < 0.4]
        
        trait_descriptions = {
            'openness': 'creative, curious, and open to new experiences',
            'conscientiousness': 'organized, responsible, and detail-oriented',
            'extraversion': 'outgoing, energetic, and socially confident',
            'agreeableness': 'cooperative, empathetic, and team-oriented',
            'neuroticism': 'emotionally sensitive and reactive to stress'
        }
        
        for trait, score in sorted_traits:
            level = "High" if score > 0.6 else "Moderate" if score > 0.4 else "Low"
            insights.append(f"- **{trait.title()}**: {level} ({score:.2f}) - {trait_descriptions.get(trait, 'characteristic trait')}")
        
        insights.append("")
        
        # Career Recommendations
        insights.append("## 🚀 Career Path Recommendations")
        
        if top_traits:
            primary_trait = sorted_traits[0][0]
            
            career_recommendations = {
                'openness': [
                    "Product Manager - Innovation focused roles",
                    "UX/UI Designer - Creative problem solving",
                    "Research & Development - Experimental projects",
                    "Strategy Consultant - Novel solution development"
                ],
                'conscientiousness': [
                    "Project Manager - Detail-oriented execution",
                    "Quality Assurance - Systematic testing approaches",
                    "Operations Manager - Process optimization",
                    "Financial Analyst - Thorough data analysis"
                ],
                'extraversion': [
                    "Sales Manager - Client relationship building",
                    "Team Lead - People management and motivation",
                    "Business Development - Networking and partnerships",
                    "Customer Success Manager - Client engagement"
                ],
                'agreeableness': [
                    "Human Resources - Employee relations and support",
                    "Social Work - Community and individual assistance",
                    "Team Coordinator - Collaborative project management",
                    "Customer Service Manager - Client satisfaction focus"
                ],
                'neuroticism': [
                    "Data Analyst - Structured analytical work",
                    "Technical Writer - Detailed documentation",
                    "Quality Control - Attention to detail and standards",
                    "Research Assistant - Methodical investigation"
                ]
            }
            
            recommendations = career_recommendations.get(primary_trait, ["General management roles", "Analytical positions"])
            
            for rec in recommendations:
                insights.append(f"- {rec}")
        else:
            insights.append("- Versatile profile suitable for various analytical and collaborative roles")
            insights.append("- Consider roles that balance technical skills with interpersonal interaction")
        
        insights.append("")
        
        # Development Areas
        insights.append("## 📈 Development Areas and Suggestions")
        
        if low_traits:
            for trait in low_traits:
                development_suggestions = {
                    'openness': "Consider exploring creative projects, attending innovation workshops, or engaging in brainstorming sessions",
                    'conscientiousness': "Focus on developing organizational systems, time management tools, and attention to detail practices",
                    'extraversion': "Practice public speaking, join networking events, or take on presentation opportunities",
                    'agreeableness': "Develop empathy through active listening training and collaborative project participation",
                    'neuroticism': "Consider stress management techniques, mindfulness practices, and emotional regulation strategies"
                }
                
                suggestion = development_suggestions.get(trait, f"Focus on developing {trait} skills")
                insights.append(f"- **{trait.title()} Development**: {suggestion}")
        
        # General development recommendations
        insights.append("- **Continuous Learning**: Stay updated with industry trends and emerging technologies")
        insights.append("- **Network Building**: Engage with professional communities and industry groups")
        insights.append("- **Skill Certification**: Consider relevant certifications to validate expertise")
        
        return '\n'.join(insights)

def create_personality_radar_chart(personality_data):
    """Create a radar chart for personality traits"""
    traits = list(personality_data.keys())
    values = list(personality_data.values())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[trait.title() for trait in traits],
        fill='toself',
        name='Personality Profile',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='rgb(31, 119, 180)', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['Low', 'Below Avg', 'Average', 'Above Avg', 'High']
            )),
        showlegend=True,
        title={
            'text': "OCEAN Personality Traits Profile",
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=12),
        height=400
    )
    
    return fig

def create_skills_chart(skills_list):
    """Create a bar chart for skills"""
    if not skills_list:
        return None
    
    # Categorize skills
    categories = {
        'Programming': ['Python', 'Java', 'Javascript', 'C++', 'C#'],
        'Web Technologies': ['React', 'Angular', 'Vue', 'Node.js', 'Express'],
        'Database': ['SQL', 'MySQL', 'MongoDB', 'PostgreSQL'],
        'Cloud & DevOps': ['AWS', 'Azure', 'Docker', 'Kubernetes'],
        'AI/ML': ['Machine Learning', 'Deep Learning', 'AI', 'Tensorflow'],
        'Soft Skills': ['Leadership', 'Communication', 'Teamwork', 'Problem Solving']
    }
    
    skill_counts = {cat: 0 for cat in categories}
    
    for skill in skills_list:
        for category, keywords in categories.items():
            if any(keyword.lower() in skill.lower() for keyword in keywords):
                skill_counts[category] += 1
                break
    
    # Filter categories with skills
    filtered_counts = {k: v for k, v in skill_counts.items() if v > 0}
    
    if not filtered_counts:
        return None
    
    fig = go.Figure(data=[
        go.Bar(x=list(filtered_counts.keys()), 
               y=list(filtered_counts.values()),
               marker_color='rgba(31, 119, 180, 0.7)')
    ])
    
    fig.update_layout(
        title="Skills Distribution by Category",
        xaxis_title="Skill Category",
        yaxis_title="Number of Skills",
        showlegend=False
    )
    
    return fig

# Initialize the database simulation
def init_database():
    if 'users' not in st.session_state:
        st.session_state.users = {}
    if 'apprentices' not in st.session_state:
        st.session_state.apprentices = {}
    if 'companies' not in st.session_state:
        st.session_state.companies = {}
    if 'training_providers' not in st.session_state:
        st.session_state.training_providers = {}
    if 'job_postings' not in st.session_state:
        st.session_state.job_postings = {}
    if 'matches' not in st.session_state:
        st.session_state.matches = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = {}
    if 'system_config' not in st.session_state:
        st.session_state.system_config = {
            'languages': ['English', 'Punjabi', 'Urdu', 'Hindi', 'Mirpuri', 'Arabic'],
            'countries': ['UK', 'India', 'USA', 'Canada', 'Australia'],
            'cities': {
                'UK': ['London', 'Manchester', 'Birmingham', 'Leeds'],
                'India': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
                'USA': ['New York', 'Los Angeles', 'Chicago', 'Boston'],
                'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Ottawa'],
                'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth']
            },
            'subscription_plans': {
                'basic': {
                    'name': 'BASIC',
                    'price': 99,
                    'candidates': 50,
                    'features': ['50 candidate views', 'Basic filtering', 'Email support']
                },
                'standard': {
                    'name': 'STANDARD',
                    'price': 299,
                    'candidates': 100,
                    'features': ['100 candidate views', 'Advanced filtering', 'Priority support']
                },
                'professional': {
                    'name': 'PROFESSIONAL',
                    'price': 599,
                    'candidates': 300,
                    'features': ['300 candidate views', 'AI matching', 'Dedicated account manager']
                }
            }
        }
    
    # Add sample data if needed
    if not st.session_state.apprentices:
        add_sample_apprentices()
    if not st.session_state.companies:
        add_sample_companies()
    if not st.session_state.training_providers:
        add_sample_training_providers()
    if not st.session_state.job_postings:
        add_sample_jobs()
    
    # Add sample questions for psychometric test
    if 'questions' not in st.session_state:
        add_sample_questions()
    
    # Add video steps
    if 'video_steps' not in st.session_state:
        add_sample_video_steps()

def add_sample_apprentices():
    sample_apprentices = [
        {
            'id': 'ap1',
            'name': 'Sarah Johnson',
            'age': 19,
            'location': 'London',
            'education': 'A-levels in Computer Science',
            'skills': ['Python', 'JavaScript', 'Web Development'],
            'languages': ['English', 'Spanish'],
            'goals': 'Become a full-stack developer',
            'video_profile': None,
            'personality': {
                'openness': 0.85,
                'conscientiousness': 0.72,
                'extraversion': 0.65,
                'agreeableness': 0.78,
                'neuroticism': 0.42
            },
            'availability': True,
            'created_at': datetime(2024, 5, 15)
        },
        {
            'id': 'ap2',
            'name': 'Ahmed Khan',
            'age': 20,
            'location': 'Manchester',
            'education': 'BTEC in Digital Marketing',
            'skills': ['Social Media', 'Content Creation', 'Analytics'],
            'languages': ['English', 'Urdu', 'Arabic'],
            'goals': 'Digital marketing specialist',
            'video_profile': None,
            'personality': {
                'openness': 0.68,
                'conscientiousness': 0.81,
                'extraversion': 0.75,
                'agreeableness': 0.83,
                'neuroticism': 0.35
            },
            'availability': True,
            'created_at': datetime(2024, 4, 22)
        },
        {
            'id': 'ap3',
            'name': 'Priya Patel',
            'age': 18,
            'location': 'Birmingham',
            'education': 'A-levels in Healthcare',
            'skills': ['Healthcare', 'Administration', 'Communication'],
            'languages': ['English', 'Hindi', 'Gujarati'],
            'goals': 'Healthcare administration',
            'video_profile': None,
            'personality': {
                'openness': 0.72,
                'conscientiousness': 0.88,
                'extraversion': 0.58,
                'agreeableness': 0.91,
                'neuroticism': 0.29
            },
            'availability': True,
            'created_at': datetime(2024, 6, 5)
        }
    ]
    
    for apprentice in sample_apprentices:
        st.session_state.apprentices[apprentice['id']] = apprentice

def add_sample_companies():
    sample_companies = [
        {
            'id': 'co1',
            'name': 'Tech Innovations Ltd',
            'industry': 'Technology',
            'location': 'London',
            'description': 'Leading software development company specializing in AI solutions',
            'subscription': 'professional',
            'job_postings': ['job1'],
            'created_at': datetime(2024, 3, 10)
        },
        {
            'id': 'co2',
            'name': 'Global Media Group',
            'industry': 'Media & Marketing',
            'location': 'Manchester',
            'description': 'International media company with focus on digital marketing',
            'subscription': 'standard',
            'job_postings': ['job2'],
            'created_at': datetime(2024, 2, 18)
        },
        {
            'id': 'co3',
            'name': 'HealthFirst Network',
            'industry': 'Healthcare',
            'location': 'Birmingham',
            'description': 'Healthcare provider with nationwide network of clinics',
            'subscription': 'basic',
            'job_postings': ['job3'],
            'created_at': datetime(2024, 4, 30)
        }
    ]
    
    for company in sample_companies:
        st.session_state.companies[company['id']] = company

def add_sample_training_providers():
    sample_providers = [
        {
            'id': 'tp1',
            'name': 'London Tech College',
            'location': 'London',
            'accreditation': 'Ofsted Outstanding',
            'courses': ['Software Development', 'Data Science', 'AI'],
            'subscription': 'professional',
            'created_at': datetime(2024, 1, 15)
        },
        {
            'id': 'tp2',
            'name': 'Manchester Business School',
            'location': 'Manchester',
            'accreditation': 'Ofsted Good',
            'courses': ['Digital Marketing', 'Business Administration', 'Finance'],
            'subscription': 'standard',
            'created_at': datetime(2024, 2, 22)
        },
        {
            'id': 'tp3',
            'name': 'Midlands Healthcare Institute',
            'location': 'Birmingham',
            'accreditation': 'Ofsted Good',
            'courses': ['Healthcare Administration', 'Nursing', 'Pharmacy'],
            'subscription': 'basic',
            'created_at': datetime(2024, 3, 18)
        }
    ]
    
    for provider in sample_providers:
        st.session_state.training_providers[provider['id']] = provider

def add_sample_jobs():
    sample_jobs = [
        {
            'id': 'job1',
            'title': 'Software Development Apprentice',
            'company': 'co1',
            'location': 'London',
            'skills': ['Python', 'JavaScript', 'Web Development'],
            'description': 'Join our team as a software development apprentice and learn from industry experts.',
            'match_score': 92,
            'created_at': datetime(2024, 5, 10)
        },
        {
            'id': 'job2',
            'title': 'Digital Marketing Apprentice',
            'company': 'co2',
            'location': 'Manchester',
            'skills': ['Social Media', 'Content Creation', 'Analytics'],
            'description': 'Exciting opportunity for a digital marketing apprentice in a fast-paced media environment.',
            'match_score': 85,
            'created_at': datetime(2024, 5, 15)
        },
        {
            'id': 'job3',
            'title': 'Healthcare Administration Apprentice',
            'company': 'co3',
            'location': 'Birmingham',
            'skills': ['Healthcare', 'Administration', 'Communication'],
            'description': 'Support our healthcare administration team and gain valuable experience in the medical field.',
            'match_score': 78,
            'created_at': datetime(2024, 5, 20)
        }
    ]
    
    for job in sample_jobs:
        st.session_state.job_postings[job['id']] = job

def add_sample_questions():
    questions = [
        {"id": 1, "text": "I feel comfortable in social situations", "trait": "extraversion"},
        {"id": 2, "text": "I pay attention to details", "trait": "conscientiousness"},
        {"id": 3, "text": "I often feel worried or anxious", "trait": "neuroticism"},
        {"id": 4, "text": "I enjoy trying new experiences", "trait": "openness"},
        {"id": 5, "text": "I trust people easily", "trait": "agreeableness"},
        {"id": 6, "text": "I prefer working alone rather than in a team", "trait": "extraversion", "reverse": True},
        {"id": 7, "text": "I complete tasks on time", "trait": "conscientiousness"},
        {"id": 8, "text": "I get upset easily", "trait": "neuroticism"},
        {"id": 9, "text": "I have a vivid imagination", "trait": "openness"},
        {"id": 10, "text": "I consider others' feelings", "trait": "agreeableness"},
    ]
    st.session_state.questions = questions

def add_sample_video_steps():
    steps = [
        {"title": "About Yourself", "duration": 30, "prompt": "Talk about yourself for 30 seconds", "key": "about_yourself"},
        {"title": "Name/Sex/Ethnicity", "duration": 20, "prompt": "Talk about your name, sex, and ethnicity", "key": "demographics", "optional": True},
        {"title": "Country/City/College", "duration": 20, "prompt": "Talk about your country, city, and college", "key": "location", "optional": True},
        {"title": "Education", "duration": 20, "prompt": "Talk about your education", "key": "education", "optional": True},
        {"title": "Languages", "duration": 20, "prompt": "Talk about languages you speak", "key": "languages"},
        {"title": "Specialization", "duration": 20, "prompt": "Talk about your specialization", "key": "specialization", "optional": True},
        {"title": "Career Goals", "duration": 20, "prompt": "Talk about what you're looking for", "key": "goals"},
        {"title": "Location Flexibility", "duration": 20, "prompt": "Talk about your current location and flexibility to travel", "key": "flexibility", "optional": True},
        {"title": "Hobbies", "duration": 20, "prompt": "Talk about your sports and hobbies", "key": "hobbies", "optional": True},
        {"title": "Special Needs", "duration": 20, "prompt": "Talk about any special needs", "key": "needs", "optional": True},
    ]
    st.session_state.video_steps = steps

# Authentication functions
def create_user(email, password, user_type, profile_data):
    user_id = str(uuid.uuid4())
    st.session_state.users[user_id] = {
        'id': user_id,
        'email': email,
        'password': password,
        'user_type': user_type,
        'created_at': datetime.now(),
        'profile_data': profile_data
    }
    
    # Store in specific user type collections
    if user_type == 'apprentice':
        profile_data['id'] = user_id
        st.session_state.apprentices[user_id] = profile_data
    elif user_type == 'company':
        profile_data['id'] = user_id
        st.session_state.companies[user_id] = profile_data
    elif user_type == 'training_provider':
        profile_data['id'] = user_id
        st.session_state.training_providers[user_id] = profile_data
    
    return user_id

def authenticate_user(email, password):
    for user_id, user_data in st.session_state.users.items():
        if user_data['email'] == email and user_data['password'] == password:
            return user_id, user_data
    return None, None

# Guided Video Recording
def guided_video_recording():
    st.subheader("🎥 Guided Video Profile Creation")
    st.info("Follow the step-by-step prompts to create your video profile")
    
    # Steps configuration
    steps = st.session_state.video_steps
    
    # Initialize session state for steps
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
        st.session_state.video_responses = {}
        st.session_state.recording_started = False
        st.session_state.recording_time = 0
        st.session_state.video_files = {}
    
    current_step = steps[st.session_state.current_step]
    
    # Display current step
    st.markdown(f'<div class="step-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="step-header">Step {st.session_state.current_step + 1}: {current_step["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-timer">{current_step["duration"]} seconds</div>', unsafe_allow_html=True)
    st.markdown(f'<p>{current_step["prompt"]}</p>', unsafe_allow_html=True)
    
    # Skip button for optional steps
    if current_step.get("optional", False):
        if st.button("Skip this step", key=f"skip_{current_step['key']}"):
            st.session_state.video_responses[current_step['key']] = {"status": "skipped"}
            if st.session_state.current_step < len(steps) - 1:
                st.session_state.current_step += 1
            st.rerun()
    
    # Video recording
    video_file = st.camera_input(f"Record your video for: {current_step['title']}", 
                                 key=f"camera_{current_step['key']}")
    
    if video_file:
        st.session_state.video_files[current_step['key']] = video_file
    
    # Recording controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Recording", key="save_recording") and current_step['key'] in st.session_state.video_files:
            st.session_state.video_responses[current_step['key']] = {
                "status": "recorded",
                "video": st.session_state.video_files[current_step['key']]
            }
            st.success("Recording saved!")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.current_step > 0:
            if st.button("Previous Step", key="prev_step"):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col2:
        if st.session_state.current_step < len(steps) - 1:
            if st.button("Next Step", key="next_step"):
                # Ensure current step is completed or skipped
                if current_step['key'] not in st.session_state.video_responses:
                    st.warning("Please record or skip this step before proceeding")
                else:
                    st.session_state.current_step += 1
                    st.rerun()
        else:
            if st.button("Complete Profile", key="complete_profile"):
                if current_step['key'] not in st.session_state.video_responses:
                    st.warning("Please complete or skip this step before submitting")
                else:
                    # Process all responses
                    process_video_responses()
                    st.success("Video profile completed successfully!")
                    st.session_state.current_step = 0
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close step-container

def process_video_responses():
    """Process video responses and generate transcript"""
    user_id = st.session_state.current_user
    profile = st.session_state.users[user_id]['profile_data']
    profiler = ApprenticeProfiler()
    
    transcript_parts = []
    video_files = []
    
    with st.spinner("Processing your video profile..."):
        progress_bar = st.progress(0)
        total_steps = len(st.session_state.video_responses)
        
        for i, (key, response) in enumerate(st.session_state.video_responses.items()):
            if response['status'] == 'recorded' and 'video' in response:
                video_file = response['video']
                
                # Save video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(video_file.read())
                    video_path = tmp_file.name
                
                # Extract audio
                audio_path = profiler.extract_audio_from_video(video_path)
                
                if audio_path:
                    # Transcribe audio
                    transcript = profiler.transcribe_audio(audio_path)
                    transcript_parts.append(transcript)
                    
                    # Clean up files
                    os.unlink(video_path)
                    os.unlink(audio_path)
                
                # Save video file reference
                video_files.append(video_file)
            
            # Update progress
            progress = (i + 1) / total_steps
            progress_bar.progress(progress)
    
    full_transcript = " ".join(transcript_parts)
    
    # Extract candidate info
    candidate_info = profiler.extract_candidate_info(full_transcript)
    
    # Analyze personality
    personality = profiler.analyze_personality(full_transcript)
    
    # Update profile
    profile.update({
        'video_profile': {
            'transcript': full_transcript,
            'videos': video_files,
            'processed_at': datetime.now()
        },
        'personality_traits': personality
    })
    
    # Update extracted info if available
    if candidate_info.get('name'):
        profile['name'] = candidate_info['name']
    if candidate_info.get('skills'):
        profile['skills'] = list(set(profile.get('skills', []) + candidate_info['skills']))
    
    st.session_state.users[user_id]['profile_data'] = profile
    st.session_state.apprentices[user_id] = profile

# Psychometric Test
def psychometric_test():
    # Get current user's profile
    user_id = st.session_state.current_user
    profile = st.session_state.users[user_id]['profile_data']
    
    st.subheader("🧠 Psychometric Assessment")
    st.info("Complete this test to help us understand your personality traits and work preferences")
    
    # Load questions
    if 'questions' not in st.session_state:
        st.session_state.questions = []
        if 'questions' in st.session_state.get('system_config', {}):
            st.session_state.questions = st.session_state.system_config['questions']
        else:
            # Load sample questions if not configured
            add_sample_questions()
    
    # Initialize responses
    if 'test_responses' not in st.session_state:
        st.session_state.test_responses = {q['id']: None for q in st.session_state.questions}
        st.session_state.test_completed = False
    
    # Display progress
    completed = sum(1 for r in st.session_state.test_responses.values() if r is not None)
    total = len(st.session_state.questions)
    progress = int(100 * completed / total)
    
    st.markdown(f"**Progress:** {completed}/{total} questions ({progress}%)")
    st.progress(progress / 100)
    
    # Display questions
    for question in st.session_state.questions:
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.markdown(f"**{question['id']}. {question['text']}**")
        
        options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        response = st.radio(
            f"Response for question {question['id']}", 
            options,
            key=f"q_{question['id']}",
            index=2 if st.session_state.test_responses[question['id']] is None else st.session_state.test_responses[question['id']],
            horizontal=True
        )
        
        # Store response
        if response:
            st.session_state.test_responses[question['id']] = options.index(response)
        st.markdown('</div>', unsafe_allow_html=True)  # Close question-card
    
    # Submit button
    if st.button("Submit Test", type="primary", disabled=st.session_state.test_completed):
        # Calculate scores
        trait_scores = {
            'openness': 0,
            'conscientiousness': 0,
            'extraversion': 0,
            'agreeableness': 0,
            'neuroticism': 0
        }
        trait_counts = {t: 0 for t in trait_scores}
        
        for question in st.session_state.questions:
            response = st.session_state.test_responses[question['id']]
            if response is not None:
                trait = question['trait']
                score = response if not question.get('reverse', False) else 4 - response
                trait_scores[trait] += score
                trait_counts[trait] += 1
        
        # Calculate average scores (0-1 scale)
        personality = {}
        for trait, total_score in trait_scores.items():
            if trait_counts[trait] > 0:
                personality[trait] = total_score / (trait_counts[trait] * 4)  # Normalize to 0-1
        
        # Update profile
        profile = st.session_state.users[user_id]['profile_data']
        profile['psychometric_profile'] = {
            'scores': personality,
            'completed_at': datetime.now(),
            'responses': st.session_state.test_responses
        }
        
        # If personality traits not set, use psychometric results
        if 'personality_traits' not in profile:
            profile['personality_traits'] = personality
        
        st.session_state.users[user_id]['profile_data'] = profile
        st.session_state.apprentices[user_id] = profile
        st.session_state.test_completed = True
        st.success("Test submitted successfully! Your results have been saved.")
    
    # Display results if completed
    if st.session_state.test_completed and 'psychometric_profile' in profile:
        st.subheader("Your Psychometric Profile")
        
        # Radar chart
        fig = create_personality_radar_chart(profile['psychometric_profile']['scores'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores
        st.markdown("### Detailed Scores")
        col1, col2, col3, col4, col5 = st.columns(5)
        traits = {
            'openness': "Openness to Experience",
            'conscientiousness': "Conscientiousness",
            'extraversion': "Extraversion",
            'agreeableness': "Agreeableness",
            'neuroticism': "Neuroticism"
        }
        
        for trait, col in zip(traits, [col1, col2, col3, col4, col5]):
            score = profile['psychometric_profile']['scores'].get(trait, 0)
            col.metric(traits[trait], f"{score*100:.1f}%")
            
            # Progress bar
            col.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {score*100}%"></div></div>', unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("### Interpretation")
        st.markdown("""
        - **Openness to Experience**: Reflects your imagination, curiosity, and creativity.
        - **Conscientiousness**: Measures your organization, dependability, and discipline.
        - **Extraversion**: Indicates your sociability, assertiveness, and energy levels.
        - **Agreeableness**: Shows your compassion, cooperation, and trust in others.
        - **Neuroticism**: Reflects your emotional stability and resilience to stress.
        """)

# Advanced Profiling Function
def advanced_profiling(groq_api_key=None):
    st.subheader("🧠 Advanced Profile Analysis")
    st.info("Use our AI-powered tools to get deep insights into your skills and personality")
    
    # Initialize the profiler
    if 'profiler' not in st.session_state:
        st.session_state.profiler = ApprenticeProfiler()
    
    profiler = st.session_state.profiler
    user_id = st.session_state.current_user
    profile = st.session_state.users[user_id]['profile_data']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📥 Input Methods")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["🎥 Record Video Interview", "📁 Upload File", "📝 Direct Text Input"],
            help="Select how you want to provide candidate information"
        )
        
        transcript = ""
        
        if input_method == "🎥 Record Video Interview":
            st.markdown("**Record video interview directly:**")
            
            # Enhanced video recording interface
            st.markdown("""
            <div class="video-container">
                <h4>📹 Video Interview Recording</h4>
                <p>Click the button below to start recording the candidate interview.</p>
                <p><strong>Tips for best results:</strong></p>
                <ul>
                    <li>Ensure good lighting and clear audio</li>
                    <li>Keep the candidate in frame</li>
                    <li>Record for 2-5 minutes for optimal analysis</li>
                    <li>Ask about background, skills, and career goals</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Video recording widget
            video_file = st.camera_input("🎬 Start Video Recording")
            
            if video_file is not None:
                # Display recorded video
                st.video(video_file)
                
                # Add model selection for speed/accuracy tradeoff
                model_size = st.radio("Transcription Model", 
                                    ["⚡ Fast (Lower Accuracy)", "🧠 Accurate (Slower)"],
                                    horizontal=True)
                
                # Processing options
                col_a, col_b = st.columns(2)
                with col_a:
                    process_video = st.button("🔄 Process Video", type="primary")
                
                if process_video:
                    # Save uploaded video
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_file.read())
                        video_path = tmp_file.name
                    
                    with st.status("Processing video interview...", expanded=True) as status:
                        st.write("🔊 Extracting audio from video...")
                        audio_path = profiler.extract_audio_from_video(video_path)
                        
                        if audio_path and os.path.exists(audio_path):
                            st.write("📝 Transcribing speech to text...")
                            
                            # Switch model based on selection
                            if model_size == "⚡ Fast (Lower Accuracy)":
                                original_model = profiler.models.get('whisper')
                                try:
                                    # Use faster model
                                    profiler.models['whisper'] = pipeline(
                                        "automatic-speech-recognition",
                                        model="openai/whisper-tiny",
                                        device=-1
                                    )
                                except:
                                    pass
                            
                            transcript = profiler.transcribe_audio(audio_path)
                            
                            # Restore original model
                            if model_size == "⚡ Fast (Lower Accuracy)":
                                profiler.models['whisper'] = original_model
                            
                            if transcript and len(transcript.strip()) > 10:
                                status.update(label="Processing complete!", state="complete")
                                
                                # Save transcript to profile
                                profile['advanced_transcript'] = transcript
                                st.session_state.users[user_id]['profile_data'] = profile
                                st.session_state.apprentices[user_id] = profile
                                
                                # Show preview of transcript
                                with st.expander("👀 Preview Transcript"):
                                    st.text_area("Transcript Preview", transcript[:500] + "...", height=100, disabled=True)
                            else:
                                status.update(label="Processing failed", state="error")
                                st.error("❌ Could not transcribe audio. Please check audio quality or try manual input.")
                            
                            # Clean up audio file
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
                        else:
                            status.update(label="Processing failed", state="error")
                            st.error("❌ Could not extract audio from video.")
                        
                        # Clean up video file
                        try:
                            os.unlink(video_path)
                        except:
                            pass
        
        elif input_method == "📁 Upload File":
            st.markdown("**Upload candidate files:**")
            
            uploaded_file = st.file_uploader(
                "Choose file to upload",
                type=['mp3', 'wav', 'mp4', 'mov', 'webm', 'pdf', 'txt'],
                help="Supported formats: Audio (MP3, WAV), Video (MP4, MOV, WEBM), Documents (PDF, TXT)"
            )
            
            if uploaded_file is not None:
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type
                }
                
                st.success(f"📁 File uploaded: {file_info['name']} ({file_info['size']} bytes)")
                
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Process button
                process_file = st.button("🔄 Process File", type="primary")
                
                if process_file:
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        file_path = tmp_file.name
                    
                    with st.spinner(f"Processing {file_extension.upper()} file..."):
                        progress_bar = st.progress(0)
                        
                        if file_extension == 'pdf':
                            st.write("📄 Extracting text from PDF...")
                            transcript = profiler.extract_pdf_text(file_path)
                            progress_bar.progress(100)
                            
                        elif file_extension in ['mp3', 'wav']:
                            st.write("🔊 Transcribing audio file...")
                            progress_bar.progress(50)
                            transcript = profiler.transcribe_audio(file_path)
                            progress_bar.progress(100)
                            
                        elif file_extension in ['mp4', 'mov', 'webm']:
                            st.write("🔊 Extracting audio from video...")
                            progress_bar.progress(33)
                            audio_path = profiler.extract_audio_from_video(file_path)
                            
                            if audio_path:
                                progress_bar.progress(66)
                                st.write("📝 Transcribing audio...")
                                transcript = profiler.transcribe_audio(audio_path)
                                progress_bar.progress(100)
                                
                                # Clean up audio file
                                try:
                                    os.unlink(audio_path)
                                except:
                                    pass
                            else:
                                st.error("Could not extract audio from video")
                                
                        elif file_extension == 'txt':
                            st.write("📄 Reading text file...")
                            with open(file_path, 'r', encoding='utf-8') as f:
                                transcript = f.read()
                            progress_bar.progress(100)
                    
                    if transcript and len(transcript.strip()) > 10:
                        st.success("✅ File processed successfully!")
                        
                        # Save transcript to profile
                        profile['advanced_transcript'] = transcript
                        st.session_state.users[user_id]['profile_data'] = profile
                        st.session_state.apprentices[user_id] = profile
                        
                        # Show preview
                        with st.expander("👀 Preview Content"):
                            st.text_area("Content Preview", transcript[:500] + "...", height=100, disabled=True)
                    else:
                        st.error("❌ Could not process file or extract meaningful content")
                    
                    # Clean up
                    try:
                        os.unlink(file_path)
                    except:
                        pass
        
        else:  # Direct text input
            st.markdown("**Enter candidate information manually:**")
            
            transcript = st.text_area(
                "Candidate Information",
                height=200,
                placeholder="""Enter candidate details such as:
• Resume/CV content
• Interview transcript
• Background information
• Skills and experience
• Education details
• Career goals

Example: "Hi, I'm John Smith, a software engineer with 5 years of experience in Python and React. I have a Bachelor's degree in Computer Science from MIT and have worked at Google and Microsoft. I'm passionate about machine learning and looking for senior developer roles..."
""",
                help="Paste resume text, interview transcript, or any candidate information here"
            )
            
            if st.button("Save Transcript", key="save_transcript"):
                if transcript and len(transcript.strip()) > 20:
                    profile['advanced_transcript'] = transcript
                    st.session_state.users[user_id]['profile_data'] = profile
                    st.session_state.apprentices[user_id] = profile
                    st.success("Transcript saved successfully!")
                else:
                    st.error("Please enter at least 20 characters")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'advanced_transcript' in profile and profile['advanced_transcript']:
            transcript = profile['advanced_transcript']
            
            # Process the transcript
            with st.spinner("🧠 Analyzing candidate profile..."):
                analysis_progress = st.progress(0)
                
                # Step 1: Extract candidate information
                st.write("🔍 Extracting candidate information...")
                candidate_data = profiler.extract_candidate_info(transcript)
                analysis_progress.progress(33)
                
                # Step 2: Analyze personality
                st.write("🧠 Analyzing personality traits...")
                personality = profiler.analyze_personality(transcript)
                analysis_progress.progress(66)
                
                # Step 3: Generate insights
                st.write("💡 Generating AI insights...")
                insights = profiler.generate_insights(
                    candidate_data, transcript, personality, groq_api_key
                )
                analysis_progress.progress(100)
            
            st.success("✅ Analysis completed!")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs([
                "👤 Profile", "🧠 Personality", "🎯 Insights"
            ])
            
            with tab1:
                st.subheader("👤 Candidate Profile")
                
                # Personal Information
                if any(candidate_data.get(key) for key in ['name', 'email', 'phone']):
                    st.markdown("### 📇 Contact Information")
                    
                    contact_col1, contact_col2 = st.columns(2)
                    
                    with contact_col1:
                        if candidate_data.get('name'):
                            st.write(f"**👤 Name:** {candidate_data['name']}")
                        if candidate_data.get('email'):
                            st.write(f"**📧 Email:** {candidate_data['email']}")
                    
                    with contact_col2:
                        if candidate_data.get('phone'):
                            st.write(f"**📱 Phone:** {candidate_data['phone']}")
                
                # Skills
                if candidate_data.get('skills'):
                    st.markdown("### 💼 Skills")
                    
                    # Display skills as tags
                    skills_html = ""
                    for skill in candidate_data['skills'][:15]:  # Limit to 15 skills
                        skills_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 2px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 12px;">{skill}</span> '
                    
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    if len(candidate_data['skills']) > 15:
                        st.caption(f"... and {len(candidate_data['skills']) - 15} more skills")
                    
                    # Skills chart
                    skills_chart = create_skills_chart(candidate_data['skills'])
                    if skills_chart:
                        st.plotly_chart(skills_chart, use_container_width=True)
                
                # Education
                if candidate_data.get('education'):
                    st.markdown("### 🎓 Education")
                    st.write(candidate_data['education'])
                
                # Experience
                if candidate_data.get('experience'):
                    st.markdown("### 💼 Experience")
                    st.write(candidate_data['experience'])
            
            with tab2:
                st.subheader("🧠 Personality Analysis")
                
                # Create two columns for chart and details
                chart_col, details_col = st.columns([3, 2])
                
                with chart_col:
                    # Radar chart
                    fig = create_personality_radar_chart(personality)
                    st.plotly_chart(fig, use_container_width=True)
                
                with details_col:
                    st.markdown("### 📊 OCEAN Scores")
                    
                    # Sort traits by score
                    sorted_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)
                    
                    for trait, score in sorted_traits:
                        # Determine level and color
                        if score > 0.7:
                            level = "High"
                            color = "green"
                        elif score > 0.5:
                            level = "Above Average"
                            color = "blue"
                        elif score > 0.3:
                            level = "Average"
                            color = "orange"
                        else:
                            level = "Below Average"
                            color = "red"
                        
                        st.metric(
                            trait.title(), 
                            f"{score:.2f}",
                            delta=f"{level}",
                            help=f"Score range: 0.0 (Low) to 1.0 (High)"
                        )
                
                # Personality interpretation
                st.markdown("### 🎯 Personality Insights")
                
                dominant_trait = max(personality.items(), key=lambda x: x[1])
                weakest_trait = min(personality.items(), key=lambda x: x[1])
                
                trait_descriptions = {
                    'openness': "Creative, curious, and open to new experiences. Enjoys exploring ideas and trying new approaches.",
                    'conscientiousness': "Organized, responsible, and detail-oriented. Values structure and thorough planning.",
                    'extraversion': "Outgoing, energetic, and socially confident. Draws energy from interaction with others.",
                    'agreeableness': "Cooperative, empathetic, and team-oriented. Values harmony and collaboration.",
                    'neuroticism': "Emotionally sensitive and reactive to stress. May need additional support in high-pressure situations."
                }
                
                st.write(f"**🔸 Dominant Trait:** {dominant_trait[0].title()} ({dominant_trait[1]:.2f})")
                st.write(trait_descriptions.get(dominant_trait[0], ""))
                
                if weakest_trait[1] < 0.4:
                    st.write(f"**🔹 Development Area:** {weakest_trait[0].title()} ({weakest_trait[1]:.2f})")
                    st.write(f"Consider developing {weakest_trait[0]} skills for well-rounded growth.")
            
            with tab3:
                st.subheader("🎯 AI-Generated Insights")
                
                # Display insights
                st.markdown(insights)
        
        else:
            st.info("👆 Please provide candidate information to begin analysis")

# Initialize AI models
@st.cache_resource
def load_models():
    """Load all AI models"""
    models = {}
    
    try:
        # Load Whisper model for speech recognition
        try:
            models['whisper'] = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-base",
                device=-1  # Use CPU
            )
        except Exception as e:
            st.warning(f"⚠️ Could not load Whisper model: {str(e)}")
            models['whisper'] = None
        
        # Load NER model for entity extraction
        try:
            models['ner'] = pipeline(
                "ner", 
                model="dslim/bert-base-NER", 
                aggregation_strategy="simple",
                device=-1
            )
        except Exception as e:
            st.warning(f"⚠️ Could not load NER model: {str(e)}")
            models['ner'] = None
        
        # Load sentiment analysis model
        try:
            models['sentiment'] = pipeline(
                "sentiment-analysis", 
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1
            )
        except Exception as e:
            st.warning(f"⚠️ Could not load sentiment model: {str(e)}")
            models['sentiment'] = None
        
        # Load psychometric model
        try:
            model_name="KevSun/Personality_LM"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            models['psychometric'] = {"tokenizer": tokenizer, "model": model}
        
        except Exception as e:
            st.warning(f"⚠️ Could not load psychometric model: {str(e)}")
            models['psychometric'] = None

        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

# App Pages
def login_page():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Apprentice Matching Platform</h1>
        <p>Advanced AI-powered apprentice profiling and matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.subheader("🔐 Login to Your Account")
            
            with st.form("login_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    user_id, user_data = authenticate_user(email, password)
                    if user_id:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_id
                        st.session_state.user_type = user_data['user_type']
                        st.success("Login successful! Redirecting to dashboard...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
            
            st.markdown("---")
            st.markdown("Don't have an account?")
            if st.button("Register Now"):
                st.session_state.page = "register"
                st.rerun()

def register_page():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Apprentice Matching Platform</h1>
        <p>Create your account to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("Create Your Account")
            
            user_type = st.selectbox("I am a...", [
                "Apprentice", "Company", "Training Provider", "Admin"
            ]).lower().replace(" ", "_")
            
            with st.form("register_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                # Basic profile info
                st.subheader("Profile Information")
                if user_type == "apprentice":
                    name = st.text_input("Full Name")
                    age = st.number_input("Age", min_value=16, max_value=30, value=18)
                    location = st.selectbox("Location", st.session_state.system_config['cities']['UK'])
                    education = st.selectbox("Education Level", ["High School", "A-levels", "BTEC", "Diploma", "Bachelor"])
                    skills = st.text_input("Skills (comma separated)")
                    languages = st.multiselect("Languages Spoken", st.session_state.system_config['languages'])
                elif user_type == "company":
                    company_name = st.text_input("Company Name")
                    industry = st.selectbox("Industry", [
                        "Technology", "Healthcare", "Manufacturing", "Finance", 
                        "Retail", "Construction", "Automotive", "Other"
                    ])
                    location = st.selectbox("Location", st.session_state.system_config['cities']['UK'])
                    description = st.text_area("Company Description")
                elif user_type == "training_provider":
                    provider_name = st.text_input("Provider Name")
                    accreditation = st.text_input("Accreditation")
                    location = st.selectbox("Location", st.session_state.system_config['cities']['UK'])
                    courses = st.text_area("Courses Offered (one per line)")
                
                submitted = st.form_submit_button("Create Account")
                
                if submitted:
                    if password != confirm_password:
                        st.error("Passwords do not match!")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        # Create profile data based on user type
                        profile_data = {}
                        if user_type == 'apprentice':
                            profile_data = {
                                'name': name, 
                                'age': age, 
                                'location': location,
                                'education': education,
                                'skills': [s.strip() for s in skills.split(',')] if skills else [],
                                'languages': languages,
                                'video_profile': None,
                                'personality_traits': {},
                                'availability': True
                            }
                        elif user_type == 'company':
                            profile_data = {
                                'name': company_name, 
                                'industry': industry,
                                'location': location, 
                                'description': description,
                                'job_postings': [],
                                'subscription_tier': 'basic'
                            }
                        elif user_type == 'training_provider':
                            profile_data = {
                                'name': provider_name, 
                                'accreditation': accreditation,
                                'location': location, 
                                'courses': courses.split('\n') if courses else [],
                                'subscription_tier': 'basic'
                            }
                        else:
                            profile_data = {'role': 'admin'}
                        
                        user_id = create_user(email, password, user_type, profile_data)
                        st.success(f"Account created successfully! Welcome to the platform.")
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_id
                        st.session_state.user_type = user_type
                        time.sleep(1)
                        st.rerun()
    
    with col2:
        st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", 
                 caption="Start your journey with us today")
        st.markdown("### Why Join Our Platform?")
        st.markdown("""
        - 🤝 Connect with top companies and training providers
        - 🚀 Find the perfect apprenticeship opportunity
        - 🧠 AI-powered personality and skill analysis
        - 📊 Comprehensive career insights and recommendations
        - 💼 Build your professional profile with video introductions
        - 🌍 Available in multiple languages
        - 📈 Take psychometric tests to showcase your strengths
        """)

def apprentice_dashboard(models, groq_api_key=None):
    user_data = st.session_state.users[st.session_state.current_user]
    profile = user_data['profile_data']
    
    st.markdown(f"<h1 style='text-align: center;'>👤 Welcome, {profile.get('name', 'Apprentice')}</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Dashboard", "Profile", "Video Profile", "Opportunities", "AI Insights", "Psychometric Test", "Advanced Profile"
    ])
    
    with tab1:
        st.subheader("Your Dashboard")
        
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            profile_completeness = 75 if profile.get('video_profile') else 35
            st.metric("Profile Completeness", f"{profile_completeness}%")
            st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {profile_completeness}%"></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Opportunities", "12")
            st.markdown("In your area")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Match Score", "92%")
            st.markdown("Average")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommended opportunities
        st.subheader("Recommended Opportunities")
        job_opportunities = list(st.session_state.job_postings.values())[:3]
        
        for job in job_opportunities:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"#### {job['title']}")
                    st.write(f"**Company:** {st.session_state.companies[job['company']]['name']} | **Location:** {job['location']}")
                    
                    # Skills
                    skills_html = ""
                    for skill in job['skills'][:3]:
                        skills_html += f'<span class="skill-tag">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"<div style='font-size: 2rem; font-weight: bold; color: #28a745;'>{job['match_score']}%</div>", unsafe_allow_html=True)
                    st.write("Match Score")
                    
                    if st.button("View Details", key=f"view_{job['id']}"):
                        st.session_state.current_job = job['id']
                
                st.markdown("---")
        
        # Personality profile
        if profile.get('personality_traits'):
            st.subheader("Your Personality Profile")
            fig = create_personality_radar_chart(profile['personality_traits'])
            st.plotly_chart(fig, use_container_width=True, key="personality_chart")
    
    with tab2:
        st.subheader("Your Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form("profile_form"):
                name = st.text_input("Full Name", value=profile.get('name', ''))
                age = st.number_input("Age", min_value=16, max_value=30, 
                                    value=int(profile.get('age', 18)))
                location = st.selectbox("Location", st.session_state.system_config['cities']['UK'], 
                                      index=0 if not profile.get('location') else 
                                      st.session_state.system_config['cities']['UK'].index(profile.get('location', 'London')))
                
                # Skills section
                st.subheader("Skills")
                current_skills = st.text_area("Your Skills (comma separated)", 
                                            value=', '.join(profile.get('skills', [])))
                
                # Education
                education = st.selectbox("Education Level", 
                                       ["High School", "A-levels", "BTEC", "Diploma", "Bachelor", "Master"],
                                       index=0 if not profile.get('education') else 
                                       ["High School", "A-levels", "BTEC", "Diploma", "Bachelor", "Master"].index(profile.get('education', 'High School')))
                
                # Languages
                languages = st.multiselect("Languages Spoken", st.session_state.system_config['languages'],
                                          default=profile.get('languages', []))
                
                # Availability
                availability = st.checkbox("Available for opportunities", value=profile.get('availability', True))
                
                if st.form_submit_button("Update Profile"):
                    skills_list = [skill.strip() for skill in current_skills.split(',') if skill.strip()]
                    
                    profile.update({
                        'name': name, 'age': age, 'location': location,
                        'skills': skills_list, 'education': education,
                        'languages': languages, 'availability': availability
                    })
                    
                    st.session_state.users[st.session_state.current_user]['profile_data'] = profile
                    st.session_state.apprentices[st.session_state.current_user] = profile
                    st.success("Profile updated successfully!")
        
        with col2:
            st.subheader("Profile Summary")
            if profile.get('video_profile'):
                st.success("✅ Video Profile Completed")
                with st.expander("View Transcript"):
                    st.write(profile['video_profile']['transcript'][:500] + "...")
            else:
                st.warning("❌ Video Profile Not Completed")
                st.info("Complete your video profile to get better matches!")
            
            # Display personality traits if available
            if profile.get('personality_traits'):
                st.subheader("Personality Profile")
                traits = profile['personality_traits']
                
                fig = create_personality_radar_chart(traits)
                st.plotly_chart(fig, use_container_width=True, key="personality_radar")
            
            # Skills chart
            if profile.get('skills'):
                st.subheader("Skills Distribution")
                skills_chart = create_skills_chart(profile['skills'])
                if skills_chart:
                    st.plotly_chart(skills_chart, use_container_width=True, key="skills_chart")
            
            # Documents section
            st.subheader("Your Documents")
            st.info("Upload your CV, certificates, or other documents")
            uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
            if uploaded_files:
                st.success(f"{len(uploaded_files)} documents uploaded successfully")
    
    with tab3:
        guided_video_recording()
    
    with tab4:
        st.subheader("🔍 Find Opportunities")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            location_filter = st.selectbox("Location", ["All"] + st.session_state.system_config['cities']['UK'])
        with col2:
            industry_filter = st.selectbox("Industry", ["All", "Technology", "Healthcare", "Marketing", "Finance", "Other"])
        
        # Display opportunities
        job_opportunities = list(st.session_state.job_postings.values())
        
        if location_filter != "All":
            job_opportunities = [job for job in job_opportunities if job['location'] == location_filter]
        
        if industry_filter != "All":
            job_opportunities = [job for job in job_opportunities if st.session_state.companies[job['company']]['industry'] == industry_filter]
        
        for job in job_opportunities:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {job['title']}")
                    st.write(f"**Company:** {st.session_state.companies[job['company']]['name']} | **Location:** {job['location']}")
                    
                    # Skills
                    skills_html = ""
                    for skill in job['skills']:
                        skills_html += f'<span class="skill-tag">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    with st.expander("View Details"):
                        st.write(job['description'])
                
                with col2:
                    st.markdown(f"<div style='font-size: 2rem; font-weight: bold; color: #28a745;'>{job['match_score']}%</div>", unsafe_allow_html=True)
                    st.write("Match Score")
                
                with col3:
                    if st.button(f"Apply", key=f"apply_{job['id']}"):
                        st.success("Application submitted successfully!")
                    if st.button(f"Save", key=f"save_{job['id']}"):
                        st.info("Opportunity saved to your profile")
                
                st.markdown("---")
        
        # Training providers
        st.subheader("Training Providers")
        training_providers = list(st.session_state.training_providers.values())
        
        for provider in training_providers:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {provider['name']}")
                    st.write(f"**Location:** {provider['location']} | **Accreditation:** {provider['accreditation']}")
                    
                    # Courses
                    courses_html = ""
                    for course in provider['courses'][:3]:
                        courses_html += f'<span class="skill-tag">{course}</span> '
                    st.markdown(courses_html, unsafe_allow_html=True)
                
                with col2:
                    if st.button(f"Contact", key=f"contact_{provider['id']}"):
                        st.session_state.message_recipient = provider['id']
                        st.session_state.message_recipient_type = "training_provider"
                        st.rerun()
                
                st.markdown("---")
    
    with tab5:
        st.subheader("🧠 AI Career Insights")
        
        if profile.get('video_profile') and profile.get('personality_traits'):
            transcript = profile['video_profile']['transcript']
            candidate_info = {
                'name': profile.get('name', ''),
                'skills': profile.get('skills', []),
                'education': profile.get('education', ''),
                'experience': '',
                'email': '',
                'phone': ''
            }
            personality = profile['personality_traits']
            
            # Generate insights
            insights = ApprenticeProfiler().generate_insights(candidate_info, transcript, personality, groq_api_key)
            
            # Display insights
            st.markdown(insights, unsafe_allow_html=True)
            
            # Personality analysis
            st.subheader("Personality Analysis")
            fig = create_personality_radar_chart(personality)
            st.plotly_chart(fig, use_container_width=True, key="insight_radar")
            
            # Psychometric results comparison
            if profile.get('psychometric_profile'):
                st.subheader("Psychometric Test Results")
                st.markdown(f"**Test Date:** {profile['psychometric_profile']['completed_at'].strftime('%Y-%m-%d')}")
                
                # Compare with video-based personality analysis
                st.markdown("### Comparison with Video Analysis")
                
                # Create a comparison chart
                traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                video_scores = [personality.get(t, 0) for t in traits]
                test_scores = [profile['psychometric_profile']['scores'].get(t, 0) for t in traits]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=video_scores,
                    theta=[t.title() for t in traits],
                    fill='toself',
                    name='Video Analysis',
                    fillcolor='rgba(78, 121, 167, 0.3)',
                    line=dict(color='rgb(78, 121, 167)'),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatterpolar(
                    r=test_scores,
                    theta=[t.title() for t in traits],
                    fill='toself',
                    name='Psychometric Test',
                    fillcolor='rgba(242, 142, 43, 0.3)',
                    line=dict(color='rgb(242, 142, 43)'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickmode='array',
                            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                            ticktext=['Low', 'Below Avg', 'Average', 'Above Avg', 'High']
                        )),
                    showlegend=True,
                    title="Personality Analysis Comparison",
                    font=dict(size=12),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please complete your video profile to generate AI insights")
    
    with tab6:
        psychometric_test()
        
    with tab7:
        advanced_profiling(groq_api_key)

def company_dashboard():
    user_data = st.session_state.users[st.session_state.current_user]
    profile = user_data['profile_data']
    
    st.markdown(f"<h1 style='text-align: center;'>🏢 Welcome, {profile.get('name', 'Company')}</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Browse Apprentices", "Subscriptions", "Messages", "Training Providers"
    ])
    
    with tab1:
        st.subheader("Company Dashboard")
        
        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Job Postings", len(profile.get('job_postings', [])))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Applications Received", "48")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Interviews Scheduled", "15")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Apprentices Hired", "5")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Applications")
            applications = [
                {"name": "Sarah Johnson", "position": "Marketing Apprentice", "score": 92, "status": "Under Review"},
                {"name": "Ahmed Khan", "position": "Software Developer", "score": 88, "status": "Interview Scheduled"},
                {"name": "Priya Patel", "position": "Admin Assistant", "score": 85, "status": "Application Received"},
            ]
            
            for app in applications:
                with st.container():
                    st.write(f"**{app['name']}** - {app['position']}")
                    st.write(f"Match Score: {app['score']}% | Status: {app['status']}")
                    st.markdown("---")
        
        with col2:
            st.subheader("Application Trends")
            dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
            applications = np.random.randint(5, 25, 12)
            
            fig = px.line(x=dates, y=applications, title="Monthly Applications")
            fig.update_layout(xaxis_title="Month", yaxis_title="Applications")
            st.plotly_chart(fig, use_container_width=True)
            
            # Add job posting button
            if st.button("➕ Create New Job Posting"):
                st.session_state.create_job = True
                st.rerun()
    
    with tab2:
        st.subheader("Browse Apprentices")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_filter = st.selectbox("Location", ["All"] + st.session_state.system_config['cities']['UK'])
            
        with col2:
            education_filter = st.selectbox("Education Level", ["All", "High School", "A-levels", "BTEC", "Diploma"])
            
        with col3:
            age_filter = st.slider("Age Range", 16, 30, (18, 25))
        
        skills_filter = st.text_input("Skills (comma separated)")
        
        st.markdown("---")
        
        # Display profiles
        apprentices = list(st.session_state.apprentices.values())
        
        # Apply filters
        if location_filter != "All":
            apprentices = [app for app in apprentices if app['location'] == location_filter]
        
        if education_filter != "All":
            apprentices = [app for app in apprentices if app['education'] == education_filter]
        
        apprentices = [app for app in apprentices if age_filter[0] <= app['age'] <= age_filter[1]]
        
        if skills_filter:
            skills_list = [s.strip().lower() for s in skills_filter.split(',')]
            apprentices = [app for app in apprentices if any(skill.lower() in [s.lower() for s in app['skills']] for skill in skills_list)]
        
        # Show results
        for app in apprentices:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"### {app['name']}")
                    st.write(f"**Age:** {app['age']} | **Location:** {app['location']}")
                    st.write(f"**Education:** {app['education']}")
                    
                    # Skills
                    skills_html = ""
                    for skill in app['skills'][:5]:
                        skills_html += f'<span class="skill-tag">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                with col2:
                    match_score = np.random.randint(70, 95)
                    st.markdown(f"<div style='font-size: 2rem; font-weight: bold; color: #28a745;'>{match_score}%</div>", unsafe_allow_html=True)
                    st.write("Match Score")
                    
                    # Progress bar for match
                    st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {match_score}%"></div></div>', unsafe_allow_html=True)
                
                with col3:
                    if st.button(f"View Profile", key=f"view_{app['id']}"):
                        st.session_state.view_apprentice = app['id']
                    if st.button(f"Contact", key=f"contact_{app['id']}"):
                        st.session_state.message_recipient = app['id']
                        st.session_state.message_recipient_type = "apprentice"
                        st.rerun()
                
                st.markdown("---")
    
    with tab3:
        st.subheader("Subscription Management")
        st.info("Upgrade your subscription to access more features and candidates")
        
        plans = st.session_state.system_config['subscription_plans']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="subscription-card basic">', unsafe_allow_html=True)
                st.markdown(f"### {plans['basic']['name']}")
                st.markdown(f"## £{plans['basic']['price']}/mo")
                st.markdown("---")
                for feature in plans['basic']['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
                if st.button("Select Basic", key="basic_plan"):
                    profile['subscription_tier'] = 'basic'
                    st.success("Basic plan selected!")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="subscription-card standard">', unsafe_allow_html=True)
                st.markdown(f"### {plans['standard']['name']}")
                st.markdown(f"## £{plans['standard']['price']}/mo")
                st.markdown("---")
                for feature in plans['standard']['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
                if st.button("Select Standard", key="standard_plan"):
                    profile['subscription_tier'] = 'standard'
                    st.success("Standard plan selected!")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="subscription-card professional">', unsafe_allow_html=True)
                st.markdown(f"### {plans['professional']['name']}")
                st.markdown(f"## £{plans['professional']['price']}/mo")
                st.markdown("---")
                for feature in plans['professional']['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
                if st.button("Select Professional", key="professional_plan"):
                    profile['subscription_tier'] = 'professional'
                    st.success("Professional plan selected!")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("💬 Your Messages")
        
        # Mock conversations
        conversations = [
            {"id": "conv1", "name": "Sarah Johnson", "last_message": "Thank you for the opportunity...", "time": "1 hour ago"},
            {"id": "conv2", "name": "Ahmed Khan", "last_message": "When is the interview scheduled?", "time": "3 hours ago"},
            {"id": "conv3", "name": "London Tech College", "last_message": "Partnership opportunity", "time": "2 days ago"},
        ]
        
        selected_conversation = st.selectbox("Select conversation", conversations, format_func=lambda x: x['name'])
        
        # Display messages
        st.markdown("### Conversation")
        st.markdown(f"**{selected_conversation['name']}**")
        st.write(selected_conversation['last_message'])
        st.caption(selected_conversation['time'])
        
        # Message history
        with st.expander("Message History"):
            st.markdown('<div class="chat-message other-message">Hello, I\'m interested in the apprenticeship position</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-message user-message">Thank you for your interest. Can you tell me more about your experience?</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message other-message">{selected_conversation["last_message"]}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        new_message = st.text_input("Type your message...")
        if st.button("Send"):
            st.success("Message sent!")
    
    with tab5:
        st.subheader("Collaborate with Training Providers")
        st.info("Connect with training providers to find qualified apprentices")
        
        # List training providers
        providers = list(st.session_state.training_providers.values())
        selected_provider = st.selectbox("Select Training Provider", providers, format_func=lambda p: p['name'])
        
        if selected_provider:
            st.markdown(f"### {selected_provider['name']}")
            st.write(f"**Location:** {selected_provider['location']}")
            st.write(f"**Accreditation:** {selected_provider['accreditation']}")
            
            # Display courses
            st.markdown("**Courses Offered:**")
            for course in selected_provider['courses'][:5]:
                st.markdown(f"- {course}")
            
            # Shortlisted apprentices
            if 'shortlisted_apprentices' in selected_provider:
                st.markdown("**Shortlisted Apprentices:**")
                for app_id in selected_provider['shortlisted_apprentices'][:5]:
                    if app_id in st.session_state.apprentices:
                        app = st.session_state.apprentices[app_id]
                        st.markdown(f"- {app['name']} ({app['location']})")
            
            # Action buttons
            if st.button("Request Apprentice Recommendations"):
                # In a real app, this would send a notification
                st.success(f"Request sent to {selected_provider['name']}")
            
            st.markdown("---")
            st.subheader("Send Apprentice to Training Provider")
            
            # Select apprentice to send
            apprentices = list(st.session_state.apprentices.values())
            selected_apprentice = st.selectbox("Select Apprentice", apprentices, format_func=lambda a: a['name'])
            
            if selected_apprentice and st.button("Send to Training Provider"):
                if 'shortlisted_apprentices' not in selected_provider:
                    selected_provider['shortlisted_apprentices'] = []
                if selected_apprentice['id'] not in selected_provider['shortlisted_apprentices']:
                    selected_provider['shortlisted_apprentices'].append(selected_apprentice['id'])
                    st.success(f"{selected_apprentice['name']} sent to {selected_provider['name']}")

def training_provider_dashboard():
    user_data = st.session_state.users[st.session_state.current_user]
    profile = user_data['profile_data']
    
    st.markdown(f"<h1 style='text-align: center;'>🏫 Welcome, {profile.get('name', 'Training Provider')}</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Browse Apprentices", "Subscriptions", "Messages", "Company Collaboration"
    ])
    
    with tab1:
        st.subheader("Training Provider Dashboard")
        
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Courses", len(profile.get('courses', [])))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Apprentices Enrolled", "48")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Placement Rate", "92%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Apprentices")
            apprentices = [
                {"name": "Sarah Johnson", "course": "Software Development", "status": "Active"},
                {"name": "Ahmed Khan", "course": "Digital Marketing", "status": "Graduated"},
                {"name": "Priya Patel", "course": "Healthcare Administration", "status": "Active"},
            ]
            
            for app in apprentices:
                with st.container():
                    st.write(f"**{app['name']}** - {app['course']}")
                    st.write(f"Status: {app['status']}")
                    st.markdown("---")
        
        with col2:
            st.subheader("Placement Trends")
            dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
            placements = np.random.randint(5, 15, 6)
            
            fig = px.line(x=dates, y=placements, title="Monthly Placements")
            fig.update_layout(xaxis_title="Month", yaxis_title="Placements")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Browse Apprentices")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_filter = st.selectbox("Location", ["All"] + st.session_state.system_config['cities']['UK'])
            
        with col2:
            education_filter = st.selectbox("Education Level", ["All", "High School", "A-levels", "BTEC", "Diploma"])
            
        with col3:
            age_filter = st.slider("Age Range", 16, 30, (18, 25))
        
        skills_filter = st.text_input("Skills (comma separated)")
        
        st.markdown("---")
        
        # Display profiles
        apprentices = list(st.session_state.apprentices.values())
        
        # Apply filters
        if location_filter != "All":
            apprentices = [app for app in apprentices if app['location'] == location_filter]
        
        if education_filter != "All":
            apprentices = [app for app in apprentices if app['education'] == education_filter]
        
        apprentices = [app for app in apprentices if age_filter[0] <= app['age'] <= age_filter[1]]
        
        if skills_filter:
            skills_list = [s.strip().lower() for s in skills_filter.split(',')]
            apprentices = [app for app in apprentices if any(skill.lower() in [s.lower() for s in app['skills']] for skill in skills_list)]
        
        # Show results
        for app in apprentices:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"### {app['name']}")
                    st.write(f"**Age:** {app['age']} | **Location:** {app['location']}")
                    st.write(f"**Education:** {app['education']}")
                    
                    # Skills
                    skills_html = ""
                    for skill in app['skills'][:5]:
                        skills_html += f'<span class="skill-tag">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                with col2:
                    match_score = np.random.randint(70, 95)
                    st.markdown(f"<div style='font-size: 2rem; font-weight: bold; color: #28a745;'>{match_score}%</div>", unsafe_allow_html=True)
                    st.write("Placement Score")
                    
                    # Progress bar for match
                    st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {match_score}%"></div></div>', unsafe_allow_html=True)
                
                with col3:
                    if st.button(f"View Profile", key=f"view_{app['id']}"):
                        st.session_state.view_apprentice = app['id']
                    if st.button(f"Contact", key=f"contact_{app['id']}"):
                        st.session_state.message_recipient = app['id']
                        st.session_state.message_recipient_type = "apprentice"
                        st.rerun()
                
                st.markdown("---")
    
    with tab3:
        st.subheader("Subscription Management")
        st.info("Upgrade your subscription to access more features and candidates")
        
        plans = st.session_state.system_config['subscription_plans']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="subscription-card basic">', unsafe_allow_html=True)
                st.markdown(f"### {plans['basic']['name']}")
                st.markdown(f"## £{plans['basic']['price']}/mo")
                st.markdown("---")
                for feature in plans['basic']['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
                if st.button("Select Basic", key="basic_plan"):
                    profile['subscription_tier'] = 'basic'
                    st.success("Basic plan selected!")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="subscription-card standard">', unsafe_allow_html=True)
                st.markdown(f"### {plans['standard']['name']}")
                st.markdown(f"## £{plans['standard']['price']}/mo")
                st.markdown("---")
                for feature in plans['standard']['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
                if st.button("Select Standard", key="standard_plan"):
                    profile['subscription_tier'] = 'standard'
                    st.success("Standard plan selected!")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="subscription-card professional">', unsafe_allow_html=True)
                st.markdown(f"### {plans['professional']['name']}")
                st.markdown(f"## £{plans['professional']['price']}/mo")
                st.markdown("---")
                for feature in plans['professional']['features']:
                    st.markdown(f"- {feature}")
                st.markdown("---")
                if st.button("Select Professional", key="professional_plan"):
                    profile['subscription_tier'] = 'professional'
                    st.success("Professional plan selected!")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("💬 Your Messages")
        
        # Mock conversations
        conversations = [
            {"id": "conv1", "name": "Sarah Johnson", "last_message": "Thank you for the opportunity...", "time": "1 hour ago"},
            {"id": "conv2", "name": "Ahmed Khan", "last_message": "When is the interview scheduled?", "time": "3 hours ago"},
            {"id": "conv3", "name": "Tech Innovations Ltd", "last_message": "Partnership opportunity", "time": "2 days ago"},
        ]
        
        selected_conversation = st.selectbox("Select conversation", conversations, format_func=lambda x: x['name'])
        
        # Display messages
        st.markdown("### Conversation")
        st.markdown(f"**{selected_conversation['name']}**")
        st.write(selected_conversation['last_message'])
        st.caption(selected_conversation['time'])
        
        # Message history
        with st.expander("Message History"):
            st.markdown('<div class="chat-message other-message">Hello, I\'m interested in your courses</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-message user-message">Thank you for your interest. Can you tell me more about your goals?</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message other-message">{selected_conversation["last_message"]}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        new_message = st.text_input("Type your message...")
        if st.button("Send"):
            st.success("Message sent!")
    
    with tab5:
        st.subheader("Collaborate with Companies")
        st.info("Connect with companies to place your apprentices")
        
        # List companies
        companies = list(st.session_state.companies.values())
        selected_company = st.selectbox("Select Company", companies, format_func=lambda c: c['name'])
        
        if selected_company:
            st.markdown(f"### {selected_company['name']}")
            st.write(f"**Industry:** {selected_company['industry']}")
            st.write(f"**Location:** {selected_company['location']}")
            
            # Shortlisted apprentices
            if 'shortlisted_apprentices' in st.session_state.training_providers[st.session_state.current_user]:
                shortlisted = st.session_state.training_providers[st.session_state.current_user]['shortlisted_apprentices']
                st.markdown("**Your Shortlisted Apprentices:**")
                for app_id in shortlisted[:5]:
                    if app_id in st.session_state.apprentices:
                        app = st.session_state.apprentices[app_id]
                        st.markdown(f"- {app['name']} ({app['location']})")
            
            # Action buttons
            if st.button(f"Recommend Apprentices to {selected_company['name']}"):
                # In a real app, this would send a notification
                st.success(f"Recommendations sent to {selected_company['name']}")
            
            st.markdown("---")
            st.subheader("Shortlist Apprentice for Company")
            
            # Select apprentice to shortlist
            apprentices = list(st.session_state.apprentices.values())
            selected_apprentice = st.selectbox("Select Apprentice", apprentices, format_func=lambda a: a['name'])
            
            if selected_apprentice and st.button("Shortlist for Company"):
                if 'shortlisted_apprentices' not in st.session_state.training_providers[st.session_state.current_user]:
                    st.session_state.training_providers[st.session_state.current_user]['shortlisted_apprentices'] = []
                
                if selected_apprentice['id'] not in st.session_state.training_providers[st.session_state.current_user]['shortlisted_apprentices']:
                    st.session_state.training_providers[st.session_state.current_user]['shortlisted_apprentices'].append(selected_apprentice['id'])
                    st.success(f"{selected_apprentice['name']} shortlisted for {selected_company['name']}")

def admin_dashboard():
    st.markdown(f"<h1 style='text-align: center;'>⚙️ Admin Dashboard</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["System Analytics", "User Management", "Configuration"])
    
    with tab1:
        st.subheader("System Analytics")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Users", len(st.session_state.users))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Apprentices", len(st.session_state.apprentices))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Companies", len(st.session_state.companies))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training Providers", len(st.session_state.training_providers))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Registration Trends")
            dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
            registrations = np.random.randint(10, 50, 6)
            
            fig = px.line(x=dates, y=registrations, title="Monthly User Registrations")
            st.plotly_chart(fig, use_container_width=True)
            
            # Psychometric test analytics
            st.subheader("Psychometric Test Analytics")
            test_taken = sum(1 for a in st.session_state.apprentices.values() if 'psychometric_profile' in a)
            test_not_taken = len(st.session_state.apprentices) - test_taken
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tests Completed", test_taken)
            with col2:
                st.metric("Tests Pending", test_not_taken)
            
            # Trait distribution
            if test_taken > 0:
                st.markdown("### Trait Distribution")
                trait_data = []
                traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                
                for trait in traits:
                    scores = [a['psychometric_profile']['scores'].get(trait, 0) for a in st.session_state.apprentices.values() if 'psychometric_profile' in a]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        trait_data.append({"Trait": trait.title(), "Average Score": avg_score})
                
                if trait_data:
                    df_traits = pd.DataFrame(trait_data)
                    fig = px.bar(df_traits, x='Trait', y='Average Score', 
                                 title="Average Personality Trait Scores",
                                 color='Trait', color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("User Type Distribution")
            user_types = ['Apprentices', 'Companies', 'Training Providers', 'Admins']
            counts = [
                len(st.session_state.apprentices), 
                len(st.session_state.companies), 
                len(st.session_state.training_providers), 
                1
            ]
            
            fig = px.pie(values=counts, names=user_types, title="User Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("User Management")
        
        # User table
        user_data = []
        for user_id, user in st.session_state.users.items():
            user_data.append({
                "ID": user_id[:8],
                "Email": user['email'],
                "Type": user['user_type'],
                "Created": user['created_at'].strftime("%Y-%m-%d")
            })
        
        if user_data:
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found")
        
        st.markdown("---")
        st.subheader("Add New User")
        
        with st.form("new_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
            
            with col2:
                user_type = st.selectbox("User Type", ["apprentice", "company", "training_provider", "admin"])
                role = st.selectbox("Role", ["Standard", "Admin"])
            
            if st.form_submit_button("Create User"):
                profile_data = {"role": role}
                user_id = create_user(email, password, user_type, profile_data)
                st.success(f"User created successfully with ID: {user_id[:8]}")
    
    with tab3:
        st.subheader("System Configuration")
        
        # Language configuration
        st.markdown("### Language Settings")
        current_languages = st.session_state.system_config['languages']
        languages = st.multiselect(
            "Supported Languages", 
            ["English", "Punjabi", "Urdu", "Hindi", "Mirpuri", "Arabic", "Spanish", "French", "German"],
            default=current_languages
        )
        
        # Subscription plans
        st.markdown("### Subscription Plans")
        plans = st.session_state.system_config['subscription_plans']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Basic Plan**")
            price_basic = st.number_input("Price (Basic)", min_value=0, value=plans['basic']['price'])
            candidates_basic = st.number_input("Candidates (Basic)", min_value=1, value=plans['basic']['candidates'])
            features_basic = st.text_area("Features (Basic)", value="\n".join(plans['basic']['features']))
        
        with col2:
            st.markdown("**Standard Plan**")
            price_standard = st.number_input("Price (Standard)", min_value=0, value=plans['standard']['price'])
            candidates_standard = st.number_input("Candidates (Standard)", min_value=1, value=plans['standard']['candidates'])
            features_standard = st.text_area("Features (Standard)", value="\n".join(plans['standard']['features']))
        
        with col3:
            st.markdown("**Professional Plan**")
            price_professional = st.number_input("Price (Professional)", min_value=0, value=plans['professional']['price'])
            candidates_professional = st.number_input("Candidates (Professional)", min_value=1, value=plans['professional']['candidates'])
            features_professional = st.text_area("Features (Professional)", value="\n".join(plans['professional']['features']))
        
        # Psychometric Test Configuration
        st.subheader("Psychometric Test Configuration")
        st.info("Manage the questions for the personality assessment test")
        
        # Question management
        st.markdown("### Current Questions")
        if 'questions' not in st.session_state:
            st.session_state.questions = []
        
        if st.session_state.questions:
            for i, q in enumerate(st.session_state.questions):
                with st.expander(f"Question {i+1}: {q['text']}"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        new_text = st.text_input("Question Text", value=q['text'], key=f"q_text_{i}")
                    with col2:
                        trait = st.selectbox("Trait", 
                                            ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'], 
                                            index=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'].index(q['trait']),
                                            key=f"q_trait_{i}")
                        reverse = st.checkbox("Reverse Score", value=q.get('reverse', False), key=f"q_reverse_{i}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update Question", key=f"update_{i}"):
                            st.session_state.questions[i]['text'] = new_text
                            st.session_state.questions[i]['trait'] = trait
                            st.session_state.questions[i]['reverse'] = reverse
                            st.success("Question updated!")
                    with col2:
                        if st.button("Delete Question", key=f"delete_{i}"):
                            del st.session_state.questions[i]
                            st.rerun()
        else:
            st.info("No questions configured yet")
        
        st.markdown("---")
        st.subheader("Add New Question")
        
        with st.form("new_question_form"):
            question_text = st.text_input("Question Text")
            trait = st.selectbox("Personality Trait", 
                                ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'])
            reverse = st.checkbox("Reverse Scoring (higher score means lower trait)")
            
            if st.form_submit_button("Add Question"):
                new_id = max([q['id'] for q in st.session_state.questions], default=0) + 1
                st.session_state.questions.append({
                    "id": new_id,
                    "text": question_text,
                    "trait": trait,
                    "reverse": reverse
                })
                st.success("Question added successfully!")
        
        # Video Profile Configuration
        st.subheader("Video Profile Configuration")
        st.info("Configure the steps and prompts for the guided video profile")
        
        # Steps configuration
        st.markdown("### Current Steps")
        if 'video_steps' not in st.session_state:
            st.session_state.video_steps = []
        
        if st.session_state.video_steps:
            for i, step in enumerate(st.session_state.video_steps):
                with st.expander(f"Step {i+1}: {step['title']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        title = st.text_input("Title", value=step['title'], key=f"step_title_{i}")
                        key = st.text_input("Key", value=step['key'], key=f"step_key_{i}")
                    with col2:
                        duration = st.number_input("Duration (seconds)", min_value=5, max_value=60, value=step['duration'], key=f"step_duration_{i}")
                        optional = st.checkbox("Optional", value=step.get('optional', False), key=f"step_optional_{i}")
                    
                    prompt = st.text_area("Prompt", value=step['prompt'], key=f"step_prompt_{i}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update Step", key=f"update_step_{i}"):
                            st.session_state.video_steps[i] = {
                                "title": title,
                                "duration": duration,
                                "prompt": prompt,
                                "key": key,
                                "optional": optional
                            }
                            st.success("Step updated!")
                    with col2:
                        if st.button("Delete Step", key=f"delete_step_{i}"):
                            del st.session_state.video_steps[i]
                            st.rerun()
        else:
            st.info("No steps configured yet")
        
        st.markdown("---")
        st.subheader("Add New Step")
        
        with st.form("new_step_form"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Step Title")
                key = st.text_input("Unique Key")
            with col2:
                duration = st.number_input("Duration (seconds)", min_value=5, max_value=60, value=30)
                optional = st.checkbox("Optional")
            
            prompt = st.text_area("Prompt Text")
            
            if st.form_submit_button("Add Step"):
                st.session_state.video_steps.append({
                    "title": title,
                    "duration": duration,
                    "prompt": prompt,
                    "key": key,
                    "optional": optional
                })
                st.success("Step added successfully!")
        
        # Save configuration
        st.markdown("---")
        if st.button("Save Configuration"):
            st.session_state.system_config['languages'] = languages
            st.session_state.system_config['subscription_plans']['basic']['price'] = price_basic
            st.session_state.system_config['subscription_plans']['basic']['candidates'] = candidates_basic
            st.session_state.system_config['subscription_plans']['basic']['features'] = [f.strip() for f in features_basic.split('\n') if f.strip()]
            st.session_state.system_config['subscription_plans']['standard']['price'] = price_standard
            st.session_state.system_config['subscription_plans']['standard']['candidates'] = candidates_standard
            st.session_state.system_config['subscription_plans']['standard']['features'] = [f.strip() for f in features_standard.split('\n') if f.strip()]
            st.session_state.system_config['subscription_plans']['professional']['price'] = price_professional
            st.session_state.system_config['subscription_plans']['professional']['candidates'] = candidates_professional
            st.session_state.system_config['subscription_plans']['professional']['features'] = [f.strip() for f in features_professional.split('\n') if f.strip()]
            st.success("Configuration updated successfully!")

# Main App
def main():
    # Initialize database
    init_database()
    
    # Initialize models
    models = load_models()
    
    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.user_type = None
        st.session_state.page = "login"
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ AI Configuration")
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Enter your Groq API key for enhanced AI insights generation",
            placeholder="gsk_..."
        )
        
        if groq_api_key:
            st.success("✅ Groq API Key configured")
        else:
            st.info("💡 Enter Groq API key for advanced AI insights")
        
        st.markdown("---")
        st.header("📋 System Status")
        
        # Model status
        model_status = []
        if models:
            for model_name, model in models.items():
                if model:
                    model_status.append(f"✅ {model_name.title()}")
                else:
                    model_status.append(f"❌ {model_name.title()}")
        
        if model_status:
            for status in model_status:
                st.write(status)
        else:
            st.write("🔧 Using fallback methods")
    
    # Navigation
    if not st.session_state.authenticated:
        if st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "register":
            register_page()
    else:
        # Sidebar navigation
        with st.sidebar:
            user_data = st.session_state.users[st.session_state.current_user]
            
            # User info
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
            with col2:
                st.markdown(f"**{user_data['profile_data'].get('name', 'User')}**")
                st.caption(f"{user_data['user_type'].replace('_', ' ').title()}")
            
            st.markdown("---")
            
            # Navigation
            if user_data['user_type'] == 'apprentice':
                page = st.radio("Menu", ["Dashboard", "Profile", "Video Profile", "Opportunities", "AI Insights", "Psychometric Test", "Advanced Profile"], 
                               format_func=lambda x: f"📊 {x}" if x == "Dashboard" else f"👤 {x}" if x == "Profile" else f"🎥 {x}" if x == "Video Profile" else f"🔍 {x}" if x == "Opportunities" else f"🧠 {x}" if x == "AI Insights" else f"🧠 {x}" if x == "Psychometric Test" else f"🔬 {x}")
            elif user_data['user_type'] == 'company':
                page = st.radio("Menu", ["Dashboard", "Browse Apprentices", "Subscriptions", "Messages", "Training Providers"],
                               format_func=lambda x: f"📊 {x}" if x == "Dashboard" else f"🔍 {x}" if x == "Browse Apprentices" else f"💳 {x}" if x == "Subscriptions" else f"💬 {x}" if x == "Messages" else f"🏫 {x}")
            elif user_data['user_type'] == 'training_provider':
                page = st.radio("Menu", ["Dashboard", "Browse Apprentices", "Subscriptions", "Messages", "Company Collaboration"],
                               format_func=lambda x: f"📊 {x}" if x == "Dashboard" else f"🔍 {x}" if x == "Browse Apprentices" else f"💳 {x}" if x == "Subscriptions" else f"💬 {x}" if x == "Messages" else f"🤝 {x}")
            elif user_data['user_type'] == 'admin':
                page = st.radio("Menu", ["System Analytics", "User Management", "Configuration"],
                               format_func=lambda x: f"📈 {x}" if x == "System Analytics" else f"👥 {x}" if x == "User Management" else f"⚙️ {x}")
            
            st.markdown("---")
            
            # Notifications
            st.markdown("🔔 **Notifications**")
            st.caption("You have 3 new messages")
            st.caption("2 applications need review")
            
            st.markdown("---")
            
            if st.button("Logout", key="logout_btn"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.session_state.user_type = None
                st.rerun()
        
        # Main content
        if user_data['user_type'] == 'apprentice':
            apprentice_dashboard(models, groq_api_key)
        elif user_data['user_type'] == 'company':
            company_dashboard()
        elif user_data['user_type'] == 'training_provider':
            training_provider_dashboard()
        elif user_data['user_type'] == 'admin':
            admin_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>© 2024 AI Apprentice Platform. All rights reserved.</p>
        <p>Powered by AI | Built with Streamlit | Support: support@apprentice-ai.com</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()