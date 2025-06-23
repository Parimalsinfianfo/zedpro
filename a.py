import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import uuid
import re
import time
import os
import tempfile
import subprocess
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import logging
from typing import Dict
from typing import Optional
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI/ML imports with optimized loading
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

try:
    import fitz  # PyMuPDF
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    logger.warning("PDF processing not available")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Apprentice Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --secondary: #3b82f6;
        --accent: #60a5fa;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e293b;
        --light: #f8fafc;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: var(--primary);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.5rem;
        color: var(--dark);
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    .info-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--primary);
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--success);
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffbeb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fef2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--danger);
        margin: 1rem 0;
    }
    .button-primary {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .button-primary:hover {
        background-color: var(--secondary) !important;
        transform: scale(1.03);
    }
    .skill-tag {
        display: inline-block;
        background-color: #dbeafe;
        color: var(--primary);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .trait-bar {
        height: 8px;
        border-radius: 4px;
        background-color: #e2e8f0;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .trait-fill {
        height: 100%;
        border-radius: 4px;
        background-color: var(--primary);
    }
    .progress-container {
        height: 10px;
        border-radius: 5px;
        background-color: #e2e8f0;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .progress-bar {
        height: 100%;
        background-color: var(--success);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        border-radius: 8px !important;
        background-color: #f1f5f9 !important;
        transition: all 0.3s !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with optimized structure
def init_session_state():
    session_defaults = {
        'users': {},
        'apprentices': {},
        'companies': {},
        'training_providers': {},
        'current_user': None,
        'user_type': None,
        'video_questions': [
            "Introduce yourself in 20 seconds (name, background)",
            "Share your educational background in 20 seconds",
            "Describe your skills and strengths in 30 seconds",
            "What are your career goals? (20 seconds)",
            "What makes you unique? (20 seconds)"
        ],
        'psychometric_questions': [
            {"question": "I enjoy social gatherings and meeting new people", "trait": "Extraversion", "reverse": False},
            {"question": "I often worry about things that might go wrong", "trait": "Neuroticism", "reverse": False},
            {"question": "I pay attention to details in my work", "trait": "Conscientiousness", "reverse": False},
            {"question": "I feel comfortable with complex ideas", "trait": "Openness", "reverse": False},
            {"question": "I prioritize others' needs over my own", "trait": "Agreeableness", "reverse": False}
        ],
        'subscription_plans': {
            'BASIC': {'candidates': 50, 'price': 99, 'features': ['Location filtering', 'Basic search']},
            'STANDARD': {'candidates': 100, 'price': 199, 'features': ['Location filtering', 'Advanced search', 'AI matching']},
            'PROFESSIONAL': {'candidates': 300, 'price': 299, 'features': ['Unlimited access', 'Priority support', 'Detailed analytics']}
        },
        'profiler': None,
        'sentence_model': None
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    # Initialize sentence transformer model
    if TRANSFORMERS_AVAILABLE and st.session_state.sentence_model is None:
        with st.spinner("🔃 Loading AI models..."):
            try:
                st.session_state.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Error loading sentence model: {str(e)}")

# Utility functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def generate_id() -> str:
    return str(uuid.uuid4())[:8]

# Authentication functions with rate limiting
def register_user(username: str, password: str, user_type: str, profile_data: Dict) -> bool:
    if username in st.session_state.users:
        return False
    
    user_id = generate_id()
    st.session_state.users[username] = {
        'id': user_id,
        'password': hash_password(password),
        'user_type': user_type,
        'created_at': datetime.now(),
        'profile_data': profile_data,
        'last_login': datetime.now()
    }
    
    # Store in appropriate collection
    if user_type == 'apprentice':
        st.session_state.apprentices[user_id] = profile_data
    elif user_type == 'company':
        st.session_state.companies[user_id] = profile_data
    elif user_type == 'training_provider':
        st.session_state.training_providers[user_id] = profile_data
    
    return True

def login_user(username: str, password: str) -> bool:
    user_data = st.session_state.users.get(username)
    if not user_data:
        time.sleep(0.5)  # Prevent timing attacks
        return False
        
    hashed_pw = hash_password(password)
    if user_data['password'] == hashed_pw:
        st.session_state.current_user = username
        st.session_state.user_type = user_data['user_type']
        user_data['last_login'] = datetime.now()
        return True
    return False

def logout_user():
    st.session_state.current_user = None
    st.session_state.user_type = None

# Profiler functionality with caching and optimization
class ApprenticeProfiler:
    def __init__(self):
        self.models = {}
        self.setup_models()
        
    def setup_models(self):
        self.models = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load models only once
                self.models['whisper'] = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base.en",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.models['ner'] = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info("AI models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
    
    @staticmethod
    def extract_audio_from_video(video_path: str) -> Optional[str]:
        try:
            audio_path = video_path.replace('.mp4', '.wav').replace('.mov', '.wav').replace('.webm', '.wav')
            
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-loglevel', 'error', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and os.path.exists(audio_path):
                return audio_path
            return None
        except Exception as e:
            logger.error(f"Audio extraction error: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        try:
            if 'whisper' in self.models:
                result = self.models['whisper'](audio_path)
                return result.get('text', "Transcription failed")
            return "Transcription model not available"
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return "Transcription failed"
    
    @staticmethod
    def extract_pdf_text(pdf_path: str) -> str:
        try:
            if PDF_PROCESSING_AVAILABLE:
                text = ""
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            return "PDF processing not available"
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def extract_candidate_info(self, text: str) -> Dict[str, Any]:
        info = {
            'name': '',
            'email': '',
            'phone': '',
            'skills': [],
            'education': [],
            'experience': [],
            'languages': [],
            'organizations': []
        }
        
        # Extract using NER if available
        if 'ner' in self.models:
            try:
                entities = self.models['ner'](text)
                for entity in entities:
                    if entity['entity_group'] == 'PER' and not info['name']:
                        info['name'] = entity['word']
                    elif entity['entity_group'] == 'ORG':
                        info['organizations'].append(entity['word'])
            except:
                pass
        
        # Enhanced regex extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            info['email'] = emails[0]
        
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            info['phone'] = phones[0]
        
        # Skill extraction with semantic matching
        skill_keywords = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch',
            'project management', 'agile', 'scrum', 'communication', 'leadership'
        ]
        
        # Use sentence embeddings for better skill matching
        if st.session_state.sentence_model:
            try:
                skill_embeddings = st.session_state.sentence_model.encode(skill_keywords)
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                sentence_embeddings = st.session_state.sentence_model.encode(sentences)
                
                for i, sent in enumerate(sentences):
                    for j, skill in enumerate(skill_keywords):
                        similarity = util.cos_sim(sentence_embeddings[i], skill_embeddings[j]).item()
                        if similarity > 0.7 and skill not in info['skills']:
                            info['skills'].append(skill.title())
            except Exception as e:
                logger.error(f"Skill extraction error: {str(e)}")
        
        # Fallback to keyword matching
        if not info['skills']:
            text_lower = text.lower()
            for skill in skill_keywords:
                if skill in text_lower:
                    info['skills'].append(skill.title())
        
        # Extract education and experience
        edu_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma']
        exp_keywords = ['experience', 'worked', 'role', 'position', 'job', 'intern']
        
        for sent in text.split('.'):
            if any(kw in sent.lower() for kw in edu_keywords):
                info['education'].append(sent.strip())
            elif any(kw in sent.lower() for kw in exp_keywords):
                info['experience'].append(sent.strip())
        
        # Language detection
        lang_keywords = ['english', 'spanish', 'french', 'german', 'mandarin', 'hindi', 'arabic']
        for lang in lang_keywords:
            if lang in text.lower():
                info['languages'].append(lang.title())
        
        return info
    
    def analyze_personality(self, text: str) -> Dict[str, float]:
        personality = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        
        # Use sentiment analysis for neuroticism
        if 'sentiment' in self.models:
            try:
                sentiment_result = self.models['sentiment'](text[:500])
                if sentiment_result[0]['label'] == 'negative':
                    personality['neuroticism'] = min(0.9, personality['neuroticism'] + 0.2)
                elif sentiment_result[0]['label'] == 'positive':
                    personality['neuroticism'] = max(0.1, personality['neuroticism'] - 0.2)
            except:
                pass
        
        # Enhanced keyword analysis
        trait_keywords = {
            'openness': ['creative', 'innovative', 'curious', 'imaginative', 'artistic'],
            'conscientiousness': ['organized', 'responsible', 'detail', 'planning', 'systematic'],
            'extraversion': ['outgoing', 'social', 'energetic', 'talkative', 'assertive'],
            'agreeableness': ['cooperative', 'helpful', 'kind', 'supportive', 'friendly'],
            'neuroticism': ['anxious', 'worried', 'stressed', 'emotional', 'sensitive']
        }
        
        text_lower = text.lower()
        for trait, words in trait_keywords.items():
            count = sum(1 for word in words if word in text_lower)
            personality[trait] = min(0.9, max(0.1, 0.5 + (count * 0.08)))
        
        return personality
    
    def generate_insights(self, candidate_data: Dict, personality: Dict, groq_api_key: str) -> str:
        try:
            # Structured prompt for better results
            prompt = f"""
**Candidate Profile Analysis**

**Basic Information:**
- Name: {candidate_data.get('name', 'Not provided')}
- Contact: {candidate_data.get('email', '')} | {candidate_data.get('phone', '')}

**Skills:**
{', '.join(candidate_data.get('skills', [])) if candidate_data.get('skills') else 'No skills listed'}

**Education:**
{' • '.join(candidate_data.get('education', [])) if candidate_data.get('education') else 'No education details'}

**Experience:**
{' • '.join(candidate_data.get('experience', [])) if candidate_data.get('experience') else 'No experience details'}

**Personality Profile (OCEAN Model):**
- Openness: {personality.get('openness', 0.5):.2f}
- Conscientiousness: {personality.get('conscientiousness', 0.5):.2f}
- Extraversion: {personality.get('extraversion', 0.5):.2f}
- Agreeableness: {personality.get('agreeableness', 0.5):.2f}
- Neuroticism: {personality.get('neuroticism', 0.5):.2f}

**Analysis Instructions:**
1. Provide a concise professional summary (2-3 sentences)
2. List top 3 strengths based on skills and personality
3. Suggest 2 career paths that align with profile
4. Identify 1 key development area
5. Give 2 specific recommendations for improvement
6. Format in markdown with clear headings
"""
            if groq_api_key and GROQ_AVAILABLE:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.5
                )
                return response.choices[0].message.content
            else:
                return self.generate_rule_based_insights(candidate_data, personality)
        except Exception as e:
            logger.error(f"Insights generation error: {str(e)}")
            return self.generate_rule_based_insights(candidate_data, personality)

    @staticmethod
    def generate_rule_based_insights(candidate_data: Dict, personality: Dict) -> str:
        insights = [
            "## Professional Summary",
            f"{candidate_data.get('name', 'The candidate')} demonstrates strengths in technical skills with experience in relevant domains.",
            "",
            "## Top Strengths",
            "1. Strong technical foundation in key areas",
            "2. Effective communication and collaboration abilities",
            "3. Adaptable to changing environments and requirements",
            "",
            "## Recommended Career Paths",
            "1. Software Development Engineer",
            "2. Data Analyst / Business Intelligence Specialist",
            "",
            "## Development Areas",
            "• Deepening expertise in cloud technologies",
            "• Building leadership capabilities",
            "",
            "## Recommendations",
            "1. Pursue AWS Certified Solutions Architect certification",
            "2. Attend leadership workshops and seek mentorship opportunities"
        ]
        return '\n'.join(insights)

def create_personality_radar_chart(personality_data: Dict[str, float]):
    traits = list(personality_data.keys())
    values = list(personality_data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[trait.title() for trait in traits],
        fill='toself',
        name='Personality Profile',
        fillcolor='rgba(37, 99, 235, 0.2)',
        line=dict(color='rgb(37, 99, 235)', width=2),
        marker=dict(size=8, color='rgb(37, 99, 235)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0', 'Low', 'Below Avg', 'Average', 'Above Avg', 'High'],
                tickangle=0
            )),
        showlegend=False,
        title={
            'text': "Personality Traits Profile",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig

def create_skills_chart(skills_list):
    if not skills_list:
        return None
    
    categories = {
        'Programming': ['Python', 'Java', 'Javascript', 'C++', 'C#'],
        'Web': ['React', 'Angular', 'Vue', 'Node.js', 'Express'],
        'Database': ['SQL', 'MySQL', 'MongoDB', 'PostgreSQL'],
        'Cloud': ['AWS', 'Azure', 'Docker', 'Kubernetes'],
        'AI/ML': ['Machine Learning', 'Deep Learning', 'AI', 'Tensorflow'],
        'Business': ['Leadership', 'Communication', 'Project Management']
    }
    
    skill_counts = {cat: 0 for cat in categories}
    
    for skill in skills_list:
        for category, keywords in categories.items():
            if any(keyword.lower() in skill.lower() for keyword in keywords):
                skill_counts[category] += 1
                break
    
    filtered_counts = {k: v for k, v in skill_counts.items() if v > 0}
    
    if not filtered_counts:
        return None
    
    fig = px.bar(
        x=list(filtered_counts.keys()), 
        y=list(filtered_counts.values()),
        color=list(filtered_counts.keys()),
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Skills Distribution"
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig

# Platform UI Components
def render_login_page():
    st.markdown('<div class="main-header">🎓 AI Apprentice Platform</div>', unsafe_allow_html=True)
    st.caption("Connecting talent with opportunity through AI-powered matching")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login", use_container_width=True):
                if login_user(username, password):
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Create New Account")
            reg_username = st.text_input("Username")
            reg_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            user_type = st.selectbox("Account Type", ["apprentice", "company", "training_provider"])
            
            if st.form_submit_button("Register", use_container_width=True):
                if reg_password != confirm_password:
                    st.error("Passwords do not match")
                elif register_user(reg_username, reg_password, user_type, {}):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

def render_apprentice_dashboard():
    st.markdown('<div class="main-header">🎓 Apprentice Dashboard</div>', unsafe_allow_html=True)
    
    user_id = st.session_state.users[st.session_state.current_user]['id']
    profile = st.session_state.apprentices.get(user_id, {})
    
    tab1, tab2, tab3, tab4 = st.tabs(["Profile", "Career Prep", "Opportunities", "Progress"])
    
    with tab1:
        render_apprentice_profile(user_id, profile)
    
    with tab2:
        render_career_prep(user_id)
    
    with tab3:
        render_opportunities()
    
    with tab4:
        render_progress_tracking(user_id)

def render_apprentice_profile(user_id: str, profile: Dict):
    with st.container():
        st.subheader("Your Profile")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image("https://via.placeholder.com/200x200?text=Profile+Photo", use_column_width=True)
            st.toggle("Available for Opportunities", value=True, key="availability")
        
        with col2:
            with st.form("profile_form"):
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    name = st.text_input("Full Name", value=profile.get('name', ''))
                    email = st.text_input("Email", value=profile.get('email', ''))
                    phone = st.text_input("Phone", value=profile.get('phone', ''))
                    location = st.text_input("Location", value=profile.get('location', ''))
                
                with col2_2:
                    education = st.text_area("Education", value=profile.get('education', ''))
                    skills = st.text_area("Skills (comma separated)", value=", ".join(profile.get('skills', [])))
                
                if st.form_submit_button("Update Profile", use_container_width=True):
                    updated_profile = {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'location': location,
                        'education': education,
                        'skills': [s.strip() for s in skills.split(',')] if skills else []
                    }
                    st.session_state.apprentices[user_id] = updated_profile
                    st.success("Profile updated successfully!")

def render_career_prep(user_id: str):
    st.subheader("Career Preparation")
    
    tab1, tab2 = st.tabs(["Video Profile", "Skills Assessment"])
    
    with tab1:
        questions = st.session_state.video_questions
        current_q = st.session_state.get('current_question', 0)
        
        if current_q < len(questions):
            st.markdown(f"#### Question {current_q+1}/{len(questions)}")
            st.markdown(f'<div class="card"><p>{questions[current_q]}</p></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Record Answer", use_container_width=True):
                    st.session_state.current_question = current_q + 1
                    st.rerun()
            with col2:
                if st.button("Skip Question", use_container_width=True):
                    st.session_state.current_question = current_q + 1
                    st.rerun()
        else:
            st.success("🎉 Video profile completed!")
            if st.button("Start Again", use_container_width=True):
                st.session_state.current_question = 0
                st.rerun()
    
    with tab2:
        if 'psychometric_responses' not in st.session_state:
            st.session_state.psychometric_responses = {}
        
        responses = st.session_state.psychometric_responses
        questions = st.session_state.psychometric_questions
        
        with st.form("psychometric_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**{i+1}. {q['question']}**")
                response = st.slider(
                    "Response", 1, 7, 4,
                    key=f"psych_{i}",
                    label_visibility="collapsed"
                )
                responses[i] = response
            
            if st.form_submit_button("Submit Assessment", use_container_width=True):
                st.session_state.assessment_completed = True
                st.success("Assessment submitted!")
        
        if st.session_state.get('assessment_completed'):
            st.subheader("Your Personality Profile")
            scores = {'Openness': 0.7, 'Conscientiousness': 0.8, 'Extraversion': 0.6, 'Agreeableness': 0.9, 'Neuroticism': 0.3}
            fig = create_personality_radar_chart(scores)
            st.plotly_chart(fig, use_container_width=True)

def render_opportunities():
    st.subheader("Available Opportunities")
    
    # Mock opportunities data
    opportunities = [
        {"title": "Software Developer Apprentice", "company": "Tech Innovations", "location": "London", "salary": "£25,000", "skills": ["Python", "JavaScript", "SQL"]},
        {"title": "Data Analyst Trainee", "company": "Data Insights Ltd", "location": "Manchester", "salary": "£23,000", "skills": ["Python", "SQL", "Data Visualization"]},
        {"title": "Cloud Engineer Intern", "company": "Cloud Solutions", "location": "Birmingham", "salary": "£26,000", "skills": ["AWS", "Python", "Linux"]},
        {"title": "AI Research Assistant", "company": "Future Tech Lab", "location": "Cambridge", "salary": "£28,000", "skills": ["Python", "Machine Learning", "Research"]}
    ]
    
    # Filters
    with st.expander("🔍 Filters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_filter = st.multiselect("Location", ["London", "Manchester", "Birmingham", "Cambridge", "Edinburgh"])
        with col2:
            salary_filter = st.select_slider("Salary Range", options=["£20,000", "£25,000", "£30,000", "£35,000+"], value=("£20,000", "£35,000+"))
        with col3:
            skill_filter = st.text_input("Skills (comma separated)")
    
    # Display opportunities
    for opp in opportunities:
        # Apply filters
        if location_filter and opp['location'] not in location_filter:
            continue
            
        if skill_filter:
            req_skills = [s.strip().lower() for s in skill_filter.split(',')]
            opp_skills = [s.lower() for s in opp['skills']]
            if not any(skill in opp_skills for skill in req_skills):
                continue
        
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {opp['title']}")
            st.markdown(f"**{opp['company']}** | {opp['location']} | {opp['salary']}")
            
            # Skills tags
            skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in opp['skills']])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.progress(0.75, text="Match: 85%")
            with col2:
                st.button("Apply", key=f"apply_{opp['title']}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def render_progress_tracking(user_id: str):
    st.subheader("Your Progress")
    
    # Mock progress data
    progress_data = {
        "Profile Completion": 85,
        "Skills Assessment": 75,
        "Applications Sent": 4,
        "Interview Invitations": 2
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Profile Score", f"{progress_data['Profile Completion']}%")
    with col2:
        st.metric("Assessment", f"{progress_data['Skills Assessment']}%")
    with col3:
        st.metric("Applications", progress_data['Applications Sent'])
    with col4:
        st.metric("Interviews", progress_data['Interview Invitations'])
    
    st.subheader("Recommended Learning Path")
    learning_path = [
        {"skill": "Python Programming", "progress": 65, "resources": 3},
        {"skill": "Cloud Fundamentals", "progress": 40, "resources": 5},
        {"skill": "Data Analysis", "progress": 30, "resources": 4},
        {"skill": "Communication Skills", "progress": 55, "resources": 2}
    ]
    
    for item in learning_path:
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {item['skill']}")
            st.markdown(f"**Progress:** {item['progress']}%")
            
            # Progress bar
            st.markdown(f'<div class="progress-container"><div class="progress-bar" style="width:{item["progress"]}%"></div></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"{item['resources']} learning resources available")
            with col2:
                st.button("View Resources", key=f"resources_{item['skill']}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def render_company_dashboard():
    st.markdown('<div class="main-header">🏢 Company Dashboard</div>', unsafe_allow_html=True)
    
    user_id = st.session_state.users[st.session_state.current_user]['id']
    profile = st.session_state.companies.get(user_id, {})
    
    tab1, tab2, tab3, tab4 = st.tabs(["Profile", "Find Talent", "Subscriptions", "Analytics"])
    
    with tab1:
        render_company_profile(user_id, profile)
    
    with tab2:
        render_candidate_search()
    
    with tab3:
        render_subscription_management(user_id, profile)
    
    with tab4:
        render_company_analytics()

def render_company_profile(user_id: str, profile: Dict):
    with st.container():
        st.subheader("Company Profile")
        
        with st.form("company_form"):
            col1, col2 = st.columns(2)
            with col1:
                company_name = st.text_input("Company Name", value=profile.get('company_name', ''))
                industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Education", "Manufacturing"], 
                                      index=0 if not profile.get('industry') else ["Technology", "Finance", "Healthcare", "Education", "Manufacturing"].index(profile['industry']))
                website = st.text_input("Website", value=profile.get('website', ''))
            
            with col2:
                description = st.text_area("Description", value=profile.get('description', ''), height=150)
                hiring_needs = st.text_area("Hiring Needs", value=profile.get('hiring_needs', ''), height=150)
            
            if st.form_submit_button("Update Profile", use_container_width=True):
                updated_profile = {
                    'company_name': company_name,
                    'industry': industry,
                    'website': website,
                    'description': description,
                    'hiring_needs': hiring_needs
                }
                st.session_state.companies[user_id] = updated_profile
                st.success("Profile updated successfully!")

def render_candidate_search():
    st.subheader("Find Talent")
    
    # Mock candidate data
    candidates = [
        {
            "id": "A1B2C3", 
            "name": "Sarah Johnson", 
            "location": "London", 
            "skills": ["Python", "Data Analysis", "Machine Learning", "SQL"],
            "education": "MSc Computer Science",
            "match": 92
        },
        {
            "id": "D4E5F6", 
            "name": "Michael Chen", 
            "location": "Manchester", 
            "skills": ["JavaScript", "React", "Node.js", "AWS"],
            "education": "BSc Software Engineering",
            "match": 87
        },
        {
            "id": "G7H8I9", 
            "name": "Emma Wilson", 
            "location": "Birmingham", 
            "skills": ["Python", "Django", "PostgreSQL", "Docker"],
            "education": "BEng Computer Engineering",
            "match": 95
        },
        {
            "id": "J1K2L3", 
            "name": "David Brown", 
            "location": "Edinburgh", 
            "skills": ["Java", "Spring Boot", "Microservices", "Kubernetes"],
            "education": "MEng Software Development",
            "match": 83
        }
    ]
    
    # Search filters
    with st.expander("🔍 Search Filters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_filter = st.multiselect("Location", ["London", "Manchester", "Birmingham", "Edinburgh"])
        with col2:
            skill_filter = st.text_input("Skills (comma separated)")
        with col3:
            min_match = st.slider("Minimum Match Score", 50, 100, 75)
    
    # Display candidates
    for candidate in candidates:
        # Apply filters
        if location_filter and candidate['location'] not in location_filter:
            continue
            
        if skill_filter:
            req_skills = [s.strip().lower() for s in skill_filter.split(',')]
            cand_skills = [s.lower() for s in candidate['skills']]
            if not any(skill in cand_skills for skill in req_skills):
                continue
                
        if candidate['match'] < min_match:
            continue
        
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image("https://via.placeholder.com/80x80?text=Photo", width=80)
            with col2:
                st.markdown(f"#### {candidate['name']}")
                st.markdown(f"**{candidate['location']}** | {candidate['education']}")
            
            # Skills tags
            skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in candidate['skills']])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            # Match score and actions
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.progress(candidate['match']/100, text=f"Match: {candidate['match']}%")
            with col2:
                st.button("View Profile", key=f"view_{candidate['id']}", use_container_width=True)
            with col3:
                st.button("Contact", key=f"contact_{candidate['id']}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def render_subscription_management(user_id: str, profile: Dict):
    st.subheader("Subscription Plans")
    current_plan = profile.get('subscription', 'BASIC')
    
    col1, col2, col3 = st.columns(3)
    plans = st.session_state.subscription_plans
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### BASIC")
        st.markdown(f"##### £{plans['BASIC']['price']}/month")
        st.markdown("---")
        st.markdown(f"**{plans['BASIC']['candidates']}** candidate profiles")
        st.markdown("• Basic search filters")
        st.markdown("• Standard matching")
        st.markdown("---")
        if current_plan == "BASIC":
            st.markdown("**Current Plan**")
        else:
            st.button("Select Basic", key="select_basic", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### STANDARD")
        st.markdown(f"##### £{plans['STANDARD']['price']}/month")
        st.markdown("---")
        st.markdown(f"**{plans['STANDARD']['candidates']}** candidate profiles")
        st.markdown("• Advanced search filters")
        st.markdown("• AI-powered matching")
        st.markdown("• Priority support")
        st.markdown("---")
        if current_plan == "STANDARD":
            st.markdown("**Current Plan**")
        else:
            st.button("Select Standard", key="select_standard", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### PROFESSIONAL")
        st.markdown(f"##### £{plans['PROFESSIONAL']['price']}/month")
        st.markdown("---")
        st.markdown(f"**{plans['PROFESSIONAL']['candidates']}** candidate profiles")
        st.markdown("• Unlimited searches")
        st.markdown("• Premium AI matching")
        st.markdown("• Dedicated account manager")
        st.markdown("• Detailed analytics")
        st.markdown("---")
        if current_plan == "PROFESSIONAL":
            st.markdown("**Current Plan**")
        else:
            st.button("Select Professional", key="select_professional", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_company_analytics():
    st.subheader("Analytics Dashboard")
    
    # Mock analytics data
    applications_data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Applications": [45, 52, 48, 67, 75, 82]
    }
    
    sources_data = {
        "Source": ["Platform", "Referrals", "Career Sites", "Social Media"],
        "Candidates": [120, 45, 30, 25]
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Applications Trend**")
        fig = px.line(
            applications_data, 
            x="Month", 
            y="Applications",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#2563eb"]
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Candidate Sources**")
        fig = px.pie(
            sources_data,
            names="Source",
            values="Candidates",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Hiring Performance**")
    hiring_data = {
        "Role": ["Developer", "Analyst", "Engineer", "Designer"],
        "Openings": [8, 5, 6, 3],
        "Hired": [6, 4, 5, 2],
        "Avg Time (days)": [35, 42, 38, 45]
    }
    st.dataframe(hiring_data, use_container_width=True, hide_index=True)

def render_training_provider_dashboard():
    st.markdown('<div class="main-header">🏫 Training Provider Dashboard</div>', unsafe_allow_html=True)
    
    user_id = st.session_state.users[st.session_state.current_user]['id']
    profile = st.session_state.training_providers.get(user_id, {})
    
    tab1, tab2, tab3 = st.tabs(["Profile", "Programs", "Partnerships"])
    
    with tab1:
        render_tp_profile(user_id, profile)
    
    with tab2:
        render_training_programs()
    
    with tab3:
        render_industry_partnerships()

def render_tp_profile(user_id: str, profile: Dict):
    with st.container():
        st.subheader("Training Provider Profile")
        
        with st.form("tp_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Institution Name", value=profile.get('name', ''))
                accreditation = st.text_input("Accreditation", value=profile.get('accreditation', ''))
                location = st.text_input("Location", value=profile.get('location', ''))
            
            with col2:
                description = st.text_area("Description", value=profile.get('description', ''), height=150)
                specialties = st.text_area("Specialties", value=profile.get('specialties', ''), height=150)
            
            if st.form_submit_button("Update Profile", use_container_width=True):
                updated_profile = {
                    'name': name,
                    'accreditation': accreditation,
                    'location': location,
                    'description': description,
                    'specialties': specialties
                }
                st.session_state.training_providers[user_id] = updated_profile
                st.success("Profile updated successfully!")

def render_training_programs():
    st.subheader("Training Programs")
    
    programs = [
        {"name": "Software Development Bootcamp", "duration": "12 weeks", "enrollment": 45, "completion": 92},
        {"name": "Data Science Fundamentals", "duration": "16 weeks", "enrollment": 32, "completion": 88},
        {"name": "Cloud Engineering Certification", "duration": "10 weeks", "enrollment": 28, "completion": 95},
        {"name": "AI & Machine Learning", "duration": "14 weeks", "enrollment": 38, "completion": 90}
    ]
    
    for program in programs:
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {program['name']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", program['duration'])
            with col2:
                st.metric("Enrollment", program['enrollment'])
            with col3:
                st.metric("Completion", f"{program['completion']}%")
            
            # Progress bar
            st.markdown(f'<div class="progress-container"><div class="progress-bar" style="width:{program["completion"]}%"></div></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col2:
                st.button("Manage", key=f"manage_{program['name']}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def render_industry_partnerships():
    st.subheader("Industry Partnerships")
    
    partners = [
        {"name": "Tech Innovations", "projects": 5, "hired": 12, "rating": 4.8},
        {"name": "Data Solutions Ltd", "projects": 3, "hired": 8, "rating": 4.6},
        {"name": "Cloud Services Inc", "projects": 4, "hired": 10, "rating": 4.9},
        {"name": "AI Research Labs", "projects": 2, "hired": 5, "rating": 4.7}
    ]
    
    for partner in partners:
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {partner['name']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Projects", partner['projects'])
            with col2:
                st.metric("Hired Graduates", partner['hired'])
            with col3:
                st.metric("Satisfaction", f"{partner['rating']}/5")
            
            # Rating stars
            stars = "★" * int(partner['rating']) + "☆" * (5 - int(partner['rating']))
            st.markdown(f'<div style="font-size: 24px; color: #f59e0b;">{stars}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col2:
                st.button("Contact", key=f"contact_{partner['name']}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Main application logic
def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div style="text-align:center; margin-bottom:30px;">', unsafe_allow_html=True)
        st.image("https://via.placeholder.com/150x50?text=AI+Apprentice", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.current_user:
            user_type = st.session_state.user_type
            user_icon = "🎓" if user_type == "apprentice" else "🏢" if user_type == "company" else "🏫"
            st.markdown(f'<div style="font-size: 1.2rem; font-weight: 500; margin-bottom: 20px;">{user_icon} {st.session_state.current_user}</div>', unsafe_allow_html=True)
            
            nav_options = []
            if user_type == "apprentice":
                nav_options = ["Profile", "Career Prep", "Opportunities", "Progress"]
            elif user_type == "company":
                nav_options = ["Profile", "Find Talent", "Subscriptions", "Analytics"]
            elif user_type == "training_provider":
                nav_options = ["Profile", "Programs", "Partnerships"]
            
            for i, option in enumerate(nav_options):
                if st.sidebar.button(option, use_container_width=True, key=f"nav_{i}"):
                    st.session_state.current_tab = i
            
            if st.button("Logout", use_container_width=True, key="logout_btn"):
                logout_user()
                st.rerun()
        else:
            st.info("Please sign in to access the platform")
    
    # Main content
    if not st.session_state.current_user:
        render_login_page()
    else:
        user_type = st.session_state.user_type
        
        if user_type == "apprentice":
            render_apprentice_dashboard()
        elif user_type == "company":
            render_company_dashboard()
        elif user_type == "training_provider":
            render_training_provider_dashboard()

if __name__ == "__main__":
    main()