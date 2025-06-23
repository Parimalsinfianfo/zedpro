#!/usr/bin/env python3
"""
AI-Powered Apprenticeship Platform
A comprehensive Streamlit application connecting apprentices with companies and training providers
using advanced AI profiling, video interviews, and psychometric analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import uuid
from datetime import datetime, timedelta
import time
import random
import base64
import io
import re
import os
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mock imports for AI models (would be real in production)
try:
    import requests
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# App Configuration
APP_CONFIG = {
    'title': '🎓 AI Apprenticeship Platform',
    'version': '1.0.0',
    'groq_api_url': 'https://api.groq.com/openai/v1/chat/completions',
    'whisper_api_url': 'https://api.groq.com/openai/v1/audio/transcriptions',
    'supported_languages': ['English', 'Punjabi', 'Urdu', 'Hindi', 'Mirpuri', 'Arabic'],
    'video_max_duration': 180,  # 3 minutes
    'audio_chunk_size': 30,     # 30 seconds
}

# Subscription Plans
SUBSCRIPTION_PLANS = {
    'BASIC': {
        'name': 'Basic',
        'price': 29.99,
        'candidates_limit': 50,
        'features': ['Location filtering', 'Basic search', 'Email support'],
        'color': '#3498db'
    },
    'STANDARD': {
        'name': 'Standard',
        'price': 79.99,
        'candidates_limit': 300,
        'features': ['Advanced filtering', 'AI matching', 'Priority support', 'Analytics'],
        'color': '#e74c3c'
    },
    'PROFESSIONAL': {
        'name': 'Professional',
        'price': 149.99,
        'candidates_limit': float('inf'),
        'features': ['Unlimited access', 'Dedicated support', 'Custom reports', 'API access'],
        'color': '#f39c12'
    }
}

# OCEAN Personality Model
OCEAN_TRAITS = {
    'Openness': 'Creativity, curiosity, and openness to new experiences',
    'Conscientiousness': 'Organization, responsibility, and dependability',
    'Extraversion': 'Sociability, assertiveness, and positive emotions',
    'Agreeableness': 'Cooperation, trust, and empathy towards others',
    'Neuroticism': 'Emotional stability and stress management'
}

# Interview Questions for Apprentices
INTERVIEW_QUESTIONS = [
    "Tell us about yourself and your background (20 seconds)",
    "What is your educational background? (20 seconds)",
    "Which languages can you speak fluently? (20 seconds)",
    "What are your main areas of specialization or interest? (20 seconds)",
    "What type of opportunities are you looking for? (20 seconds)",
    "Tell us about your location and travel flexibility (20 seconds)",
    "What are your hobbies and interests outside of work? (20 seconds)",
    "Do you have any special requirements or accommodations? (20 seconds)",
    "Describe your career goals and aspirations (30 seconds)",
    "Why do you want to become an apprentice? (30 seconds)"
]

# Psychometric Test Questions (OCEAN Model)
PSYCHOMETRIC_QUESTIONS = [
    {"question": "I am outgoing and sociable", "trait": "Extraversion"},
    {"question": "I am systematic and organized", "trait": "Conscientiousness"},
    {"question": "I am creative and imaginative", "trait": "Openness"},
    {"question": "I am helpful and cooperative", "trait": "Agreeableness"},
    {"question": "I remain calm under pressure", "trait": "Neuroticism", "reverse": True},
    {"question": "I enjoy meeting new people", "trait": "Extraversion"},
    {"question": "I pay attention to details", "trait": "Conscientiousness"},
    {"question": "I enjoy exploring new ideas", "trait": "Openness"},
    {"question": "I trust others easily", "trait": "Agreeableness"},
    {"question": "I worry about things frequently", "trait": "Neuroticism"}
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_user_id() -> str:
    """Generate unique user ID"""
    return str(uuid.uuid4())

def format_currency(amount: float) -> str:
    """Format currency display"""
    return f"${amount:.2f}"

def calculate_compatibility_score(profile1: Dict, profile2: Dict) -> float:
    """Calculate compatibility score between profiles"""
    score = 0.0
    
    # Location match
    if profile1.get('location') == profile2.get('location'):
        score += 0.3
    
    # Skills overlap
    skills1 = set(profile1.get('skills', []))
    skills2 = set(profile2.get('skills', []))
    if skills1 and skills2:
        overlap = len(skills1.intersection(skills2)) / len(skills1.union(skills2))
        score += 0.4 * overlap
    
    # Language match
    lang1 = set(profile1.get('languages', []))
    lang2 = set(profile2.get('languages', []))
    if lang1 and lang2:
        overlap = len(lang1.intersection(lang2)) / len(lang1.union(lang2))
        score += 0.3 * overlap
    
    return min(score, 1.0)

def extract_text_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from text using regex patterns"""
    entities = {
        'skills': [],
        'languages': [],
        'locations': [],
        'education': [],
        'experience': []
    }
    
    # Skills pattern
    skill_patterns = [
        r'\b(?:programming|coding|python|javascript|java|html|css|sql|database|web development|mobile development|ai|machine learning|data analysis|project management|communication|leadership|teamwork|problem solving|creativity)\b'
    ]
    
    # Languages pattern
    lang_patterns = [
        r'\b(?:english|spanish|french|german|mandarin|hindi|arabic|punjabi|urdu|mirpuri)\b'
    ]
    
    # Education pattern
    edu_patterns = [
        r'\b(?:bachelor|master|phd|degree|diploma|certificate|university|college|school|gcse|a-level|btec)\b'
    ]
    
    text_lower = text.lower()
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        entities['skills'].extend(matches)
    
    for pattern in lang_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        entities['languages'].extend(matches)
    
    for pattern in edu_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        entities['education'].extend(matches)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def analyze_personality_traits(text: str) -> Dict[str, float]:
    """Analyze personality traits from text using keyword analysis"""
    traits = {trait: 0.0 for trait in OCEAN_TRAITS.keys()}
    
    # Keyword mappings for each trait
    trait_keywords = {
        'Openness': ['creative', 'innovative', 'curious', 'imaginative', 'artistic', 'explore', 'new', 'ideas'],
        'Conscientiousness': ['organized', 'responsible', 'reliable', 'disciplined', 'careful', 'thorough', 'systematic'],
        'Extraversion': ['outgoing', 'social', 'talkative', 'energetic', 'assertive', 'confident', 'friendly'],
        'Agreeableness': ['helpful', 'cooperative', 'trusting', 'kind', 'sympathetic', 'considerate', 'generous'],
        'Neuroticism': ['anxious', 'stressed', 'worried', 'nervous', 'emotional', 'sensitive', 'unstable']
    }
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    for trait, keywords in trait_keywords.items():
        score = 0
        for keyword in keywords:
            score += text_lower.count(keyword)
        
        # Normalize by text length
        if word_count > 0:
            traits[trait] = min(score / word_count * 10, 1.0)
    
    return traits

def generate_ai_insights(profile_data: Dict, use_groq: bool = False) -> str:
    """Generate AI insights about the candidate profile"""
    if use_groq and GROQ_AVAILABLE:
        return generate_groq_insights(profile_data)
    else:
        return generate_rule_based_insights(profile_data)

def generate_groq_insights(profile_data: Dict) -> str:
    """Generate insights using Groq API (mock implementation)"""
    try:
        # This would be a real API call in production
        insights = [
            "Strong technical foundation with diverse skill set",
            "Excellent communication skills evident in profile responses",
            "High potential for leadership roles based on personality assessment",
            "Well-suited for collaborative work environments",
            "Shows adaptability and learning orientation"
        ]
        return ". ".join(random.sample(insights, 3))
    except:
        return generate_rule_based_insights(profile_data)

def generate_rule_based_insights(profile_data: Dict) -> str:
    """Generate insights using rule-based analysis"""
    insights = []
    
    skills = profile_data.get('skills', [])
    personality = profile_data.get('personality_traits', {})
    
    if len(skills) > 5:
        insights.append("Demonstrates versatile skill set across multiple domains")
    
    if personality.get('Extraversion', 0) > 0.7:
        insights.append("Shows strong leadership and communication potential")
    
    if personality.get('Conscientiousness', 0) > 0.7:
        insights.append("Exhibits high reliability and organizational skills")
    
    if personality.get('Openness', 0) > 0.7:
        insights.append("Displays creativity and adaptability to new challenges")
    
    if not insights:
        insights.append("Profile shows good overall potential for apprenticeship opportunities")
    
    return ". ".join(insights)

# =============================================================================
# AI PROCESSING CLASSES
# =============================================================================

class AIProfileAnalyzer:
    """AI-powered profile analysis system"""
    
    def __init__(self):
        self.groq_api_key = None  # Would be set from environment
        self.supported_formats = ['mp4', 'wav', 'mp3', 'pdf', 'txt']
    
    def process_video(self, video_file) -> Dict[str, Any]:
        """Process video file and extract insights"""
        st.info("🎥 Processing video file...")
        
        # Simulate video processing
        time.sleep(2)
        
        # Mock audio extraction
        st.info("🔊 Extracting audio from video...")
        time.sleep(1)
        
        # Mock speech-to-text
        st.info("🗣️ Converting speech to text...")
        time.sleep(2)
        
        # Generate mock transcript
        transcript = self._generate_mock_transcript()
        
        return self._analyze_transcript(transcript)
    
    def process_audio(self, audio_file) -> Dict[str, Any]:
        """Process audio file and extract insights"""
        st.info("🔊 Processing audio file...")
        time.sleep(2)
        
        # Mock speech-to-text
        transcript = self._generate_mock_transcript()
        
        return self._analyze_transcript(transcript)
    
    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Process PDF file and extract text"""
        st.info("📄 Extracting text from PDF...")
        time.sleep(1)
        
        # Mock PDF text extraction
        text = self._generate_mock_cv_text()
        
        return self._analyze_transcript(text)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process direct text input"""
        return self._analyze_transcript(text)
    
    def _generate_mock_transcript(self) -> str:
        """Generate mock transcript for demo purposes"""
        samples = [
            "Hi, my name is Sarah Johnson. I'm 22 years old and from Manchester. I recently graduated with a degree in Computer Science from Manchester University. I'm passionate about web development and have experience with Python, JavaScript, and HTML. I'm looking for an apprenticeship opportunity in software development where I can apply my technical skills and learn from experienced professionals. I speak English and Spanish fluently, and I'm willing to relocate for the right opportunity.",
            
            "Hello, I'm Ahmed Hassan, 20 years old from Birmingham. I completed my A-levels in Mathematics, Physics, and Computer Science. I've been teaching myself programming for the past two years and have built several web applications using React and Node.js. I'm particularly interested in artificial intelligence and machine learning. I speak English, Arabic, and Urdu. I'm seeking an apprenticeship in tech where I can grow my skills in AI and software development.",
            
            "My name is Emma Williams, I'm 19 and from London. I've just finished my BTEC in Digital Media and have a strong interest in graphic design and user experience. I'm proficient in Adobe Creative Suite and have basic knowledge of HTML and CSS. I enjoy creative problem-solving and working in teams. I'm looking for opportunities in digital marketing or UX design. I speak English and French, and I love traveling and photography in my spare time."
        ]
        
        return random.choice(samples)
    
    def _generate_mock_cv_text(self) -> str:
        """Generate mock CV text for demo purposes"""
        return """
        CURRICULUM VITAE
        
        Name: Alex Thompson
        Age: 21
        Location: Leeds, UK
        Email: alex.thompson@email.com
        Phone: +44 7700 900123
        
        EDUCATION:
        - BSc Computer Science, University of Leeds (2021-2024)
        - A-Levels: Mathematics (A), Physics (A), Computer Science (A*)
        - GCSEs: 9 subjects including English and Mathematics (Grades A-B)
        
        SKILLS:
        - Programming: Python, Java, JavaScript, C++
        - Web Development: HTML, CSS, React, Node.js
        - Database: MySQL, MongoDB
        - Tools: Git, Docker, VS Code
        - Languages: English (Native), German (Intermediate)
        
        EXPERIENCE:
        - Software Development Intern, TechStart Ltd (Summer 2023)
        - Freelance Web Developer (2022-Present)
        - IT Support Assistant, University of Leeds (Part-time, 2022-2023)
        
        INTERESTS:
        - Technology innovation and artificial intelligence
        - Open source contributing
        - Football and rock climbing
        """
    
    def _analyze_transcript(self, text: str) -> Dict[str, Any]:
        """Analyze transcript and extract insights"""
        # Extract entities
        entities = extract_text_entities(text)
        
        # Analyze personality traits
        personality_traits = analyze_personality_traits(text)
        
        # Generate insights
        profile_data = {
            'text': text,
            'skills': entities['skills'],
            'languages': entities['languages'],
            'education': entities['education'],
            'personality_traits': personality_traits
        }
        
        insights = generate_ai_insights(profile_data)
        
        return {
            'transcript': text,
            'entities': entities,
            'personality_traits': personality_traits,
            'insights': insights,
            'word_count': len(text.split()),
            'analysis_timestamp': datetime.now().isoformat()
        }

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # User Database (In production, this would be a real database)
    if 'users_db' not in st.session_state:
        st.session_state.users_db = {
            'apprentices': {},
            'companies': {},
            'training_providers': {},
            'admins': {
                'admin': {
                    'id': 'admin-001',
                    'username': 'admin',
                    'password': hash_password('admin123'),
                    'email': 'admin@apprenticeai.com',
                    'created_at': datetime.now().isoformat()
                }
            }
        }
    
    # Application Data
    if 'apprentice_profiles' not in st.session_state:
        st.session_state.apprentice_profiles = {}
    if 'company_profiles' not in st.session_state:
        st.session_state.company_profiles = {}
    if 'training_provider_profiles' not in st.session_state:
        st.session_state.training_provider_profiles = {}
    
    # AI Analyzer
    if 'ai_analyzer' not in st.session_state:
        st.session_state.ai_analyzer = AIProfileAnalyzer()
    
    # Mock Data
    if 'mock_candidates' not in st.session_state:
        st.session_state.mock_candidates = generate_mock_candidates()
    if 'mock_opportunities' not in st.session_state:
        st.session_state.mock_opportunities = generate_mock_opportunities()

def generate_mock_candidates() -> List[Dict]:
    """Generate mock candidate data for demo"""
    candidates = []
    names = ['Sarah Johnson', 'Ahmed Hassan', 'Emma Williams', 'Alex Thompson', 'Maya Patel', 'James Wilson']
    locations = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol', 'Edinburgh']
    skills_pool = ['Python', 'JavaScript', 'HTML/CSS', 'React', 'Node.js', 'SQL', 'Java', 'C++', 'Git', 'Docker']
    
    for i, name in enumerate(names):
        candidate = {
            'id': f'cand-{i+1:03d}',
            'name': name,
            'age': random.randint(18, 25),
            'location': random.choice(locations),
            'skills': random.sample(skills_pool, random.randint(3, 7)),
            'languages': random.sample(APP_CONFIG['supported_languages'], random.randint(1, 3)),
            'education_level': random.choice(['A-Levels', 'BTEC', 'Bachelor\'s Degree', 'Diploma']),
            'availability': random.choice(['Full-time', 'Part-time', 'Flexible']),
            'personality_scores': {trait: random.uniform(0.3, 0.9) for trait in OCEAN_TRAITS.keys()},
            'compatibility_score': random.uniform(0.6, 0.95),
            'status': random.choice(['Available', 'In Process', 'Placed']),
            'created_at': datetime.now() - timedelta(days=random.randint(1, 90))
        }
        candidates.append(candidate)
    
    return candidates

def generate_mock_opportunities() -> List[Dict]:
    """Generate mock opportunity data for demo"""
    opportunities = []
    companies = ['TechCorp Ltd', 'Digital Solutions Inc', 'InnovateTech', 'FutureSoft', 'DataDriven Co']
    roles = ['Software Developer', 'Web Developer', 'Data Analyst', 'Digital Marketing', 'UX Designer']
    
    for i in range(15):
        opportunity = {
            'id': f'opp-{i+1:03d}',
            'title': random.choice(roles) + ' Apprenticeship',
            'company': random.choice(companies),
            'location': random.choice(['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol']),
            'type': random.choice(['Level 3', 'Level 4', 'Level 6', 'Degree Apprenticeship']),
            'duration': random.choice(['12 months', '18 months', '24 months', '36 months']),
            'salary': random.randint(16000, 25000),
            'skills_required': random.sample(['Python', 'JavaScript', 'HTML/CSS', 'React', 'SQL', 'Communication'], 3),
            'description': f"Join our team as a {random.choice(roles)} apprentice and develop your skills in a supportive environment.",
            'applications': random.randint(5, 50),
            'status': 'Open',
            'posted_date': datetime.now() - timedelta(days=random.randint(1, 30))
        }
        opportunities.append(opportunity)
    
    return opportunities

# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================

def render_login_page():
    """Render the login/registration page"""
    st.title("🎓 AI Apprenticeship Platform")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        render_login_form()
    
    with tab2:
        render_registration_form()

def render_login_form():
    """Render login form"""
    st.subheader("Login to Your Account")
    
    with st.form("login_form"):
        role = st.selectbox("Select Your Role", 
                           ["Apprentice", "Company", "Training Provider", "Admin"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate_user(username, password, role.lower().replace(" ", "_")):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

def render_registration_form():
    """Render registration form"""
    st.subheader("Create New Account")
    
    with st.form("registration_form"):
        role = st.selectbox("Account Type", 
                           ["Apprentice", "Company", "Training Provider"])
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long!")
            elif register_user(username, email, password, role.lower().replace(" ", "_")):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists!")

def authenticate_user(username: str, password: str, role: str) -> bool:
    """Authenticate user credentials"""
    role_key = f"{role}s" if role in ['apprentice', 'company'] else f"{role.replace('_', '_')}s"
    
    if role_key in st.session_state.users_db:
        users = st.session_state.users_db[role_key]
        if username in users:
            stored_hash = users[username]['password']
            if stored_hash == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.user_role = role
                return True
    
    return False

def register_user(username: str, email: str, password: str, role: str) -> bool:
    """Register new user"""
    role_key = f"{role}s" if role in ['apprentice', 'company'] else f"{role.replace('_', '_')}s"
    
    if role_key in st.session_state.users_db:
        users = st.session_state.users_db[role_key]
        if username not in users:
            users[username] = {
                'id': generate_user_id(),
                'username': username,
                'email': email,
                'password': hash_password(password),
                'role': role,
                'created_at': datetime.now().isoformat(),
                'subscription': 'BASIC' if role in ['company', 'training_provider'] else None
            }
            return True
    
    return False

# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_apprentice_dashboard():
    """Render apprentice dashboard"""
    st.title(f"👨‍🎓 Welcome, {st.session_state.current_user}")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Profile", "🎥 Video Recording", "🧠 Psychometric Test", 
        "🔍 Opportunities", "📊 AI Analysis"
    ])
    
    with tab1:
        render_apprentice_profile()
    
    with tab2:
        render_video_recording_interface()
    
    with tab3:
        render_psychometric_test()
    
    with tab4:
        render_opportunity_search()
    
    with tab5:
        render_ai_profile_analyzer()

def render_apprentice_profile():
    """Render apprentice profile management"""
    st.subheader("📋 Your Profile")
    
    # Get or create profile
    username = st.session_state.current_user
    if username not in st.session_state.apprentice_profiles:
        st.session_state.apprentice_profiles[username] = {
            'personal_info': {},
            'academic_info': {},
            'skills': [],
            'languages': [],
            'availability': True,
            'documents': {}
        }
    
    profile = st.session_state.apprentice_profiles[username]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        with st.form("personal_info"):
            full_name = st.text_input("Full Name", value=profile['personal_info'].get('full_name', ''))
            age = st.number_input("Age", min_value=16, max_value=30, 
                                value=profile['personal_info'].get('age', 18))
            location = st.text_input("Location", value=profile['personal_info'].get('location', ''))
            phone = st.text_input("Phone", value=profile['personal_info'].get('phone', ''))
            
            if st.form_submit_button("Update Personal Info"):
                profile['personal_info'].update({
                    'full_name': full_name,
                    'age': age,
                    'location': location,
                    'phone': phone
                })
                st.success("Personal information updated!")
    
    with col2:
        st.markdown("#### Academic Information")
        with st.form("academic_info"):
            education_level = st.selectbox("Education Level", 
                                         ["GCSE", "A-Levels", "BTEC", "Diploma", "Bachelor's", "Master's"],
                                         index=0 if not profile['academic_info'].get('education_level') else 
                                         ["GCSE", "A-Levels", "BTEC", "Diploma", "Bachelor's", "Master's"].index(profile['academic_info'].get('education_level')))
            institution = st.text_input("Institution", value=profile['academic_info'].get('institution', ''))
            field_of_study = st.text_input("Field of Study", value=profile['academic_info'].get('field_of_study', ''))
            graduation_year = st.number_input("Graduation Year", min_value=2010, max_value=2030, 
                                            value=profile['academic_info'].get('graduation_year', 2024))
            
            if st.form_submit_button("Update Academic Info"):
                profile['academic_info'].update({
                    'education_level': education_level,
                    'institution': institution,
                    'field_of_study': field_of_study,
                    'graduation_year': graduation_year
                })
                st.success("Academic information updated!")
    
    # Skills and Languages
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Skills")
        skills_input = st.text_area("Enter your skills (one per line)", 
                                   value='\n'.join(profile['skills']))
        if st.button("Update Skills"):
            profile['skills'] = [skill.strip() for skill in skills_input.split('\n') if skill.strip()]
            st.success("Skills updated!")
    
    with col4:
        st.markdown("#### Languages")
        selected_languages = st.multiselect("Select Languages", 
                                           APP_CONFIG['supported_languages'],
                                           default=profile['languages'])
        if st.button("Update Languages"):
            profile['languages'] = selected_languages
            st.success("Languages updated!")
    
    # Availability Toggle
    st.markdown("#### Availability")
    availability = st.toggle("Available for Opportunities", value=profile['availability'])
    profile['availability'] = availability
    
    if availability:
        st.success("✅ Your profile is visible to employers")
    else:
        st.warning("⚠️ Your profile is hidden from employers")

def render_video_recording_interface():
    """Render video recording interface for apprentices"""
    st.subheader("🎥 Video Profile Recording")
    st.markdown("Record your video profile to help employers get to know you better!")
    
    # Recording instructions
    with st.expander("📋 Recording Instructions", expanded=True):
        st.markdown("""
        **Tips for a great video profile:**
        - Find a quiet, well-lit space
        - Look directly at the camera
        - Speak clearly and at a moderate pace
        - Be yourself and stay positive
        - Each question has a recommended time limit
        """)
    
    # Question selector
    st.markdown("#### Select Question to Record")
    question_idx = st.selectbox("Choose a question:", 
                               range(len(INTERVIEW_QUESTIONS)),
                               format_func=lambda x: f"Q{x+1}: {INTERVIEW_QUESTIONS[x][:50]}...")
    
    selected_question = INTERVIEW_QUESTIONS[question_idx]
    st.info(f"**Question {question_idx + 1}:** {selected_question}")
    
    # Recording interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 Start Recording", type="primary"):
            st.session_state.recording = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Recording"):
            st.session_state.recording = False
            st.success("Recording stopped!")
    
    with col3:
        if st.button("▶️ Play Recording"):
            st.info("Playing back your recording...")
    
    # Camera input (mock)
    st.markdown("#### Camera Preview")
    camera_input = st.camera_input("Take a photo for your profile")
    
    if camera_input is not None:
        st.success("Photo captured! This would be processed as video in production.")
    
    # Recording status
    if hasattr(st.session_state, 'recording') and st.session_state.recording:
        st.warning("🔴 Recording in progress...")
        
        # Mock countdown
        countdown_placeholder = st.empty()
        for i in range(3, 0, -1):
            countdown_placeholder.error(f"Recording starts in {i}...")
            time.sleep(1)
        countdown_placeholder.success("Recording now!")
    
    # Recorded videos list
    st.markdown("#### Your Recorded Answers")
    username = st.session_state.current_user
    
    if username not in st.session_state.apprentice_profiles:
        st.session_state.apprentice_profiles[username] = {'recorded_answers': {}}
    
    profile = st.session_state.apprentice_profiles[username]
    if 'recorded_answers' not in profile:
        profile['recorded_answers'] = {}
    
    for i, question in enumerate(INTERVIEW_QUESTIONS):
        status = "✅ Recorded" if i in profile['recorded_answers'] else "⏳ Not recorded"
        st.write(f"**Q{i+1}:** {question[:60]}... - {status}")

def render_psychometric_test():
    """Render psychometric test interface"""
    st.subheader("🧠 Psychometric Assessment")
    st.markdown("Complete this personality assessment to help match you with suitable opportunities.")
    
    # Test instructions
    with st.expander("ℹ️ About This Test", expanded=True):
        st.markdown("""
        This assessment measures the **Big Five** personality traits:
        - **Openness**: Creativity and openness to new experiences
        - **Conscientiousness**: Organization and dependability
        - **Extraversion**: Sociability and assertiveness
        - **Agreeableness**: Cooperation and empathy
        - **Neuroticism**: Emotional stability
        
        Answer honestly - there are no right or wrong answers!
        """)
    
    # Check if test is already completed
    username = st.session_state.current_user
    if username not in st.session_state.apprentice_profiles:
        st.session_state.apprentice_profiles[username] = {}
    
    profile = st.session_state.apprentice_profiles[username]
    
    if 'psychometric_results' in profile:
        st.success("✅ Test completed!")
        render_psychometric_results(profile['psychometric_results'])
        
        if st.button("🔄 Retake Test"):
            del profile['psychometric_results']
            st.rerun()
    else:
        render_psychometric_questions()

def render_psychometric_questions():
    """Render psychometric test questions"""
    st.markdown("#### Assessment Questions")
    st.markdown("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree)")
    
    with st.form("psychometric_test"):
        responses = {}
        
        for i, q in enumerate(PSYCHOMETRIC_QUESTIONS):
            responses[i] = st.slider(
                f"**{i+1}.** {q['question']}", 
                min_value=1, max_value=5, value=3,
                key=f"q_{i}"
            )
        
        if st.form_submit_button("Complete Assessment", type="primary"):
            # Calculate trait scores
            trait_scores = calculate_trait_scores(responses)
            
            # Store results
            username = st.session_state.current_user
            st.session_state.apprentice_profiles[username]['psychometric_results'] = {
                'trait_scores': trait_scores,
                'responses': responses,
                'completed_at': datetime.now().isoformat()
            }
            
            st.success("Assessment completed successfully!")
            st.rerun()

def calculate_trait_scores(responses: Dict[int, int]) -> Dict[str, float]:
    """Calculate OCEAN trait scores from responses"""
    trait_scores = {trait: [] for trait in OCEAN_TRAITS.keys()}
    
    for i, response in responses.items():
        question = PSYCHOMETRIC_QUESTIONS[i]
        trait = question['trait']
        
        # Reverse scoring for negatively worded items
        if question.get('reverse', False):
            score = 6 - response  # Reverse 1-5 scale
        else:
            score = response
        
        trait_scores[trait].append(score)
    
    # Calculate average scores (normalized to 0-1)
    final_scores = {}
    for trait, scores in trait_scores.items():
        if scores:
            final_scores[trait] = (sum(scores) / len(scores) - 1) / 4  # Convert 1-5 to 0-1
        else:
            final_scores[trait] = 0.5  # Default neutral score
    
    return final_scores

def render_psychometric_results(results: Dict):
    """Render psychometric test results"""
    st.markdown("#### Your Personality Profile")
    
    trait_scores = results['trait_scores']
    
    # Create radar chart
    fig = go.Figure()
    
    traits = list(trait_scores.keys())
    scores = [trait_scores[trait] * 100 for trait in traits]  # Convert to percentage
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=traits,
        fill='toself',
        name='Your Profile',
        line_color='#3498db'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Personality Radar Chart",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trait descriptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Trait Scores")
        for trait, score in trait_scores.items():
            percentage = score * 100
            st.metric(trait, f"{percentage:.1f}%")
    
    with col2:
        st.markdown("#### Trait Descriptions")
        for trait, description in OCEAN_TRAITS.items():
            score = trait_scores[trait] * 100
            level = "High" if score > 70 else "Medium" if score > 40 else "Low"
            st.write(f"**{trait}** ({level}): {description}")

def render_opportunity_search():
    """Render opportunity search interface"""
    st.subheader("🔍 Find Opportunities")
    
    # Search filters
    with st.expander("🔧 Search Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_filter = st.selectbox("Location", 
                                         ["All"] + list(set([opp['location'] for opp in st.session_state.mock_opportunities])))
        
        with col2:
            type_filter = st.selectbox("Apprenticeship Type",
                                     ["All"] + list(set([opp['type'] for opp in st.session_state.mock_opportunities])))
        
        with col3:
            min_salary = st.number_input("Minimum Salary (£)", min_value=0, value=16000)
    
    # Filter opportunities
    filtered_opportunities = st.session_state.mock_opportunities
    
    if location_filter != "All":
        filtered_opportunities = [opp for opp in filtered_opportunities if opp['location'] == location_filter]
    
    if type_filter != "All":
        filtered_opportunities = [opp for opp in filtered_opportunities if opp['type'] == type_filter]
    
    filtered_opportunities = [opp for opp in filtered_opportunities if opp['salary'] >= min_salary]
    
    st.markdown(f"#### Found {len(filtered_opportunities)} Opportunities")
    
    # Display opportunities
    for opp in filtered_opportunities:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{opp['title']}**")
                st.write(f"📍 {opp['location']} | 🏢 {opp['company']}")
                st.write(f"💷 £{opp['salary']:,} | ⏱️ {opp['duration']}")
                st.write(f"📋 {opp['type']}")
            
            with col2:
                st.write(f"👥 {opp['applications']} applications")
                st.write(f"📅 {opp['posted_date'].strftime('%d %b %Y')}")
            
            with col3:
                if st.button("Apply Now", key=f"apply_{opp['id']}"):
                    st.success("Application submitted!")
                if st.button("Save", key=f"save_{opp['id']}"):
                    st.info("Opportunity saved!")
            
            st.markdown("---")

def render_ai_profile_analyzer():
    """Render AI profile analysis interface"""
    st.subheader("🤖 AI Profile Analyzer")
    st.markdown("Upload your CV, record audio, or enter text to get AI-powered insights about your profile.")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload File", "🎙️ Record Audio", "✍️ Enter Text", "🎥 Upload Video"]
    )
    
    analyzer = st.session_state.ai_analyzer
    analysis_results = None
    
    if input_method == "📄 Upload File":
        uploaded_file = st.file_uploader(
            "Upload your CV or document",
            type=['pdf', 'txt', 'docx'],
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Document", type="primary"):
                with st.spinner("Analyzing document..."):
                    if uploaded_file.type == "application/pdf":
                        analysis_results = analyzer.process_pdf(uploaded_file)
                    else:
                        # For demo, treat as text
                        content = str(uploaded_file.read(), "utf-8")
                        analysis_results = analyzer.process_text(content)
    
    elif input_method == "🎙️ Record Audio":
        st.info("🎙️ Audio recording feature (demo mode)")
        
        # Mock audio recording interface
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Start Recording"):
                st.session_state.audio_recording = True
        with col2:
            if st.button("⏹️ Stop & Analyze"):
                if st.button("Process Audio", type="primary"):
                    with st.spinner("Processing audio..."):
                        analysis_results = analyzer.process_audio("mock_audio.wav")
    
    elif input_method == "✍️ Enter Text":
        text_input = st.text_area(
            "Enter your profile text, CV content, or personal statement:",
            height=200,
            placeholder="Tell us about yourself, your skills, education, and career goals..."
        )
        
        if text_input and st.button("🔍 Analyze Text", type="primary"):
            with st.spinner("Analyzing text..."):
                analysis_results = analyzer.process_text(text_input)
    
    elif input_method == "🎥 Upload Video":
        uploaded_video = st.file_uploader(
            "Upload your video profile",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video file (max 10MB)"
        )
        
        if uploaded_video is not None:
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Processing video..."):
                    analysis_results = analyzer.process_video(uploaded_video)
    
    # Display analysis results
    if analysis_results:
        render_analysis_results(analysis_results)

def render_analysis_results(results: Dict[str, Any]):
    """Render AI analysis results"""
    st.markdown("## 📊 Analysis Results")
    
    # Transcript section
    if 'transcript' in results:
        with st.expander("📝 Extracted Text", expanded=False):
            st.text_area("Transcript", value=results['transcript'], height=150, disabled=True)
    
    # Entities section
    if 'entities' in results:
        st.markdown("### 🏷️ Extracted Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Skills**")
            skills = results['entities'].get('skills', [])
            if skills:
                for skill in skills:
                    st.write(f"• {skill.title()}")
            else:
                st.write("No skills detected")
        
        with col2:
            st.markdown("**Languages**")
            languages = results['entities'].get('languages', [])
            if languages:
                for lang in languages:
                    st.write(f"• {lang.title()}")
            else:
                st.write("No languages detected")
        
        with col3:
            st.markdown("**Education**")
            education = results['entities'].get('education', [])
            if education:
                for edu in education:
                    st.write(f"• {edu.title()}")
            else:
                st.write("No education info detected")
    
    # Personality analysis
    if 'personality_traits' in results:
        st.markdown("### 🧠 Personality Analysis")
        
        traits = results['personality_traits']
        
        # Create personality radar chart
        fig = go.Figure()
        
        trait_names = list(traits.keys())
        trait_values = [traits[trait] * 100 for trait in trait_names]
        
        fig.add_trace(go.Scatterpolar(
            r=trait_values,
            theta=trait_names,
            fill='toself',
            name='Personality Profile',
            line_color='#e74c3c'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="AI-Detected Personality Traits",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trait breakdown
        col1, col2 = st.columns(2)
        with col1:
            for i, (trait, score) in enumerate(traits.items()):
                if i < 3:
                    st.metric(trait, f"{score*100:.1f}%")
        with col2:
            for i, (trait, score) in enumerate(traits.items()):
                if i >= 3:
                    st.metric(trait, f"{score*100:.1f}%")
    
    # Skills visualization
    if 'entities' in results and results['entities'].get('skills'):
        st.markdown("### 📈 Skills Analysis")
        
        skills = results['entities']['skills']
        skill_categories = {
            'Technical': ['python', 'javascript', 'html', 'css', 'sql', 'java', 'programming'],
            'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving'],
            'Creative': ['design', 'creativity', 'artistic']
        }
        
        categorized_skills = {'Technical': 0, 'Soft Skills': 0, 'Creative': 0, 'Other': 0}
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            for category, keywords in skill_categories.items():
                if any(keyword in skill_lower for keyword in keywords):
                    categorized_skills[category] += 1
                    categorized = True
                    break
            if not categorized:
                categorized_skills['Other'] += 1
        
        # Create skills distribution chart
        fig = px.bar(
            x=list(categorized_skills.keys()),
            y=list(categorized_skills.values()),
            title="Skills Distribution by Category",
            color=list(categorized_skills.values()),
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Skill Category",
            yaxis_title="Number of Skills",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights
    if 'insights' in results:
        st.markdown("### 🔍 AI Insights")
        st.success(results['insights'])
    
    # Export results
    st.markdown("### 💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export as JSON"):
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"profile_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Save to Profile"):
            username = st.session_state.current_user
            if username not in st.session_state.apprentice_profiles:
                st.session_state.apprentice_profiles[username] = {}
            
            st.session_state.apprentice_profiles[username]['ai_analysis'] = results
            st.success("Analysis saved to your profile!")

def render_company_dashboard():
    """Render company dashboard"""
    st.title(f"🏢 Company Dashboard - {st.session_state.current_user}")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Profile", "👥 Candidates", "📊 Analytics", "💳 Subscription", "🤝 Collaboration"
    ])
    
    with tab1:
        render_company_profile()
    
    with tab2:
        render_candidate_search()
    
    with tab3:
        render_company_analytics()
    
    with tab4:
        render_subscription_management()
    
    with tab5:
        render_company_collaboration()

def render_company_profile():
    """Render company profile management"""
    st.subheader("🏢 Company Profile")
    
    username = st.session_state.current_user
    if username not in st.session_state.company_profiles:
        st.session_state.company_profiles[username] = {
            'company_info': {},
            'job_opportunities': [],
            'preferences': {}
        }
    
    profile = st.session_state.company_profiles[username]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Company Information")
        with st.form("company_info"):
            company_name = st.text_input("Company Name", 
                                       value=profile['company_info'].get('company_name', ''))
            industry = st.selectbox("Industry", 
                                  ["Technology", "Healthcare", "Finance", "Manufacturing", 
                                   "Retail", "Education", "Other"],
                                  index=0 if not profile['company_info'].get('industry') else
                                  ["Technology", "Healthcare", "Finance", "Manufacturing", 
                                   "Retail", "Education", "Other"].index(profile['company_info'].get('industry', 'Technology')))
            company_size = st.selectbox("Company Size",
                                      ["1-10", "11-50", "51-200", "201-1000", "1000+"])
            location = st.text_input("Headquarters Location",
                                   value=profile['company_info'].get('location', ''))
            website = st.text_input("Website",
                                  value=profile['company_info'].get('website', ''))
            
            if st.form_submit_button("Update Company Info"):
                profile['company_info'].update({
                    'company_name': company_name,
                    'industry': industry,
                    'company_size': company_size,
                    'location': location,
                    'website': website
                })
                st.success("Company information updated!")
    
    with col2:
        st.markdown("#### About Us")
        about_us = st.text_area("Company Description",
                               value=profile['company_info'].get('about_us', ''),
                               height=200,
                               placeholder="Tell apprentices about your company, culture, and values...")
        
        if st.button("Update Description"):
            profile['company_info']['about_us'] = about_us
            st.success("Description updated!")
        
        st.markdown("#### Services & Specializations")
        services = st.text_area("Services Offered",
                               value=profile['company_info'].get('services', ''),
                               placeholder="List your main services and areas of expertise...")
        
        if st.button("Update Services"):
            profile['company_info']['services'] = services
            st.success("Services updated!")

def render_candidate_search():
    """Render candidate search interface for companies"""
    st.subheader("👥 Candidate Search")
    
    # Get user's subscription info
    username = st.session_state.current_user
    user_info = st.session_state.users_db['companies'].get(username, {})
    subscription = user_info.get('subscription', 'BASIC')
    plan_info = SUBSCRIPTION_PLANS[subscription]
    
    st.info(f"Current Plan: **{plan_info['name']}** - Access to {plan_info['candidates_limit']} candidates")
    
    # Search filters
    with st.expander("🔍 Search Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            location_filter = st.selectbox("Location", 
                                         ["All"] + list(set([c['location'] for c in st.session_state.mock_candidates])))
        
        with col2:
            skills_filter = st.multiselect("Required Skills",
                                         ["Python", "JavaScript", "HTML/CSS", "React", "SQL", "Java", "Communication"])
        
        with col3:
            education_filter = st.selectbox("Education Level",
                                          ["All", "A-Levels", "BTEC", "Bachelor's Degree", "Diploma"])
        
        with col4:
            availability_filter = st.selectbox("Availability",
                                             ["All", "Full-time", "Part-time", "Flexible"])
    
    # Filter candidates
    candidates = st.session_state.mock_candidates.copy()
    
    if location_filter != "All":
        candidates = [c for c in candidates if c['location'] == location_filter]
    
    if skills_filter:
        candidates = [c for c in candidates if any(skill in c['skills'] for skill in skills_filter)]
    
    if education_filter != "All":
        candidates = [c for c in candidates if c['education_level'] == education_filter]
    
    if availability_filter != "All":
        candidates = [c for c in candidates if c['availability'] == availability_filter]
    
    # Apply subscription limits
    max_candidates = plan_info['candidates_limit']
    if max_candidates != float('inf'):
        candidates = candidates[:max_candidates]
    
    st.markdown(f"#### Found {len(candidates)} Candidates")
    
    # Candidate cards
    for candidate in candidates:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.markdown(f"**{candidate['name']}**")
                st.write(f"📍 {candidate['location']} | 🎂 {candidate['age']} years")
                st.write(f"🎓 {candidate['education_level']}")
                st.write(f"⏰ {candidate['availability']}")
            
            with col2:
                st.markdown("**Skills:**")
                skills_display = ", ".join(candidate['skills'][:4])
                if len(candidate['skills']) > 4:
                    skills_display += f" +{len(candidate['skills'])-4} more"
                st.write(skills_display)
                
                st.markdown("**Languages:**")
                st.write(", ".join(candidate['languages']))
            
            with col3:
                # Compatibility score
                compatibility = candidate['compatibility_score']
                color = "green" if compatibility > 0.8 else "orange" if compatibility > 0.6 else "red"
                st.metric("Match", f"{compatibility*100:.0f}%")
                
                # Status
                status_color = {"Available": "green", "In Process": "orange", "Placed": "red"}
                st.markdown(f"Status: :{status_color.get(candidate['status'], 'gray')}[{candidate['status']}]")
            
            with col4:
                if st.button("👁️ View Profile", key=f"view_{candidate['id']}"):
                    st.session_state.selected_candidate = candidate
                    st.success("Profile opened!")
                
                if st.button("⭐ Shortlist", key=f"shortlist_{candidate['id']}"):
                    st.success("Candidate shortlisted!")
                
                if st.button("💬 Message", key=f"message_{candidate['id']}"):
                    st.info("Message sent!")
            
            st.markdown("---")
    
    # Selected candidate details
    if hasattr(st.session_state, 'selected_candidate'):
        render_candidate_details(st.session_state.selected_candidate)

def render_candidate_details(candidate: Dict):
    """Render detailed candidate information"""
    st.markdown("### 👤 Candidate Details")
    
    with st.expander(f"📋 {candidate['name']} - Full Profile", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Personal Information")
            st.write(f"**Name:** {candidate['name']}")
            st.write(f"**Age:** {candidate['age']}")
            st.write(f"**Location:** {candidate['location']}")
            st.write(f"**Education:** {candidate['education_level']}")
            st.write(f"**Availability:** {candidate['availability']}")
            
            st.markdown("#### Skills & Languages")
            st.write("**Skills:**")
            for skill in candidate['skills']:
                st.write(f"• {skill}")
            
            st.write("**Languages:**")
            for lang in candidate['languages']:
                st.write(f"• {lang}")
        
        with col2:
            st.markdown("#### Personality Profile")
            
            # Create mini radar chart
            scores = candidate['personality_scores']
            fig = go.Figure()
            
            traits = list(scores.keys())
            values = [scores[trait] * 100 for trait in traits]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=traits,
                fill='toself',
                name='Personality',
                line_color='#3498db'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=300,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📧 Send Interview Invite", key="interview_invite"):
                st.success("Interview invitation sent!")
        
        with col2:
            if st.button("📄 Request Documents", key="request_docs"):
                st.success("Document request sent!")
        
        with col3:
            if st.button("⭐ Add to Favorites", key="add_favorite"):
                st.success("Added to favorites!")
        
        with col4:
            if st.button("❌ Remove from View", key="remove_view"):
                if hasattr(st.session_state, 'selected_candidate'):
                    del st.session_state.selected_candidate
                st.rerun()

def render_company_analytics():
    """Render company analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    # Generate mock analytics data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    applications = np.random.poisson(5, len(dates)).cumsum()
    interviews = np.random.poisson(2, len(dates)).cumsum()
    hires = np.random.poisson(1, len(dates)).cumsum()
    
    # Application trends
    st.markdown("#### Application Trends")
    trend_data = pd.DataFrame({
        'Date': dates,
        'Applications': applications,
        'Interviews': interviews,
        'Hires': hires
    })
    
    fig = px.line(
        trend_data,
        x='Date',
        y=['Applications', 'Interviews', 'Hires'],
        title='Application Pipeline Trend',
        labels={'value': 'Count', 'variable': 'Stage'},
        color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Skills distribution
    st.markdown("#### Top Skills in Candidate Pool")
    all_skills = [skill for cand in st.session_state.mock_candidates for skill in cand['skills']]
    skills_df = pd.DataFrame(all_skills, columns=['Skill'])
    skills_count = skills_df['Skill'].value_counts().reset_index()
    skills_count.columns = ['Skill', 'Count']
    
    fig = px.bar(
        skills_count.head(10),
        x='Skill',
        y='Count',
        title='Top 10 Skills',
        color='Count',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional distribution
    st.markdown("#### Regional Distribution")
    locations = [cand['location'] for cand in st.session_state.mock_candidates]
    loc_df = pd.DataFrame(locations, columns=['Location'])
    loc_count = loc_df['Location'].value_counts().reset_index()
    loc_count.columns = ['Location', 'Count']
    
    fig = px.pie(
        loc_count,
        names='Location',
        values='Count',
        title='Candidate Distribution by Location'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_subscription_management():
    """Render subscription management interface"""
    st.subheader("💳 Subscription Management")
    
    username = st.session_state.current_user
    user_info = st.session_state.users_db['companies'].get(username, {})
    current_plan = user_info.get('subscription', 'BASIC')
    
    st.info(f"Current Plan: **{SUBSCRIPTION_PLANS[current_plan]['name']}**")
    
    # Display plans
    col1, col2, col3 = st.columns(3)
    
    for i, (plan_id, plan) in enumerate(SUBSCRIPTION_PLANS.items()):
        col = [col1, col2, col3][i]
        with col:
            st.markdown(
                f"<div style='border: 2px solid {plan['color']}; border-radius: 10px; padding: 20px; "
                f"background-color: {'#f0f8ff' if plan_id == current_plan else '#ffffff'}; height: 400px;'>"
                f"<h3 style='color: {plan['color']};'>{plan['name']}</h3>"
                f"<h2>{format_currency(plan['price'])}/month</h2>"
                f"<p>Candidates: {'Unlimited' if plan['candidates_limit'] == float('inf') else plan['candidates_limit']}</p>"
                "<ul>", 
                unsafe_allow_html=True
            )
            
            for feature in plan['features']:
                st.markdown(f"<li>{feature}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul>", unsafe_allow_html=True)
            
            if plan_id == current_plan:
                st.success("Current Plan")
            elif st.button(f"Upgrade to {plan['name']}", key=f"upgrade_{plan_id}"):
                st.session_state.users_db['companies'][username]['subscription'] = plan_id
                st.success(f"Subscription upgraded to {plan['name']}!")
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def render_company_collaboration():
    """Render company collaboration interface"""
    st.subheader("🤝 Training Provider Collaboration")
    
    # Mock training providers
    providers = [
        {"name": "Tech Skills Academy", "location": "London", "specializations": ["IT", "Software Development"]},
        {"name": "Digital Futures Institute", "location": "Manchester", "specializations": ["Data Science", "AI"]},
        {"name": "Creative Media College", "location": "Birmingham", "specializations": ["Design", "Marketing"]}
    ]
    
    st.markdown("#### Find Training Providers")
    
    # Provider cards
    for provider in providers:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**{provider['name']}**")
                st.write(f"📍 {provider['location']}")
                st.write(f"**Specializations:** {', '.join(provider['specializations'])}")
            
            with col2:
                st.markdown("**Partnership Opportunities**")
                st.write("Apprenticeship programs, Talent pipelines, Custom training")
            
            with col3:
                if st.button("🤝 Request Partnership", key=f"partner_{provider['name']}"):
                    st.success("Partnership request sent!")
            
            st.markdown("---")

def render_training_provider_dashboard():
    """Render training provider dashboard"""
    st.title(f"🏫 Training Provider Dashboard - {st.session_state.current_user}")
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs([
        "🏫 Profile", "👥 Apprentice Management", "🤝 Company Collaboration"
    ])
    
    with tab1:
        st.subheader("Training Provider Profile")
        st.info("Training provider profile management would be implemented here")
    
    with tab2:
        st.subheader("Apprentice Management")
        st.info("Apprentice management and tracking would be implemented here")
    
    with tab3:
        st.subheader("Company Collaboration")
        st.info("Company collaboration interface would be implemented here")

def render_admin_dashboard():
    """Render admin dashboard"""
    st.title(f"🔧 Admin Dashboard - {st.session_state.current_user}")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ System Config", "👤 User Management", "📈 Analytics", "📢 Content Management"
    ])
    
    with tab1:
        st.subheader("System Configuration")
        st.info("System configuration options would be implemented here")
    
    with tab2:
        st.subheader("User Management")
        st.info("User management interface would be implemented here")
    
    with tab3:
        st.subheader("Analytics Dashboard")
        st.info("System-wide analytics would be implemented here")
    
    with tab4:
        st.subheader("Content Management")
        st.info("Content management interface would be implemented here")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Custom styling
    st.set_page_config(
        page_title="AI Apprenticeship Platform",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide Streamlit branding
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Application routing
    if not st.session_state.authenticated:
        render_login_page()
    else:
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.user_role = None
            st.rerun()
        
        st.sidebar.title(f"Welcome, {st.session_state.current_user}")
        st.sidebar.subheader(f"Role: {st.session_state.user_role.replace('_', ' ').title()}")
        
        if st.session_state.user_role == "apprentice":
            render_apprentice_dashboard()
        elif st.session_state.user_role == "company":
            render_company_dashboard()
        elif st.session_state.user_role == "training_provider":
            render_training_provider_dashboard()
        elif st.session_state.user_role == "admin":
            render_admin_dashboard()

if __name__ == "__main__":
    main()