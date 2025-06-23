import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import subprocess
import base64
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import re
import time
from datetime import datetime

# Core ML/AI imports
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Using fallback methods.")

try:
    import pymupdf  # PyMuPDF for PDF processing
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="AI Apprentice Profiler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32cd32;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fff0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .video-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .record-button {
        background-color: #ff4444;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

class GroqLLM:
    """Custom wrapper for Groq API"""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not available")
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
        
    def setup_models(self):
        """Initialize all AI models with proper error handling"""
        self.models = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Only load models if transformers is available
                with st.spinner("Loading AI models..."):
                    # Try to load Whisper for transcription
                    try:
                        self.models['whisper'] = pipeline(
                            "automatic-speech-recognition",
                            model="openai/whisper-tiny.en",  # Using smaller model for better compatibility
                            device=-1  # Force CPU usage for compatibility
                        )
                        st.success("✅ Speech recognition model loaded")
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
                        st.success("✅ Information extraction model loaded")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load NER model: {str(e)}")
                    
                    # Try to load sentiment model for personality analysis
                    try:
                        self.models['sentiment'] = pipeline(
                            "sentiment-analysis",
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            device=-1
                        )
                        st.success("✅ Sentiment analysis model loaded")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load sentiment model: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error setting up models: {str(e)}")
        else:
            st.info("🔧 Using fallback methods (Transformers not available)")
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video using ffmpeg"""
        try:
            audio_path = video_path.replace('.mp4', '.wav').replace('.mov', '.wav').replace('.webm', '.wav')
            
            # Try different ffmpeg approaches
            commands = [
                # Standard approach
                ['ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_path],
                # Alternative approach
                ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'wav', '-y', audio_path],
                # Simple approach
                ['ffmpeg', '-i', video_path, audio_path, '-y']
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and os.path.exists(audio_path):
                        return audio_path
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            st.error("Could not extract audio. Please ensure ffmpeg is installed.")
            return None
                
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper or fallback method"""
        try:
            if 'whisper' in self.models:
                with st.spinner("Transcribing audio with AI model..."):
                    result = self.models['whisper'](audio_path)
                    return result['text']
            else:
                # Fallback: return placeholder text
                st.warning("⚠️ Using fallback transcription method")
                return "Transcription not available - please enter text manually or ensure proper model setup."
                
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return f"Transcription failed: {str(e)}"
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF or fallback"""
        try:
            if PDF_PROCESSING_AVAILABLE:
                doc = pymupdf.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            else:
                st.warning("PDF processing not available - please install PyMuPDF")
                return "PDF processing not available"
                
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
            if groq_api_key and GROQ_AVAILABLE:
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

def create_personality_radar_chart(personality_data: Dict[str, float]):
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
        width=500,
        height=500
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

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.markdown('<h1 class="main-header">🧠 AI Apprentice Profiler</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive AI-powered candidate assessment using speech, text, and personality analysis**")
    
    # Initialize the profiler
    if 'profiler' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.profiler = ApprenticeProfiler()
    
    profiler = st.session_state.profiler
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
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
        if hasattr(profiler, 'models') and profiler.models:
            for model_name, model in profiler.models.items():
                if model:
                    model_status.append(f"✅ {model_name.title()}")
                else:
                    model_status.append(f"❌ {model_name.title()}")
        
        if model_status:
            for status in model_status:
                st.write(status)
        else:
            st.write("🔧 Using fallback methods")
        
        # Dependencies status
        st.markdown("**Dependencies:**")
        st.write(f"{'✅' if TRANSFORMERS_AVAILABLE else '❌'} Transformers")
        st.write(f"{'✅' if PDF_PROCESSING_AVAILABLE else '❌'} PDF Processing")
        st.write(f"{'✅' if GROQ_AVAILABLE else '❌'} Groq API")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📥 Input Section</div>', unsafe_allow_html=True)
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["🎥 Record Video Interview", "📁 Upload File", "📝 Direct Text Input"],
            help="Select how you want to provide candidate information"
        )
        
        transcript = ""
        candidate_data = {}
        
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
                
                # Processing options
                col_a, col_b = st.columns(2)
                
                with col_a:
                    process_video = st.button("🔄 Process Video", type="primary")
                
                with col_b:
                    st.info("Processing will extract audio and transcribe speech")
                
                if process_video:
                    # Save uploaded video
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_file.read())
                        video_path = tmp_file.name
                    
                    with st.spinner("Processing video interview..."):
                        progress_bar = st.progress(0)
                        
                        # Step 1: Extract audio
                        st.write("🔊 Extracting audio from video...")
                        progress_bar.progress(25)
                        audio_path = profiler.extract_audio_from_video(video_path)
                        
                        if audio_path and os.path.exists(audio_path):
                            progress_bar.progress(50)
                            
                            # Step 2: Transcribe audio
                            st.write("📝 Transcribing speech to text...")
                            transcript = profiler.transcribe_audio(audio_path)
                            progress_bar.progress(75)
                            
                            if transcript and len(transcript.strip()) > 10:
                                st.success("✅ Video processed successfully!")
                                progress_bar.progress(100)
                                
                                # Show preview of transcript
                                with st.expander("👀 Preview Transcript"):
                                    st.text_area("Transcript Preview", transcript[:500] + "...", height=100, disabled=True)
                            else:
                                st.error("❌ Could not transcribe audio. Please check audio quality or try manual input.")
                            
                            # Clean up audio file
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
                        else:
                            st.error("❌ Could not extract audio from video. Please ensure ffmpeg is installed.")
                        
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
            
            if transcript:
                word_count = len(transcript.split())
                char_count = len(transcript)
                st.caption(f"📊 Content: {word_count} words, {char_count} characters")
    
    with col2:
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if transcript and len(transcript.strip()) > 20:
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
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📄 Content", "👤 Profile", "🧠 Personality", "🎯 Insights", "📊 Summary"
            ])
            
            with tab1:
                st.subheader("📄 Full Content")
                
                # Content statistics
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Words", len(transcript.split()))
                with col_stats2:
                    st.metric("Characters", len(transcript))
                with col_stats3:
                    st.metric("Lines", len(transcript.split('\n')))
                
                # Display content
                st.text_area("Full Transcript/Content", value=transcript, height=400, disabled=True)
                
                # Download original content
                st.download_button(
                    label="💾 Download Original Content",
                    data=transcript,
                    file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with tab2:
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
                
                # Organizations
                if candidate_data.get('organizations'):
                    st.markdown("### 🏢 Organizations")
                    for org in candidate_data['organizations'][:5]:
                        st.write(f"• {org}")
                
                # Raw JSON data
                with st.expander("🔍 View Raw Extracted Data"):
                    st.json(candidate_data)
            
            with tab3:
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
            
            with tab4:
                st.subheader("🎯 AI-Generated Insights")
                
                # Display insights
                st.markdown(insights)
                
                # Additional analysis metrics
                st.markdown("---")
                st.markdown("### 📈 Quick Assessment")
                
                # Calculate overall scores
                skill_score = min(len(candidate_data.get('skills', [])) / 10, 1.0)
                personality_balance = 1 - np.std(list(personality.values()))
                content_richness = min(len(transcript.split()) / 200, 1.0)
                
                assessment_col1, assessment_col2, assessment_col3 = st.columns(3)
                
                with assessment_col1:
                    st.metric("Skill Diversity", f"{skill_score:.1%}", help="Based on number of identified skills")
                
                with assessment_col2:
                    st.metric("Personality Balance", f"{personality_balance:.1%}", help="How balanced the personality traits are")
                
                with assessment_col3:
                    st.metric("Content Richness", f"{content_richness:.1%}", help="Amount of analyzable content provided")
            
            with tab5:
                st.subheader("📊 Executive Summary")
                
                # Key metrics dashboard
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    st.metric("Skills Identified", len(candidate_data.get('skills', [])))
                
                with col_metric2:
                    dominant_trait = max(personality.items(), key=lambda x: x[1])
                    st.metric("Primary Trait", dominant_trait[0].title(), f"{dominant_trait[1]:.2f}")
                
                with col_metric3:
                    contact_completeness = sum(1 for key in ['name', 'email', 'phone'] if candidate_data.get(key))
                    st.metric("Contact Info", f"{contact_completeness}/3", "Complete" if contact_completeness == 3 else "Partial")
                
                with col_metric4:
                    content_score = len(transcript.split())
                    st.metric("Content Words", content_score)
                
                # Summary sections
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("#### 🎯 Key Strengths")
                    strengths = []
                    
                    # Top skills
                    if candidate_data.get('skills'):
                        top_skills = candidate_data['skills'][:3]
                        strengths.append(f"Technical skills: {', '.join(top_skills)}")
                    
                    # Top personality traits
                    top_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)[:2]
                    for trait, score in top_traits:
                        if score > 0.6:
                            strengths.append(f"{trait.title()}: {score:.2f}")
                    
                    for strength in strengths:
                        st.write(f"• {strength}")
                
                with summary_col2:
                    st.markdown("#### 📈 Development Areas")
                    development_areas = []
                    
                    # Low personality traits
                    low_traits = [trait for trait, score in personality.items() if score < 0.4]
                    for trait in low_traits:
                        development_areas.append(f"Develop {trait} skills")
                    
                    # General suggestions
                    if not development_areas:
                        development_areas = [
                            "Continue skill development",
                            "Expand professional network",
                            "Consider leadership opportunities"
                        ]
                    
                    for area in development_areas[:3]:
                        st.write(f"• {area}")
                
                # Download comprehensive report
                st.markdown("---")
                st.markdown("### 💾 Export Options")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # JSON Report
                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'candidate_info': candidate_data,
                        'personality_analysis': personality,
                        'transcript': transcript,
                        'ai_insights': insights,
                        'analysis_metadata': {
                            'word_count': len(transcript.split()),
                            'char_count': len(transcript),
                            'skills_count': len(candidate_data.get('skills', [])),
                            'dominant_trait': max(personality.items(), key=lambda x: x[1])[0]
                        }
                    }
                    
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"candidate_analysis_{candidate_data.get('name', 'unknown').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with export_col2:
                    # Text Summary Report
                    text_report = f"""CANDIDATE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CANDIDATE INFORMATION:
Name: {candidate_data.get('name', 'Not provided')}
Email: {candidate_data.get('email', 'Not provided')}
Phone: {candidate_data.get('phone', 'Not provided')}

SKILLS ({len(candidate_data.get('skills', []))} identified):
{', '.join(candidate_data.get('skills', ['None identified']))}

PERSONALITY ANALYSIS (OCEAN):
{chr(10).join(f'{trait.title()}: {score:.2f}' for trait, score in sorted(personality.items(), key=lambda x: x[1], reverse=True))}

AI INSIGHTS:
{insights}

ORIGINAL CONTENT:
{transcript}
"""
                    
                    st.download_button(
                        label="📝 Download Text Report",
                        data=text_report,
                        file_name=f"candidate_report_{candidate_data.get('name', 'unknown').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        else:
            # Help and instructions
            st.info("👆 Please provide candidate information using one of the input methods above to begin analysis.")
            
            st.markdown("### 🚀 How to Use This Tool")
            
            with st.expander("📹 Video Recording Tips"):
                st.markdown("""
                **For best results when recording video interviews:**
                - Ensure good lighting and clear audio quality
                - Keep the candidate visible and centered in frame
                - Record for 2-5 minutes for optimal analysis
                - Ask about background, skills, experience, and career goals
                - Speak clearly and avoid background noise
                - Use a stable camera position
                """)
            
            with st.expander("📁 File Upload Guidelines"):
                st.markdown("""
                **Supported file formats:**
                - **Audio**: MP3, WAV (for interview recordings)
                - **Video**: MP4, MOV, WEBM (will extract audio for transcription)
                - **Documents**: PDF (for resumes/CVs), TXT (for text content)
                
                **File size recommendations:**
                - Audio/Video: Up to 100MB for best performance
                - Documents: Up to 10MB
                """)
            
            with st.expander("📝 Manual Input Best Practices"):
                st.markdown("""
                **What to include for comprehensive analysis:**
                - Full name and contact information
                - Educational background and qualifications
                - Work experience and achievements
                - Technical and soft skills
                - Career goals and aspirations
                - Personal interests and hobbies
                - Any relevant certifications or projects
                """)
    
    # Footer with additional information
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**🚀 Powered by:**")
        st.markdown("- Hugging Face Transformers")
        st.markdown("- OpenAI Whisper")
        st.markdown("- Groq API")
        st.markdown("- Streamlit")
    
    with footer_col2:
        st.markdown("**🔧 AI Models:**")
        st.markdown("- Speech-to-Text: Whisper")
        st.markdown("- NER: BERT-based")
        st.markdown("- Sentiment: RoBERTa")
        st.markdown("- Insights: LLaMA3 (via Groq)")
    
    with footer_col3:
        st.markdown("**📊 Analysis Features:**")
        st.markdown("- Information Extraction")
        st.markdown("- OCEAN Personality Analysis")
        st.markdown("- Career Recommendations")
        st.markdown("- Skills Assessment")
    
    # Debug information (only show in development)
    if st.checkbox("🔧 Show Debug Info", help="Display technical information for troubleshooting"):
        with st.expander("Debug Information"):
            st.write("**Session State:**")
            st.write(f"Profiler initialized: {'profiler' in st.session_state}")
            
            if 'profiler' in st.session_state:
                st.write(f"Models loaded: {list(st.session_state.profiler.models.keys())}")
            
            st.write("**Environment:**")
            st.write(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
            st.write(f"PDF processing available: {PDF_PROCESSING_AVAILABLE}")
            st.write(f"Groq available: {GROQ_AVAILABLE}")
            
            st.write("**System Info:**")
            st.write(f"Python version: {st.__version__}")

if __name__ == "__main__":
    main()