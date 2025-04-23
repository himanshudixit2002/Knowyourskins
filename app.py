import os
import sys

# Set Google Cloud credentials path at startup - must be done before any Google Cloud imports
credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_cloud_credentials.json")
if os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    print(f"✅ Google Cloud credentials set to: {credentials_path}")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Allow HTTP for local development
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Continue with regular imports
import sqlite3
import re
import io
import uuid
import cv2
import pandas as pd
import requests
import traceback
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, session, url_for, jsonify, abort, send_file
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import dateparser
from werkzeug.utils import secure_filename
import markdown
from langchain_core.prompts import PromptTemplate
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import h5py
import gdown
import json
import jinja2
import random
import time
from deep_translator import GoogleTranslator
from textblob import TextBlob
from langdetect import detect, LangDetectException
import google.cloud.translate_v2 as translate
import tempfile
import base64
from gtts import gTTS
import io
from PIL import Image
import sys
import html
import datetime

# Advanced text-to-speech imports
try:
    import pyttsx3
    have_pyttsx3 = True
except ImportError:
    have_pyttsx3 = False

try:
    from google.cloud import texttospeech
    have_google_tts = True
except ImportError:
    have_google_tts = False

# Speech synthesis helper
try:
    import soundfile as sf
    import librosa
    import numpy as np
    from scipy import signal
    have_audio_processing = True
except ImportError:
    have_audio_processing = False

# SSML processing
import xml.etree.ElementTree as ET
from html import unescape

# New imports for enhanced chatbot functionality
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
import emoji

# Initialize NLP components with fallbacks
nlp = None
have_textblob = False
have_nltk = False
have_emoji = False

# Download necessary NLTK resources with better error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    have_nltk = True
    print("✓ NLTK resources loaded successfully")
except Exception as e:
    print(f"Warning: Could not load NLTK resources: {str(e)}")
    print("Some enhanced chatbot features may be limited.")

# Try to import and initialize spaCy
try:
    import spacy
    # Try to load the spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
        print("✓ spaCy model loaded successfully")
    except OSError:
        # Try one more way to load the model
        try:
            import en_core_web_sm
            nlp = en_core_web_sm.load()
            print("✓ spaCy model loaded via package import")
        except ImportError:
            print("× spaCy model not found. Entity extraction will be limited.")
            # Create a minimal blank model as fallback
            nlp = spacy.blank("en")
except ImportError:
    print("× spaCy not available. Entity extraction will be limited.")

# Check if TextBlob is properly initialized
try:
    # Simple test
    test_blob = TextBlob("Test sentence")
    _ = test_blob.sentiment.polarity
    have_textblob = True
    print("✓ TextBlob initialized successfully")
except Exception as e:
    print(f"× TextBlob initialization error: {str(e)}")
    print("Sentiment analysis will be limited.")

# Check if emoji is available
try:
    test_emoji = emoji.emojize(":sparkles:")
    if test_emoji != ":sparkles:":
        have_emoji = True
        print("✓ Emoji package initialized successfully")
    else:
        print("× Emoji package not working correctly")
except Exception as e:
    print(f"× Emoji package error: {str(e)}")

# Check if nlp is a blank model
is_full_spacy_model = False
if nlp:
    try:
        # Try to use a pipeline component that wouldn't be in a blank model
        is_full_spacy_model = 'ner' in nlp.pipe_names
    except:
        is_full_spacy_model = False

print(f"NLP Components Status: NLTK: {'✓' if have_nltk else '×'}, TextBlob: {'✓' if have_textblob else '×'}, "
      f"spaCy: {'✓' if nlp and is_full_spacy_model else '×'}, "
      f"Emoji: {'✓' if have_emoji else '×'}")

# -----------------------------------------------------------------------------
# Load Environment Variables and App Configuration
# -----------------------------------------------------------------------------
load_dotenv()

SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise Exception("No secret key provided in environment.")

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['TEMPLATES_AUTO_RELOAD'] = os.getenv("FLASK_ENV") == "development"
app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['DEBUG_TRANSLATIONS'] = os.getenv("DEBUG_TRANSLATIONS", "").lower() == "true"

# Define upload folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
ANNOTATIONS_FOLDER = os.path.join("static", "annotations")

# Define oiliness class mapping
class_mapping = {
    "oily": "oiliness",
    "dry": "dryness",
    "combination": "combination skin",
    "normal": "normal skin"
}

# Initialize translator for multilingual support
supported_languages = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german", 
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "zh-cn": "chinese (simplified)",
    "ja": "japanese",
    "ko": "korean",
    "ar": "arabic",
    "hi": "hindi",
    "ta": "tamil"
}

# -----------------------------------------------------------------------------
# Environment & API Configuration (Model Download, Gemini API, etc.)
# -----------------------------------------------------------------------------
file_id = "1HtlPCminjDnnc9Z5LURmWKjRJxPuEHnZ"
file_path = "./model/skin_disease_model.h5"

def is_valid_h5_file(filepath):
    try:
        with h5py.File(filepath, "r") as f:
            return True
    except OSError:
        return False

if os.path.exists(file_path) and is_valid_h5_file(file_path):
    print("✅ Model already exists and is valid. Skipping download.")
else:
    print("❌ Model file is missing or corrupt. Downloading again...")
    if os.path.exists(file_path):
        os.remove(file_path)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    if os.path.exists(file_path) and is_valid_h5_file(file_path):
        print("✅ Model Download Complete and Verified!")
    else:
        print("❌ Model Download Failed or File is Still Corrupt.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY not set. Please add it to your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

DATABASE = os.getenv("DATABASE_URL")

# -----------------------------------------------------------------------------
# Data & Model Loading
# -----------------------------------------------------------------------------
df = pd.read_csv(os.path.join("dataset", "updated_skincare_products.csv"))

rf_skin = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("OILINESS_API_KEY")
)

# -----------------------------------------------------------------------------
# Helper Function: LangChain Summarizer
# -----------------------------------------------------------------------------
def langchain_summarize(text, max_length, min_length):
    prompt_template = """
Create a professional, empathetic and helpful response to the skincare concern below.
Make your response warm, conversational and reassuring - like talking to a supportive friend who understands skin concerns.
Keep it concise—between {min_length} and {max_length} words.
Include these elements:
1. A brief acknowledgment of their concern that shows empathy
2. Clear, practical advice or solutions with specific product recommendations where appropriate
3. A conclusive statement with an actionable tip or reassurance
4. Make sure your response is complete and self-contained

-----------------------------------
{text}
-----------------------------------

Always end with a clear, complete thought and a gentle encouragement. 
Aim for a balance of warmth, scientific accuracy, and actionable advice.
"""
    prompt = PromptTemplate(
        input_variables=["text", "max_length", "min_length"],
        template=prompt_template
    ).format(text=text, max_length=max_length, min_length=min_length)
    
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    if response.status_code == 200:
        data = response.json()
        summary_text = (data.get("candidates", [{}])[0]
                          .get("content", {})
                          .get("parts", [{}])[0]
                          .get("text", ""))
        return summary_text.strip()
    else:
        return "Failed to summarize text."

# -----------------------------------------------------------------------------
# Database Functions
# -----------------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    with get_db_connection() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_doctor INTEGER DEFAULT 0
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS appointment (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            date TEXT,
            skin TEXT,
            phone TEXT,
            age TEXT,
            address TEXT,
            username TEXT,
            status TEXT DEFAULT 'pending'
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS appointment_notes (
            id INTEGER PRIMARY KEY,
            appointment_id INTEGER UNIQUE NOT NULL,
            notes TEXT NOT NULL,
            FOREIGN KEY (appointment_id) REFERENCES appointment (id)
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS survey (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            name TEXT,
            age TEXT,
            gender TEXT,
            concerns TEXT,
            acne_frequency TEXT,
            comedones_count TEXT,
            first_concern TEXT,
            cosmetics_usage TEXT,
            skin_reaction TEXT,
            skin_type TEXT,
            medications TEXT,
            skincare_routine TEXT,
            stress_level TEXT
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS skincare_routines (
            id INTEGER PRIMARY KEY,
            user_id INTEGER UNIQUE,
            morning_routine TEXT,
            night_routine TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS doctor_profiles (
            id INTEGER PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL,
            name TEXT,
            bio TEXT,
            specialization TEXT,
            experience_years INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS chatbot_feedback (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            message_id TEXT,
            rating INTEGER,
            feedback_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY,
            user_id INTEGER UNIQUE,
            language TEXT DEFAULT 'en',
            notifications_enabled INTEGER DEFAULT 0,
            preferred_theme TEXT DEFAULT 'light'
        )
        ''')
        
        conn.commit()

def ensure_database_compatibility():
    # Check if is_doctor column exists in users table
    with get_db_connection() as conn:
        cursor = conn.cursor()
        table_info = cursor.execute("PRAGMA table_info(users)").fetchall()
        column_names = [column[1] for column in table_info]
        
        if "is_doctor" not in column_names:
            print("Adding is_doctor column to users table...")
            conn.execute("ALTER TABLE users ADD COLUMN is_doctor INTEGER DEFAULT 0")
            conn.commit()

create_tables()
ensure_database_compatibility()

def insert_user(username, password, is_doctor=0):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (username, password, is_doctor) VALUES (?, ?, ?)",
                     (username, password, is_doctor))
        conn.commit()

def get_user(username):
    with get_db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user:
            return dict(user)  # Convert Row to dict
        return None

def insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count,
                           first_concern, cosmetics_usage, skin_reaction, skin_type, medications,
                           skincare_routine, stress_level):
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO survey
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetics_usage,
             skin_reaction, skin_type, medications, skincare_routine, stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern,
             cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level)
        )
        conn.commit()

def get_survey_response(user_id):
    with get_db_connection() as conn:
        survey = conn.execute("SELECT * FROM survey WHERE user_id = ?", (user_id,)).fetchone()
        if survey:
            return dict(survey)
        return None

def insert_appointment_data(name, email, date, skin, phone, age, address, status, username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO appointment (name, email, date, skin, phone, age, address, status, username)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (name, email, date, skin, phone, age, address, status, username)
        )
        conn.commit()
        return cursor.lastrowid

def find_appointments(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointments = cursor.execute("SELECT * FROM appointment WHERE username = ?", (username,)).fetchall()
        return [dict(row) for row in appointments]

def update_appointment_status(appointment_id, status):
    with get_db_connection() as conn:
        conn.execute("UPDATE appointment SET status = ? WHERE id = ?", (status, appointment_id))
        conn.commit()

def delete_appointment(appointment_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM appointment WHERE id = ?", (int(appointment_id),))
        conn.commit()

def insert_doctor_profile(conn, user_id, name, bio, specialization, experience_years):
    conn.execute(
        """INSERT INTO doctor_profiles 
        (user_id, name, bio, specialization, experience_years)
        VALUES (?, ?, ?, ?, ?)""",
        (user_id, name, bio, specialization, experience_years)
    )

def get_doctor_profile(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        profile = cursor.execute(
            "SELECT id, user_id, name, bio, specialization, experience_years FROM doctor_profiles WHERE user_id = ?", 
            (user_id,)
        ).fetchone()
        return profile if profile else None

def get_all_doctors():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        return cursor.execute("""
            SELECT u.username, dp.* 
            FROM users u 
            JOIN doctor_profiles dp ON u.id = dp.user_id 
            WHERE u.is_doctor = 1
        """).fetchall()

# -----------------------------------------------------------------------------
# AI Helper Functions
# -----------------------------------------------------------------------------
def get_gemini_recommendations(skin_conditions):
    if not skin_conditions:
        return "No skin conditions detected for analysis."
    prompt = f"""
You are a knowledgeable AI skincare expert. A user uploaded an image, and the following skin conditions were detected: {', '.join(skin_conditions)}.

Please provide a very short, simple recommendation in plain language. Briefly explain the conditions and suggest one or two key skincare ingredients or tips. Keep your response under 50 words, use a friendly tone, and avoid extra details.
"""
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}]})
    print("Gemini API response status:", response.status_code)
    print("Gemini API raw response:", response.json())
    if response.status_code == 200:
        data = response.json()
        summary_text = (data.get("candidates", [{}])[0]
                          .get("content", {})
                          .get("parts", [{}])[0]
                          .get("text", ""))
        return summary_text.strip()
    else:
        return "Failed to summarize text."

def recommend_products_based_on_classes(classes):
    recommendations = []
    df_columns_lower = [col.lower() for col in df.columns]
    USD_TO_INR = 83
    def convert_price(price):
        try:
            return round(float(price) * USD_TO_INR, 2)
        except (ValueError, TypeError):
            return "N/A"
    for skin_condition in classes:
        condition_lower = skin_condition.lower()
        if condition_lower in df_columns_lower:
            original_column = df.columns[df_columns_lower.index(condition_lower)]
            filtered = df[df[original_column] == 1][["Brand", "Name", "Price", "Ingredients"]].copy()
            filtered["Price"] = filtered["Price"].apply(convert_price)
            filtered["Ingredients"] = filtered["Ingredients"].apply(
                lambda x: ", ".join(x.split(", ")[:5]) if isinstance(x, str) else ""
            )
            products = filtered.head(5).to_dict(orient="records")
        else:
            products = []
        ai_analysis = get_gemini_recommendations([skin_condition])
        recommendations.append({
            "condition": skin_condition,
            "products": products,
            "ai_analysis": ai_analysis
        })
    return recommendations

def generate_skincare_routine(user_details):
    prompt_template = """
Based on the following skin details, please create a concise, structured, and formatted skincare routine:

- **Age:** {age}
- **Gender:** {gender}
- **Skin Type:** {skin_type}
- **Main Concerns:** {concerns}
- **Acne Frequency:** {acne_frequency}
- **Current Skincare Routine:** {skincare_routine}
- **Stress Level:** {stress_level}

**Output Format (Please follow exactly):**

🌞 **Morning Routine**  
1. Step 1  
2. Step 2  
3. Step 3  
4. Step 4  
5. Step 5  
6. Step 6  
7. Step 7  

🌙 **Night Routine**  
1. Step 1  
2. Step 2  
3. Step 3  
4. Step 4  
5. Step 5  
6. Step 6  
7. Step 7  
"""
    prompt = PromptTemplate(
        input_variables=["age", "gender", "skin_type", "concerns", "acne_frequency", "skincare_routine", "stress_level"],
        template=prompt_template
    ).format(
        age=user_details["age"],
        gender=user_details["gender"],
        skin_type=user_details["skin_type"],
        concerns=user_details["concerns"],
        acne_frequency=user_details["acne_frequency"],
        skincare_routine=user_details["skincare_routine"],
        stress_level=user_details["stress_level"]
    )
    def call_gemini_api(prompt_text):
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt_text}]}]})
        if response.status_code == 200:
            data = response.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        else:
            return "Failed to fetch routine from AI"
    bot_reply = call_gemini_api(prompt)
    if "🌙" in bot_reply:
        parts = bot_reply.split("🌙")
        morning_routine = parts[0].strip()
        night_routine = "🌙" + parts[1].strip()
    else:
        routines = bot_reply.split("\n\n")
        morning_routine = routines[0].strip() if routines else "No routine found"
        night_routine = routines[1].strip() if len(routines) > 1 else "No routine found"
    return {"morning_routine": morning_routine, "night_routine": night_routine}

def save_skincare_routine(user_id, morning_routine, night_routine):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM skincare_routines WHERE user_id = ?", (user_id,))
        existing_routine = cursor.fetchone()
        if existing_routine:
            cursor.execute(
                """UPDATE skincare_routines 
                   SET morning_routine = ?, night_routine = ?, last_updated = CURRENT_TIMESTAMP 
                   WHERE user_id = ?""",
                (morning_routine, night_routine, user_id)
            )
        else:
            cursor.execute(
                """INSERT INTO skincare_routines (user_id, morning_routine, night_routine) 
                   VALUES (?, ?, ?)""",
                (user_id, morning_routine, night_routine)
            )
        conn.commit()

def get_skincare_routine(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT morning_routine, night_routine FROM skincare_routines WHERE user_id = ?", (user_id,))
        routine = cursor.fetchone()
        if routine:
            return dict(routine)
        return {"morning_routine": "No routine found", "night_routine": "No routine found"}

# -----------------------------------------------------------------------------
# Chatbot Enhanced Functions
# -----------------------------------------------------------------------------

def detect_language(text):
    """Detect the language of the input text with enhanced accuracy for short texts"""
    try:
        # For very short text, we may need more aggressive analysis
        if len(text) < 10:
            # Special character patterns for different scripts
            language_patterns = {
                'ta': r'[\u0B80-\u0BFF]',  # Tamil
                'hi': r'[\u0900-\u097F]',  # Hindi
                'ar': r'[\u0600-\u06FF]',  # Arabic
                'zh-cn': r'[\u4E00-\u9FFF]',  # Chinese
                'ja': r'[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]',  # Japanese
                'ko': r'[\uAC00-\uD7AF\u1100-\u11FF]'  # Korean
            }
            
            # Check for script-specific characters
            for lang, pattern in language_patterns.items():
                if re.search(pattern, text):
                    return lang
        
        # Use standard detection for longer text
        detected = detect(text)
        return detected
    except LangDetectException as e:
        print(f"Language detection error: {str(e)}")
        return "en"  # Default to English on error

def smart_translate(text, target_language="en", source_language=None, prepare_for_speech=False):
    """
    Smart translation system that analyzes, preprocesses, translates, and postprocesses text for optimal output
    
    Args:
        text (str): The text to translate
        target_language (str): The target language code
        source_language (str, optional): The source language if known
        prepare_for_speech (bool): Whether to optimize output for speech synthesis
        
    Returns:
        dict: A dict containing translation results and metadata
    """
    if not text:
        return {"translated": "", "source_language": "en", "success": False}
    
    # Step 1: Clean and normalize input text
    # Remove excessive whitespace and normalize punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('..', '.').replace(',,', ',')
    
    # Step 2: Language detection
    if not source_language:
        source_language = detect_language(text)
    
    # Return original if source and target language are the same
    if source_language == target_language:
        return {
            "translated": text,
            "source_language": source_language,
            "target_language": target_language,
            "segments": [{"text": text, "type": "original"}],
            "success": True
        }
    
    # Step 3: Preprocessing for specific language pairs
    # Some segments might be better left untranslated
    segments = []
    preserve_patterns = {
        # Technical content that should remain unchanged
        "code": r'`.*?`|```[\s\S]*?```',
        # Named entities like specific product names
        "entities": r'@\w+|#\w+',
        # URLs and file paths
        "urls": r'https?://\S+|\S+\.(com|org|net|io)/\S*|/\S+/\S+\.\w+'
    }
    
    # Extract segments to preserve
    preserved_segments = []
    for seg_type, pattern in preserve_patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            preserved_segments.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "type": seg_type
            })
    
    # Sort preserved segments by start position
    preserved_segments.sort(key=lambda x: x["start"])
    
    # Build complete segment list with preserved and translatable parts
    last_end = 0
    final_segments = []
    
    for segment in preserved_segments:
        if segment["start"] > last_end:
            # Add the translatable text before this preserved segment
            final_segments.append({
                "text": text[last_end:segment["start"]],
                "type": "translatable"
            })
        
        # Add the preserved segment
        final_segments.append({
            "text": segment["text"],
            "type": segment["type"]
        })
        
        last_end = segment["end"]
    
    # Add any remaining translatable text
    if last_end < len(text):
        final_segments.append({
            "text": text[last_end:],
            "type": "translatable"
        })
    
    # Step 4: Translation with smart processing
    translated_segments = []
    
    # Log which translation service we're using
    if have_google_translate:
        print(f"Using Google Cloud Translation API for {source_language} → {target_language}")
    else:
        print(f"Using fallback translation service for {source_language} → {target_language}")
    
    for segment in final_segments:
        if segment["type"] == "translatable":
            # Only translate segments that need translation
            if segment["text"].strip():
                translated_text = translate_text(segment["text"], target_language, source_language)
                translated_segments.append({
                    "text": translated_text,
                    "type": "translated",
                    "original": segment["text"]
                })
            else:
                # For empty or whitespace-only segments
                translated_segments.append({
                    "text": segment["text"],
                    "type": "whitespace"
                })
        else:
            # Non-translatable segments remain unchanged
            translated_segments.append({
                "text": segment["text"],
                "type": segment["type"]
            })
    
    # Step 5: Prepare for display or speech
    # For display, join with spaces where needed
    translated_text = ""
    for i, segment in enumerate(translated_segments):
        # Add spacing intelligently
        if i > 0 and translated_text and segment["text"]:
            prev_type = translated_segments[i-1]["type"]
            curr_type = segment["type"]
            
            # Skip space for special cases (punctuation, etc.)
            if (curr_type == "translated" and segment["text"][0] in ',.:;?!') or \
               (prev_type == "translated" and translated_text[-1] in ',.:;?!') or \
               prev_type in ["urls", "code", "entities"] or \
               curr_type in ["urls", "code", "entities"] or \
               prev_type == "whitespace" or curr_type == "whitespace":
                # No space needed
                pass
            else:
                # Add space between segments
                translated_text += " "
        
        translated_text += segment["text"]
    
    # Step 6: Clean up final text
    # Fix potential spacing issues
    translated_text = re.sub(r'\s+', ' ', translated_text)
    translated_text = re.sub(r'\s([.,;:!?])', r'\1', translated_text)
    
    # If preparing for speech, make additional fixes
    if prepare_for_speech:
        # Create a copy of the text without SSML for display purposes
        display_text = translated_text
        
        # Check if the text already has SSML tags
        has_ssml = '<speak>' in translated_text or '</speak>' in translated_text or '<break' in translated_text or '<say-as' in translated_text
        
        if not has_ssml:
            # Add appropriate pauses for speech systems
            translated_text = translated_text.replace('. ', '. <break time="0.5s"/> ')
            translated_text = translated_text.replace('? ', '? <break time="0.5s"/> ')
            translated_text = translated_text.replace('! ', '! <break time="0.5s"/> ')
            
            # Spell out abbreviations and numbers if needed - but handle Hindi differently
            if target_language in ["ta", "ar"]:
                # These languages need help with number pronunciation
                translated_text = re.sub(r'(\d+)', r'<say-as interpret-as="cardinal">\1</say-as>', translated_text)
            elif target_language == "hi":
                # Hindi works better with direct numbers than with say-as tags
                # Just leave the numbers as-is for Hindi
                pass
            
            # Wrap in speak tags if not already present
            if not translated_text.startswith('<speak>'):
                translated_text = '<speak>' + translated_text + '</speak>'
        else:
            # If it already has SSML, just clean it up
            translated_text = preprocess_text_for_speech(translated_text, target_language)
        
        # Return both the display text and the speech-optimized text
        return {
            "translated": translated_text,
            "display_text": display_text,
            "source_language": source_language,
            "target_language": target_language,
            "segments": translated_segments,
            "success": True
        }
    
    # Return complete translation info
    return {
        "translated": translated_text,
        "source_language": source_language,
        "target_language": target_language,
        "segments": translated_segments,
        "success": True
    }

def translate_text(text, target_language="en", source_language=None):
    """Translate text to the target language using multiple services for better reliability"""
    if target_language == "en" or not text:
        return text
    
    try:
        # Add debug logging
        print(f"🌐 Translating to {target_language}: '{text[:30]}...'")
        
        # Use provided source language or detect it
        if not source_language:
            source_language = detect_language(text)
            print(f"🔍 Detected language: {source_language}")
            
        if source_language == target_language:
            print(f"⏭️ Source and target are the same ({source_language}), skipping translation")
            return text
        
        # First try Google Cloud Translation API if available
        if have_google_translate:
            try:
                client = translate.Client()
                result = client.translate(text, target_language=target_language, source_language=source_language)
                translated_text = result["translatedText"]
                
                # Log the successful translation
                print(f"✅ Used Google Cloud Translation API: {source_language} → {target_language}")
                print(f"🔤 Result: '{translated_text[:30]}...'")
                
                return translated_text
            except Exception as google_error:
                print(f"Google Cloud Translation error: {str(google_error)}")
                # Fall back to other methods if Google Cloud fails
        
        # If Google Cloud is not available or failed, try deep_translator
        try:
            # Check if both languages are supported
            if source_language in supported_languages and target_language in supported_languages:
                # Special handling for Tamil and Hindi which may need specific translation path
                if target_language in ["ta", "hi"]:
                    try:
                        # Try to translate directly from source
                        translator = GoogleTranslator(source=source_language, target=target_language)
                        translated_text = translator.translate(text)
                        print(f"✅ Used direct translation: {source_language} → {target_language}")
                        
                        # If result looks problematic, try going through English
                        if len(translated_text) < len(text) / 3:  # Very short result may indicate an error
                            # Try via English as pivot
                            print(f"⚠️ Direct translation looks short, trying via English pivot")
                            en_translator = GoogleTranslator(source=source_language, target='en')
                            en_text = en_translator.translate(text)
                            
                            to_target = GoogleTranslator(source='en', target=target_language)
                            translated_text = to_target.translate(en_text)
                            print(f"✅ Used English pivot: {source_language} → en → {target_language}")
                        
                        print(f"🔤 Result: '{translated_text[:30]}...'")
                        return translated_text
                    except Exception as e:
                        print(f"Direct translation error: {str(e)}")
                        # Fall back to English pivot method
                        try:
                            # Try via English as pivot
                            print(f"Trying via English pivot after direct translation failed")
                            en_translator = GoogleTranslator(source=source_language, target='en')
                            en_text = en_translator.translate(text)
                            
                            to_target = GoogleTranslator(source='en', target=target_language)
                            translated_text = to_target.translate(en_text)
                            print(f"✅ Used English pivot (fallback): {source_language} → en → {target_language}")
                            print(f"🔤 Result: '{translated_text[:30]}...'")
                            return translated_text
                        except Exception as pivot_error:
                            print(f"Pivot translation error: {str(pivot_error)}")
                            # Default to returning original text on double failure
                            print(f"❌ Translation failed completely, returning original text")
                            return text
                else:
                    # For other languages, direct translation is usually fine
                    try:
                        translator = GoogleTranslator(source=source_language, target=target_language)
                        translated_text = translator.translate(text)
                        print(f"✅ Used direct translation: {source_language} → {target_language}")
                        print(f"🔤 Result: '{translated_text[:30]}...'")
                        return translated_text
                    except Exception as direct_error:
                        print(f"Direct translation error for other language: {str(direct_error)}")
                        # Try English pivot as fallback for all languages
                        try:
                            print(f"Trying via English pivot for other language")
                            en_translator = GoogleTranslator(source=source_language, target='en')
                            en_text = en_translator.translate(text)
                            
                            to_target = GoogleTranslator(source='en', target=target_language)
                            translated_text = to_target.translate(en_text)
                            print(f"✅ Used English pivot (fallback): {source_language} → en → {target_language}")
                            print(f"🔤 Result: '{translated_text[:30]}...'")
                            return translated_text
                        except Exception as other_pivot_error:
                            print(f"Other pivot translation error: {str(other_pivot_error)}")
                            return text
            else:
                # If unsupported languages, try to use English as a pivot
                try:
                    print(f"⚠️ Unsupported language pair ({source_language} → {target_language}), trying via English")
                    en_translator = GoogleTranslator(source=source_language, target='en')
                    en_text = en_translator.translate(text)
                    
                    to_target = GoogleTranslator(source='en', target=target_language)
                    translated_text = to_target.translate(en_text)
                    print(f"✅ Used English pivot for unsupported pair: {source_language} → en → {target_language}")
                    print(f"🔤 Result: '{translated_text[:30]}...'")
                    return translated_text
                except Exception as e:
                    print(f"Pivot translation error for unsupported languages: {str(e)}")
                    # Last resort: try direct translation even with unsupported languages
                    try:
                        print(f"Last resort: trying with auto detection")
                        translator = GoogleTranslator(source='auto', target=target_language)
                        translated_text = translator.translate(text)
                        print(f"✅ Used auto-detect (last resort)")
                        print(f"🔤 Result: '{translated_text[:30]}...'")
                        return translated_text
                    except Exception as final_error:
                        print(f"Final translation attempt failed: {str(final_error)}")
                        return text
        except Exception as dt_error:
            print(f"Deep Translator error: {str(dt_error)}")
            # Last resort: if all else fails, return original text
            print(f"❌ All translation methods failed, returning original text")
            return text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def analyze_sentiment(text):
    """Analyze sentiment of the input text"""
    if not have_textblob:
        return 0  # Return a neutral sentiment score when TextBlob is not available
    
    try:
        analysis = TextBlob(text)
        # Score ranges from -1 (negative) to 1 (positive)
        score = analysis.sentiment.polarity
        
        if score <= -0.5:
            return "very_negative"
        elif -0.5 < score <= -0.1:
            return "negative"
        elif -0.1 < score < 0.1:
            return "neutral"
        elif 0.1 <= score < 0.5:
            return "positive"
        else:
            return "very_positive"
    except:
        return "neutral"  # Default to neutral on error

def analyze_uploaded_image(image_data, user_id=None):
    try:
        # Remove the data URL prefix if present
        if image_data.startswith('data:image'):
            header, encoded = image_data.split(",", 1)
            image_data = encoded

        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(image_bytes)
        
        # Create unique filename for results
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:6]}"
        
        # Directory for storing annotated images
        annotated_dir = os.path.join(app.static_folder, 'annotated_images')
        os.makedirs(annotated_dir, exist_ok=True)
        
        annotated_path = os.path.join(annotated_dir, f"{unique_filename}_annotated.jpg")
        relative_path = f"/static/annotated_images/{unique_filename}_annotated.jpg"
        
        # Process image using predict endpoint logic for skin analysis
        # This is a simplified version to generate mock results
        detected_classes = ["dry skin", "sensitivity"]
        
        # Generate AI analysis
        ai_analysis = f"Based on the analysis, I can see signs of {', '.join(detected_classes)}. " + \
                     "I recommend using gentle, hydrating products and avoiding harsh ingredients."
        
        # Get product recommendations
        recommendations = recommend_products_based_on_classes(detected_classes)
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        return {
            'analysis': ai_analysis,
            'recommendations': recommendations,
            'detected_classes': detected_classes
        }
    except Exception as e:
        print(f"Error in analyze_uploaded_image: {str(e)}")
        traceback.print_exc()
        return {"error": f"Failed to analyze image: {str(e)}"}

def generate_ai_skin_analysis(detected_classes):
    """Generate skin analysis text based on detected conditions"""
    if not detected_classes:
        return "No specific skin conditions detected. Your skin appears to be healthy."
    
    analysis = "Based on the analysis, your skin shows signs of: " + ", ".join(detected_classes)
    analysis += ". I recommend products specifically designed for these conditions."
    
    return analysis

def recommend_products_based_on_classes(detected_classes):
    """Recommend products based on detected skin conditions"""
    recommendations = []
    
    condition_product_map = {
        'acne': ['Salicylic Acid Cleanser', 'Benzoyl Peroxide Spot Treatment', 'Oil-Free Moisturizer'],
        'rosacea': ['Gentle Cleanser', 'Azelaic Acid Serum', 'Redness Relief Moisturizer'],
        'eczema': ['Ceramide Cleanser', 'Colloidal Oatmeal Lotion', 'Hydrocortisone Cream'],
        'psoriasis': ['Medicated Shampoo', 'Coal Tar Ointment', 'Salicylic Acid Lotion'],
        'hyperpigmentation': ['Vitamin C Serum', 'Niacinamide Serum', 'Retinol Cream'],
        'dry skin': ['Hyaluronic Acid Serum', 'Ceramide Moisturizer', 'Facial Oil'],
        'oily skin': ['Foaming Cleanser', 'Oil-Control Toner', 'Gel Moisturizer'],
        'sensitive skin': ['Fragrance-Free Cleanser', 'Soothing Toner', 'Hypoallergenic Moisturizer']
    }
    
    for condition in detected_classes:
        condition_lower = condition.lower()
        for key in condition_product_map:
            if key in condition_lower:
                for product in condition_product_map[key]:
                    if product not in recommendations:
                        recommendations.append(product)
    
    if not recommendations:
        recommendations = ['Gentle Cleanser', 'Moisturizer', 'Sunscreen']
    
    return recommendations[:5]  # Return top 5 recommendations

def transcribe_audio(audio_data):
    """Transcribe audio data to text using Gemini API for multilingual support"""
    try:
        # Extract base64 data if full data URI is provided
        if isinstance(audio_data, str) and audio_data.startswith('data:audio'):
            audio_data = audio_data.split(',')[1]
        
        # Use Gemini API for transcription
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        # Create prompt for audio transcription
        prompt = "Please transcribe the following audio accurately. Respond with ONLY the transcription text, no additional comments."
        
        # Send request with audio data
        response = requests.post(
            url, 
            headers=headers, 
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "audio/mp3",
                                    "data": audio_data
                                }
                            }
                        ]
                    }
                ]
            }
        )
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            transcription = (result.get("candidates", [{}])[0]
                             .get("content", {})
                             .get("parts", [{}])[0]
                             .get("text", ""))
            
            # Clean up transcription (remove any prefixes like "Transcription:" or quotes)
            transcription = re.sub(r'^(transcription:|"|\[|\()', '', transcription, flags=re.IGNORECASE).strip()
            transcription = re.sub(r'("|\.|\]|\))$', '', transcription).strip()
            
            if not transcription:
                return {"transcription": None, "error": "Could not transcribe audio. Please try again."}
                
            return {"transcription": transcription, "error": None}
        else:
            print(f"Transcription API error: {response.status_code}, {response.text}")
            return {"transcription": None, "error": f"Could not transcribe audio. API Error: {response.status_code}"}
        
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return {"transcription": None, "error": f"Error processing audio: {str(e)}"}

def generate_quick_replies(conversation_history, user_input):
    """Generate suggested quick replies based on conversation context"""
    # Default suggestions
    default_suggestions = [
        "Tell me about acne treatments",
        "How to care for dry skin?",
        "Recommend products for sensitive skin",
        "Make an appointment"
    ]
    
    # Check for specific contexts to provide relevant suggestions
    lower_input = user_input.lower()
    
    if "acne" in lower_input:
        return [
            "What causes acne?",
            "How to prevent acne?",
            "Best ingredients for acne",
            "Is my acne severe?"
        ]
    elif "dry" in lower_input and "skin" in lower_input:
        return [
            "Moisturizers for dry skin",
            "Why is my skin so dry?",
            "Home remedies for dry skin",
            "When to see a doctor for dry skin"
        ]
    elif "oily" in lower_input and "skin" in lower_input:
        return [
            "How to reduce oil production",
            "Best cleansers for oily skin",
            "Is oily skin genetic?",
            "Oily skin and acne connection"
        ]
    elif "routine" in lower_input or "regimen" in lower_input:
        return [
            "Morning skincare routine",
            "Night skincare routine",
            "How often to exfoliate?",
            "Do I need sunscreen daily?"
        ]
    elif "ingredient" in lower_input:
        return [
            "Benefits of retinol",
            "What is hyaluronic acid?",
            "AHA vs BHA",
            "Is niacinamide good for my skin?"
        ]
    elif any(word in lower_input for word in ["appointment", "doctor", "dermatologist", "visit"]):
        return [
            "Make an appointment",
            "What to expect at appointment",
            "How to prepare for appointment",
            "Virtual consultation options"
        ]
    
    # Use default suggestions if no specific context is detected
    return default_suggestions

def create_rich_response(text, conversation_history=None, user_input=None, sentiment=None):
    """Create a rich response with additional UI elements based on content"""
    response = {
        "text": text,
        "type": "text"
    }
    
    # Add quick replies/suggested responses
    if conversation_history and user_input:
        response["suggestions"] = generate_quick_replies(conversation_history, user_input)
    
    # Check for product recommendations in the text
    if re.search(r'recommend|products|try using|consider using', text, re.IGNORECASE):
        # Extract product names using simple pattern matching
        # This could be improved with NER or better extraction techniques
        product_matches = re.findall(r'(?:try|use|consider|recommend)(?:ing)? ([A-Z][A-Za-z\s\']+(?:cleanser|moisturizer|serum|cream|lotion|sunscreen|toner))', text)
        
        if product_matches:
            products = []
            for match in product_matches[:3]:  # Limit to 3 products
                # Here you would ideally look up the product in your database
                # For now, we'll create dummy data
                products.append({
                    "name": match.strip(),
                    "image": "/static/images/product_placeholder.jpg",
                    "description": f"A skincare product that might help with your concerns.",
                    "link": "#"
                })
            
            if products:
                response["products"] = products
                response["type"] = "product_recommendation"
    
    # Check for routine recommendations
    elif re.search(r'routine|regimen|steps|morning|night', text, re.IGNORECASE):
        routine_steps = []
        
        # Look for numbered steps or bullet points
        step_matches = re.findall(r'(?:\d+\.\s+|\*\s+)([^\n\.]+)', text)
        
        if step_matches:
            for step in step_matches:
                routine_steps.append(step.strip())
            
            response["routine_steps"] = routine_steps
            response["type"] = "routine"
    
    # Check for educational content about ingredients
    elif re.search(r'ingredient|benefit|contain|function', text, re.IGNORECASE):
        # Look for ingredient names
        ingredients = re.findall(r'(?:ingredient[s]? like|such as|including) ([A-Za-z\s,]+)(?:\.|\sand)', text)
        
        if ingredients:
            response["type"] = "educational"
            response["ingredients"] = [ing.strip() for ing in ingredients[0].split(',') if ing.strip()]
    
    # Add sentiment analysis results for the AI to track user satisfaction
    if sentiment:
        response["detected_sentiment"] = sentiment
    
    return response

def get_user_language_preference(user_id):
    """Get user's language preference from database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        preference = cursor.execute(
            "SELECT language FROM user_preferences WHERE user_id = ?", 
            (user_id,)
        ).fetchone()
        
        if preference:
            return preference["language"]
    
    return "en"  # Default to English

def set_user_language_preference(user_id, language_code):
    """Set user's language preference in database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM user_preferences WHERE user_id = ?", (user_id,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                "UPDATE user_preferences SET language = ? WHERE user_id = ?",
                (language_code, user_id)
            )
        else:
            cursor.execute(
                "INSERT INTO user_preferences (user_id, language) VALUES (?, ?)",
                (user_id, language_code)
            )
        conn.commit()

def save_chatbot_feedback(user_id, message_id, rating, feedback_text=""):
    """Save user feedback about chatbot responses"""
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO chatbot_feedback 
            (user_id, message_id, rating, feedback_text)
            VALUES (?, ?, ?, ?)""",
            (user_id, message_id, rating, feedback_text)
        )
        conn.commit()

def refine_response(response):
    """Post-processes the chatbot response to ensure it's professional, direct, and informative."""
    if not response:
        return "Consulting with a dermatologist is recommended for a proper assessment of your skin condition."
    
    # Trim any leading/trailing whitespace
    response = response.strip()
    
    # Remove AI self-references
    ai_references = [
        "As an AI assistant, I",
        "As a skincare assistant, I",
        "Based on the information provided,",
        "I'd like to provide some information about",
        "I'd be happy to help with that",
        "I can certainly help with that",
        "Let me answer that for you",
        "I'd recommend",
        "I suggest",
        "I think"
    ]
    
    for phrase in ai_references:
        if response.startswith(phrase):
            response = response[len(phrase):].strip()
            # Remove additional connecting words if they exist
            for connector in [", ", ". ", "! "]:
                if response.startswith(connector):
                    response = response[len(connector):].strip()
    
    # Capitalize first letter if needed
    if response and not response[0].isupper():
        response = response[0].upper() + response[1:]
    
    # Remove redundant phrases that appear twice
    duplicates = [
        "consulting with a dermatologist is recommended",
        "a dermatologist can provide",
        "for best results",
        "for optimal results"
    ]
    
    for phrase in duplicates:
        if response.lower().count(phrase) > 1:
            last_occurrence = response.lower().rindex(phrase)
            response = response[:last_occurrence] + response[last_occurrence:].replace(phrase, "", 1)
    
    # Remove exclamation marks for professional tone
    response = response.replace("!", ".")
    
    return response

def complete_answer_if_incomplete(answer):
    """Checks if the chatbot's answer is complete and continues it if necessary."""
    # Check if answer ends with proper punctuation
    if not answer.rstrip().endswith(('.', '!', '?', ':', ';', '"', ')', ']', '}')):
        # Answer appears incomplete, so continue it
        continuation_prompt = f"""
Continue the following skincare advice, making sure it ends with a complete thought and a professional, clinical conclusion.
Make the continuation factual, evidence-based, and strictly clinical without emotional language or casual phrasing.
Maintain formal medical terminology where appropriate.
Original answer: {answer}
"""
        
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": continuation_prompt}]}]})
        
        if response.status_code == 200:
            data = response.json()
            continuation = (data.get("candidates", [{}])[0]
                           .get("content", {})
                           .get("parts", [{}])[0]
                           .get("text", ""))
            return answer + " " + continuation.strip()
    
    # Add a professional closing if needed
    elif len(answer) > 50 and not any(phrase in answer.lower() for phrase in ["consult a dermatologist", "professional evaluation", "more information", "clinical evidence", "further questions"]):
        closing_prompt = f"""
Add a single concise, clinical closing sentence to this skincare advice that emphasizes medical expertise or professional consultation.
The closing should be formal, precise, and use appropriate medical terminology without emotional language.
Original complete answer: {answer}
Just provide the closing sentence, nothing else.
"""
        
        try:
            headers = {"Content-Type": "application/json"}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": closing_prompt}]}]})
            
            if response.status_code == 200:
                data = response.json()
                closing = (data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", ""))
                
                if closing and len(closing) < 100:
                    # Only add the closing if it's reasonably short
                    return answer + " " + closing.strip()
        except Exception:
            # If adding closing fails, return original answer
            pass
    
    return answer

def get_conversation_history(user_id=None, limit=10, offset=0):
    """Gets the conversation history for a user or session.
    
    Args:
        user_id: The user or session ID
        limit: Maximum number of messages to retrieve (default 10)
        offset: Number of messages to skip (for pagination)
    """
    with get_db_connection() as conn:
        if user_id:
            history = conn.execute(
                "SELECT role, text FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?", 
                (user_id, limit, offset)
            ).fetchall()
        else:
            # For non-logged-in users, use session ID
            session_id = session.get('session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
                
            history = conn.execute(
                "SELECT role, text FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?", 
                (session_id, limit, offset)
            ).fetchall()
        
        # Convert to list of dictionaries and reverse to get chronological order
        history_list = [dict(h) for h in history]
        history_list.reverse()
        return history_list

def save_conversation_message(role, text, user_id=None):
    """Saves a message to the conversation history."""
    with get_db_connection() as conn:
        if user_id:
            conn.execute(
                "INSERT INTO conversations (user_id, role, text) VALUES (?, ?, ?)",
                (user_id, role, text)
            )
        else:
            # For non-logged-in users, use session ID
            session_id = session.get('session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
                
            conn.execute(
                "INSERT INTO conversations (user_id, role, text) VALUES (?, ?, ?)",
                (session_id, role, text)
            )
        conn.commit()

def clear_conversation_history(user_id=None):
    """Clears the conversation history for a user or session."""
    with get_db_connection() as conn:
        if user_id:
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        else:
            # For non-logged-in users, use session ID
            session_id = session.get('session_id')
            if session_id:
                conn.execute("DELETE FROM conversations WHERE user_id = ?", (session_id,))
        conn.commit()

# Updated build_conversation_prompt to support advanced context handling
def build_conversation_prompt(history, user_input):
    """Builds a conversation prompt for the chatbot, including conversation history and user input.
    
    To reduce token usage, we'll only use the most recent messages for context.
    """
    
    # Basic system prompt that defines the AI's behavior
    prompt = """
You are Aura, a clinical dermatology assistant with specialized expertise in evidence-based skincare and dermatological conditions.

CLINICAL GUIDELINES:
1. Maintain a formal, clinical tone with precise medical terminology at all times
2. Present information with scientific authority and structured clinical precision
3. Provide evidence-based dermatological information with proper citations when relevant
4. Focus on active ingredients, mechanisms of action, and clinical efficacy rather than specific brands
5. Deliver concise, actionable recommendations based on dermatological best practices
6. Use formal, technical language appropriate for medical consultation while ensuring accessibility
7. Direct patients to seek professional dermatological evaluation for conditions requiring clinical diagnosis
8. Structure responses systematically with clear, logical clinical reasoning
9. IMPORTANT: Do not include the user's question in your response. Provide only your professional answer.

PROFESSIONAL TONE EXAMPLES:
Instead of: "Hyaluronic acid is amazing for plumping your skin - it attracts water like a magnet, giving you that dewy glow!"
Say: "Hyaluronic acid functions as a humectant, attracting water molecules to the epidermis and improving dermal hydration, resulting in enhanced barrier function and skin texture."

Instead of: "I'd suggest trying a gentle cleanser with benzoyl peroxide, which works wonders for clearing those pesky breakouts."
Say: "Consider incorporating a cleanser containing benzoyl peroxide (2.5-5%), which exhibits bactericidal activity against P. acnes. Clinical studies indicate efficacy in reducing inflammatory lesions when integrated into a systematic treatment approach."

Instead of: "Your routine looks great! Just add sunscreen :)"
Say: "Your regimen would benefit from the addition of broad-spectrum photoprotection. Research demonstrates that daily application of SPF 30+ significantly reduces photodamage and lowers skin cancer risk by approximately 40-50%."
"""
    
    # Add conversation history - limit to the most recent messages
    # This reduces token usage while still maintaining conversational context
    MAX_HISTORY_MESSAGES = 10
    
    if history:
        recent_history = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
        prompt += "\nPrevious conversation:\n"
        for msg in recent_history:
            if msg['role'] == 'user':
                prompt += f"User: {msg['text']}\n"
            else:
                prompt += f"Assistant: {msg['text']}\n"
    
    # Add current user input without labeling it as a "question" to avoid it being repeated
    prompt += f"\nResponse to: {user_input}\n\nYour clinical, evidence-based response:"
    
    return prompt

# -----------------------------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------------------------

@app.route("/generate_routine", methods=["POST"])
def generate_routine():
    if "username" not in session:
        return redirect(url_for("login"))
    user = get_user(session["username"])
    user_details = get_survey_response(user["id"])
    if not user_details:
        return jsonify({"error": "User details not found"})
    routine = generate_skincare_routine(user_details)
    save_skincare_routine(user["id"], routine["morning_routine"], routine["night_routine"])
    return jsonify({"message": "Routine Generated", "routine": routine})

@app.route("/")
def index():
    try:
        if "username" in session:
            # User is signed in; pass along session data for personalized content if desired.
            return render_template("index.html", logged_in=True, username=session["username"])
        else:
            # User is not signed in; render the home page without redirecting.
            return render_template("index.html", logged_in=False)
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        return render_template("error.html", error="An error occurred. Please try again later.")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        name = request.form.get("name")
        email = request.form.get("email")
        age = request.form.get("age")
        is_doctor = request.form.get("is_doctor") == "1"
        
        if not username or not password or not name:
            return render_template("register.html", error="Username, password, and name are required.")
        if get_user(username):
            return render_template("register.html", error="Username already exists.")
        
        hashed_password = generate_password_hash(password)
        with get_db_connection() as conn:
            try:
                conn.execute("BEGIN TRANSACTION")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password, is_doctor) VALUES (?, ?, ?)",
                            (username, hashed_password, 1 if is_doctor else 0))
                user_id = cursor.lastrowid
                
                conn.commit()
                return redirect(url_for("login"))
            except Exception as e:
                conn.rollback()
                return render_template("register.html", error=f"Registration failed: {str(e)}")
            
    return render_template("register.html")

@app.route("/login.html")
def login_html_redirect():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = get_user(username)
        if not user or not check_password_hash(user["password"], password):
            return render_template("login.html", error="Invalid username or password.")
        
        session["user_id"] = user["id"]
        session["username"] = username
        
        if user["is_doctor"]:
            return redirect(url_for("doctor_appointments"))
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/survey", methods=["GET", "POST"])
def survey():
    if "username" not in session:
        return redirect(url_for("login"))
        
    if request.method == "POST":
        user = get_user(session["username"])
        user_id = user["id"]
        
        # Get form data
        name = request.form.get("name", "")
        age = request.form.get("age", "")
        gender = request.form.get("gender")
        concerns = ",".join(request.form.getlist("concerns"))
        acne_frequency = request.form.get("acne_frequency")
        comedones_count = request.form.get("comedones_count")
        first_concern = request.form.get("first_concern")
        cosmetics_usage = request.form.get("cosmetics_usage")
        skin_reaction = request.form.get("skin_reaction")
        skin_type = request.form.get("skin_type_details")
        medications = request.form.get("medications")
        skincare_routine = request.form.get("skincare_routine")
        stress_level = request.form.get("stress_level")
        
        insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count,
                               first_concern, cosmetics_usage, skin_reaction, skin_type,
                               medications, skincare_routine, stress_level)
        return redirect(url_for("profile"))
        
    # Check if user already has a survey response
    user = get_user(session["username"])
    survey_response = get_survey_response(user["id"])
    
    if survey_response:
        # Pre-fill the form with existing data
        return render_template("survey.html", survey=survey_response)
    else:
        # Show empty form
        return render_template("survey.html")

@app.route("/profile")
def profile():
    try:
        if "username" in session:
            user = get_user(session["username"])
            if not user:
                return redirect(url_for("login"))
                
            # Get survey response for the user
            survey_response = get_survey_response(user["id"])
            
            # Get skincare routine for the user
            routine = get_skincare_routine(user["id"])
            
            # If user has completed the survey, show their profile
            if survey_response:
                return render_template("profile.html", survey=survey_response, routine=routine)
            else:
                # If no survey response, redirect to survey page
                return redirect(url_for("survey"))
        
        # If user is not logged in, redirect to login page
        return redirect(url_for("login"))
    except Exception as e:
        app.logger.error(f"Error in profile route: {str(e)}")
        return render_template("error.html", error="An error occurred while loading your profile. Please try again later.")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/appointment/<int:appointment_id>")
def appointment_detail(appointment_id):
    if "username" not in session:
        return redirect(url_for("login"))
    
    username = session.get("username")
    
    # Get the appointment details
    with get_db_connection() as conn:
        cursor = conn.cursor()
        appointment = cursor.execute(
            "SELECT * FROM appointment WHERE id = ? AND username = ?", 
            (appointment_id, username)
        ).fetchone()
    
    if not appointment:
        return render_template("error.html", message="Appointment not found")
    
    # Convert appointment to dictionary for easier access
    appointment = dict(appointment)
    
    # Set concerns/notes from address field for compatibility
    appointment['concerns'] = appointment['address']
    
    return render_template("appointment_detail.html", appointment=appointment)

@app.route("/update_appointment", methods=["POST"])
def update_appointment():
    if "username" not in session:
        return redirect(url_for("login"))
    user = get_user(session["username"])
    appointment_id = request.form.get("appointment_id")
    action = request.form.get("action")
    
    if not user:
        return redirect(url_for("login"))
    
    if action == "approve":
        status = "approved"
    elif action == "decline":
        status = "declined"
    else:
        status = "pending"
    
    update_appointment_status(appointment_id, status)
    return redirect(url_for("profile"))

@app.route("/delete_appointment", methods=["POST"])
def delete_appointment_route():
    if "username" not in session:
        return redirect(url_for("login"))
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json
    
    if is_ajax:
        data = request.get_json()
        appointment_id = data.get("id")
    else:
        appointment_id = request.form.get("appointment_id")
    
    if not appointment_id:
        if is_ajax:
            return jsonify({"error": "No appointment ID provided"}), 400
        else:
            return redirect(url_for("userappoint"))
    
    try:
        delete_appointment(appointment_id)
        
        if is_ajax:
            return jsonify({"message": "Appointment deleted successfully"})
        else:
            return redirect(url_for("userappoint"))
    except Exception as e:
        app.logger.error(f"Error deleting appointment: {e}")
        if is_ajax:
            return jsonify({"error": str(e)}), 500
        else:
            return redirect(url_for("userappoint"))

@app.route("/bookappointment")
def bookappointment():
    # Get list of specialists (doctors)
    specialists = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # First ensure the doctor_profiles table exists
            create_tables()
            
            try:
                doctors = cursor.execute("""
                    SELECT u.id, u.username, dp.name, dp.specialization 
                    FROM users u 
                    LEFT JOIN doctor_profiles dp ON u.id = dp.user_id 
                    WHERE u.is_doctor = 1
                """).fetchall()
                
                for doctor in doctors:
                    specialists.append({
                        "id": doctor["id"],
                        "name": doctor["name"] if doctor["name"] else doctor["username"],
                        "specialization": doctor["specialization"] if doctor["specialization"] else "General Dermatologist"
                    })
            except sqlite3.OperationalError as e:
                # If the error is about the table not existing, we've already run create_tables()
                # so this is likely due to a race condition or other issue
                app.logger.error(f"Database error in bookappointment: {str(e)}")
                # Continue with empty specialists list
    except Exception as e:
        app.logger.error(f"Error fetching doctors: {str(e)}")
        # Continue with empty specialists list, will add default below
    
    # If no specialists are found, add a default one
    if not specialists:
        specialists.append({
            "id": 1,
            "name": "Dr. Smith",
            "specialization": "General Dermatologist"
        })
    
    # Define skin types
    skin_types = ["Normal", "Dry", "Oily", "Combination", "Sensitive"]
    
    # Define time slots
    time_slots = [
        {"value": "09:00", "label": "9:00 AM"},
        {"value": "10:00", "label": "10:00 AM"},
        {"value": "11:00", "label": "11:00 AM"},
        {"value": "12:00", "label": "12:00 PM"},
        {"value": "14:00", "label": "2:00 PM"},
        {"value": "15:00", "label": "3:00 PM"},
        {"value": "16:00", "label": "4:00 PM"},
        {"value": "17:00", "label": "5:00 PM"}
    ]
    
    # Render the bookappointment.html template with necessary data
    return render_template(
        "bookappointment.html",
        specialists=specialists,
        skin_types=skin_types,
        time_slots=time_slots,
        csrf_token=lambda: ""  # Empty function for compatibility with the template
    )

@app.route("/appointment", methods=["POST"])
def appointment():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get form data
    name = request.form.get("name")
    email = request.form.get("email")
    date = request.form.get("date")
    time = request.form.get("time", "")  # Get time from form
    specialist_id = request.form.get("specialist", "")  # Get specialist ID from form
    skin = request.form.get("skin")
    phone = request.form.get("phone")
    age = request.form.get("age")
    concerns = request.form.get("concerns", "")  # Get concerns field from form
    username = session["username"]
    status = "pending"
    
    # Combine date and time for better display
    appointment_date = f"{date} {time}" if time else date
    
    # Use concerns field for address (for backward compatibility)
    address = concerns
    
    # Insert appointment data
    appointment_id = insert_appointment_data(name, email, appointment_date, skin, phone, age, address, status, username)
    
    # Check if this is an AJAX request
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify({"message": "Appointment successfully booked", "appointmentId": appointment_id})
    else:
        # If it's a regular form submission, redirect to userappointment page with success parameter
        return redirect(url_for("userappoint", success="booked"))

@app.route("/userappointment", methods=["GET"])
def userappoint():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session.get("username")
    appointments = find_appointments(username)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"all_appointments": appointments})
    return render_template("userappointment.html", all_appointments=appointments)

@app.route("/delete_user_request", methods=["POST"])
def delete_user_request():
    if "username" not in session:
        return redirect(url_for("login"))
    user = get_user(session["username"])
    appointment_id = request.form.get("appointment_id")
    
    if not user:
        return redirect(url_for("login"))
    
    delete_appointment(appointment_id)
    return redirect(url_for("userappointment"))

@app.route("/face_analysis", methods=["POST"])
def face_analysis():
    try:
        user_id = session.get("user_id") if "username" in session else None
        
        # Handle JSON data from chatbot interface
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if data and 'image_data' in data:
                result = analyze_uploaded_image(data['image_data'], user_id)
                if "error" in result:
                    return jsonify({"error": result["error"]}), 400
                return jsonify(result)
            else:
                return jsonify({"error": "No image data provided"}), 400
        
        # Handle form data with image upload
        elif request.files and 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                # Convert the uploaded file to base64
                file_data = file.read()
                encoded_data = base64.b64encode(file_data).decode('utf-8')
                result = analyze_uploaded_image(encoded_data, user_id)
                if "error" in result:
                    return jsonify({"error": result["error"]}), 400
                return jsonify(result)
            else:
                return jsonify({"error": "No image file provided"}), 400
        
        # Handle form data for appointment updates
        elif request.form and 'appointment_id' in request.form:
            if "username" not in session:
                return jsonify({"error": "Unauthorized"}), 401
            
            appointment_id = request.form.get("appointment_id")
            update_appointment_status(appointment_id, 1)  # Setting status 1 to confirm
            return jsonify({"message": "Appointment status updated after face analysis."})
        
        else:
            return jsonify({"error": "Invalid request format"}), 400
    
    except Exception as e:
        print(f"Error in face_analysis endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "An error occurred during analysis.", "details": str(e)}), 500

def get_all_appointments():
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM appointment").fetchall()

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded"}), 400
            image_file = request.files["image"]
            ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
            if not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
                return jsonify({"error": "Invalid file type. Only JPG and PNG are allowed."}), 400
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            image_filename = secure_filename(str(uuid.uuid4()) + ".jpg")
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            image_file.save(image_path)
            unique_classes = set()
            skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()
            predictions = skin_result.get("predictions", [])
            if not predictions:
                return jsonify({
                    "error": "No face detected in the uploaded image. Please upload a clear image of your face."
                }), 400
            skin_labels = [pred["class"] for pred in predictions]
            unique_classes.update(skin_labels)
            custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
            with CLIENT.use_configuration(custom_configuration):
                oiliness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")
            if not oiliness_result.get("predictions"):
                unique_classes.add("dryness")
            else:
                oiliness_classes = [
                    class_mapping.get(pred["class"], pred["class"])
                    for pred in oiliness_result.get("predictions", [])
                    if pred.get("confidence", 0) >= 0.3
                ]
                unique_classes.update(oiliness_classes)
            os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
            annotated_filename = f"annotations_{image_filename}"
            annotated_image_path = os.path.join(ANNOTATIONS_FOLDER, annotated_filename)
            img_cv = cv2.imread(image_path)
            detections = sv.Detections.from_inference(skin_result)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = box_annotator.annotate(scene=img_cv, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            cv2.imwrite(annotated_image_path, annotated_image)
            ai_analysis_text = get_gemini_recommendations(unique_classes)
            recommended_products = recommend_products_based_on_classes(list(unique_classes))
            prediction_data = {
                "classes": list(unique_classes),
                "ai_analysis": ai_analysis_text,
                "recommendations": recommended_products,
                "annotated_image": "/" + annotated_image_path.replace("\\", "/")
            }
            return jsonify(prediction_data)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "An error occurred during analysis.", "details": str(e)}), 500
    return render_template("face_analysis.html", data={})


# -----------------------------------------------------------------------------
# Skin Disease Classifier Prediction Endpoint
# -----------------------------------------------------------------------------
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
tf.keras.utils.get_custom_objects().update({'mish': layers.Activation(mish)})

SKIN_MODEL_PATH = os.path.join("model", "skin_disease_model.h5")
if os.path.exists(SKIN_MODEL_PATH):
    best_model = load_model(SKIN_MODEL_PATH)
else:
    best_model = None
    print("Model file not found at", SKIN_MODEL_PATH)

CLASSES = ['acne', 'hyperpigmentation', 'Nail_psoriasis', 'SJS-TEN', 'Vitiligo']
IMG_SIZE = (224, 224)

def predict_disease(model, image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    probabilities = np.round(predictions[0] * 100, 2)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = CLASSES[predicted_class_idx]
    formatted_probabilities = {CLASSES[i]: f"{probabilities[i]:.2f}%" for i in range(len(CLASSES))}
    return predicted_label, formatted_probabilities

def get_gemini_disease_analysis(predicted_disease):
    prompt = f"""
You are an experienced dermatology expert. A user uploaded an image and the skin disease classifier predicted the condition: "{predicted_disease}".

Please:
- Explain this condition in simple, easy-to-understand terms.
- Recommend potential treatment or skincare suggestions.
- Provide a basic skincare routine tailored for managing this condition.
- Offer lifestyle or dietary tips for overall skin health.

Keep your response concise, structured, and engaging. Use Markdown formatting, include emojis, and maintain a warm, friendly tone.
    """
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    try:
        response = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}]})
    except Exception as e:
        print("Error making request to Gemini API:", e)
        return "Failed to connect to the AI service."
    print("Gemini API response status (disease):", response.status_code)
    try:
        data = response.json()
    except Exception as e:
        print("Error decoding JSON from Gemini API response:", e)
        return "Failed to decode AI response."
    print("Gemini API raw response (disease):", data)
    if response.status_code == 200:
        try:
            candidate = data.get("candidates", [{}])[0]
            text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            if text and text.strip():
                return text.strip()
            else:
                return "AI did not return any analysis."
        except Exception as e:
            print("Error parsing Gemini API response:", e)
            return "Failed to parse AI response."
    else:
        return "Failed to get a valid response from the AI service."

@app.route("/privacy_policy")
def privacy_policy():
    return render_template("privacy_policy.html")

@app.route("/terms_of_service")
def terms_of_service():
    return render_template("terms_of_service.html")

@app.route("/skin_disease_prediction", methods=["GET", "POST"])
def skin_disease_prediction():
    if request.method == "POST":
        image_file = None
        image_data = None
        
        # Check if request is a direct file upload
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            
            ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
            if not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
                return jsonify({"error": "Invalid file type. Only JPG and PNG are allowed."}), 400
            
            # Save the uploaded file
            upload_folder = os.path.join("static", "skin_uploads")
            os.makedirs(upload_folder, exist_ok=True)
            filename = secure_filename(str(uuid.uuid4()) + "_" + image_file.filename)
            file_path = os.path.join(upload_folder, filename)
            image_file.save(file_path)
            print(f"Saved file: {file_path}")
            
        # Check if it's a base64 encoded image from camera
        elif request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if data and 'image_data' in data:
                try:
                    # Decode base64 image
                    base64_data = data['image_data'].split(',')[1] if ',' in data['image_data'] else data['image_data']
                    image_data = base64.b64decode(base64_data)
                    
                    # Save to temporary file
                    file_path = save_temp_image(image_data, prefix="camera")
                    print(f"Saved camera image: {file_path}")
                except Exception as e:
                    print(f"Error processing base64 image: {e}")
                    return jsonify({"error": f"Error processing camera image: {e}"}), 400
            else:
                return jsonify({"error": "No image data provided"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400
            
        # Verify the model is loaded
        if best_model is None:
            return jsonify({"error": "Model file not found. Please ensure the model is correctly downloaded and placed in the 'model' directory."}), 500
            
        try:
            # If real-time flag is set, optimize for speed
            is_realtime = request.args.get('realtime', 'false').lower() == 'true'
            
            if is_realtime:
                # For real-time, use faster processing path
                img_array = process_image_for_prediction(file_path)
                predicted_label, prediction_probs = predict_from_array(best_model, img_array)
                
                # For real-time, we skip generating the AI analysis to speed up response
                result = {
                    "prediction": predicted_label,
                    "probabilities": prediction_probs,
                    "image_url": "/" + file_path.replace("\\", "/")
                }
                return jsonify(result)
            else:
                # Standard path with full analysis
                predicted_label, prediction_probs = predict_disease(best_model, file_path)
                print(f"Prediction: {predicted_label}")
                
                # Generate AI analysis
                ai_analysis_text = get_gemini_disease_analysis(predicted_label)
                
                result = {
                    "prediction": predicted_label,
                    "probabilities": prediction_probs,
                    "ai_analysis": ai_analysis_text,
                    "image_url": "/" + file_path.replace("\\", "/")
                }
                
                if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.content_type == "application/json":
                    return jsonify(result)
                else:
                    return render_template("skin_disease_prediction.html", **result)
                    
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return jsonify({"error": f"An error occurred during prediction: {e}"}), 500
            
    # GET request - just render the template
    return render_template("skin_disease_prediction.html")

@app.route("/camera_predict", methods=["POST"])
def camera_predict():
    """API endpoint optimized for real-time camera prediction."""
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
        
    # Check if model is loaded
    if best_model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        # Get image from request
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if not data or 'image_data' not in data:
                return jsonify({"error": "No image data provided"}), 400
                
            # Decode base64 image
            try:
                base64_data = data['image_data'].split(',')[1] if ',' in data['image_data'] else data['image_data']
                image_data = base64.b64decode(base64_data)
                
                # Process image directly without saving
                img = Image.open(io.BytesIO(image_data))
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
                
                # Make prediction
                predictions = best_model.predict(img_array)
                probabilities = np.round(predictions[0] * 100, 2)
                predicted_class_idx = np.argmax(predictions, axis=1)[0]
                predicted_label = CLASSES[predicted_class_idx]
                
                # Format response (light version without AI analysis)
                result = {
                    "prediction": predicted_label,
                    "confidence": float(probabilities[predicted_class_idx]),
                    "timestamp": time.time()
                }
                
                return jsonify(result)
                
            except Exception as e:
                print(f"Error processing base64 image: {e}")
                return jsonify({"error": f"Error processing image: {e}"}), 400
        else:
            # Handle form data with file upload
            if "image" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
                
            image_file = request.files["image"]
            if image_file.filename == "":
                return jsonify({"error": "Empty filename"}), 400
                
            # Read image directly from memory
            file_bytes = image_file.read()
            img = Image.open(io.BytesIO(file_bytes))
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
            
            # Make prediction
            predictions = best_model.predict(img_array)
            probabilities = np.round(predictions[0] * 100, 2)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            predicted_label = CLASSES[predicted_class_idx]
            
            # Format response (light version without AI analysis)
            result = {
                "prediction": predicted_label,
                "confidence": float(probabilities[predicted_class_idx]),
                "timestamp": time.time()
            }
            
            return jsonify(result)
            
    except Exception as e:
        print(f"Error in camera prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction error: {e}"}), 500

def update_appointment_notes(appointment_id, notes):
    with get_db_connection() as conn:
        conn.execute(
            """UPDATE appointment 
            SET doctor_notes = ?, last_updated = CURRENT_TIMESTAMP 
            WHERE id = ?""", 
            (notes, appointment_id)
        )
        conn.commit()

@app.route("/doctor/appointments", methods=["GET"])
def doctor_appointments():
    try:
        if "username" not in session:
            return redirect(url_for("login"))
        
        user = get_user(session["username"])
        if not user or not user["is_doctor"]:
            return redirect(url_for("index"))
        
        # Get all appointments
        with get_db_connection() as conn:
            cursor = conn.cursor()
            appointments = cursor.execute("SELECT * FROM appointment ORDER BY date DESC").fetchall()
            appointments = [dict(appointment) for appointment in appointments]
        
        return render_template("doctor_appointments.html", appointments=appointments, active_page="appointments")
    except Exception as e:
        app.logger.error(f"Error in doctor_appointments route: {str(e)}")
        return render_template("error.html", error="An error occurred while loading doctor appointments. Please try again later.")

@app.route("/doctor/approve_appointment", methods=["POST"])
def approve_appointment():
    try:
        if "username" not in session:
            return redirect(url_for("login"))
        
        user = get_user(session["username"])
        if not user or not user["is_doctor"]:
            return redirect(url_for("index"))
        
        appointment_id = request.form.get("appointment_id")
        action = request.form.get("action")
        
        if not appointment_id:
            return render_template("error.html", error="No appointment ID provided")
        
        if action == "approve":
            status = "approved"
        elif action == "decline":
            status = "declined"
        else:
            status = "pending"
        
        update_appointment_status(appointment_id, status)
        
        return redirect(url_for("doctor_appointments"))
    except Exception as e:
        app.logger.error(f"Error in approve_appointment route: {str(e)}")
        return render_template("error.html", error="An error occurred while updating the appointment. Please try again later.")

# Inject get_user into all templates
@app.context_processor
def inject_get_user():
    return dict(get_user=get_user)

@app.errorhandler(jinja2.exceptions.UndefinedError)
def handle_jinja_undefined_error(e):
    app.logger.error(f"Jinja2 UndefinedError: {str(e)}")
    return render_template("error.html", error="A template error occurred. Please try again later."), 500

@app.errorhandler(500)
def handle_server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    return render_template("error.html", error="An internal server error occurred. Please try again later."), 500

@app.route("/setup_sample_doctors", methods=["GET"])
def setup_sample_doctors():
    setup_key = request.args.get('key')
    if not setup_key or setup_key != os.getenv("SETUP_KEY", "skincare_setup"):
        return jsonify({"error": "Unauthorized access"}), 403
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if we already have doctor users
            doctor_users = cursor.execute("SELECT * FROM users WHERE is_doctor = 1").fetchall()
            
            if not doctor_users or len(doctor_users) == 0:
                # Create sample doctor users first
                sample_doctors = [
                    {"username": "dr.smith", "password": generate_password_hash("password123"), "is_doctor": 1},
                    {"username": "dr.jones", "password": generate_password_hash("password123"), "is_doctor": 1},
                    {"username": "dr.patel", "password": generate_password_hash("password123"), "is_doctor": 1}
                ]
                
                for doctor in sample_doctors:
                    cursor.execute("""
                        INSERT INTO users (username, password, is_doctor)
                        VALUES (?, ?, ?)
                    """, (doctor["username"], doctor["password"], doctor["is_doctor"]))
                    
                    # Get the ID of the newly created user
                    user_id = cursor.lastrowid
                    
                    # Create doctor profile
                    cursor.execute("""
                        INSERT INTO doctor_profiles (user_id, name, bio, specialization, experience_years)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        f"Dr. {doctor['username'].split('.')[1].capitalize()}",
                        f"Experienced dermatologist specializing in skin conditions.",
                        ["General Dermatology", "Cosmetic Dermatology", "Pediatric Dermatology"][random.randint(0, 2)],
                        random.randint(5, 20)
                    ))
                
                conn.commit()
                return jsonify({"success": "Sample doctors have been added to the database."})
            else:
                return jsonify({"info": "Doctor users already exist in the database."})
    
    except Exception as e:
        app.logger.error(f"Error setting up sample doctors: {str(e)}")
        return jsonify({"error": f"Error setting up sample doctors: {str(e)}"}), 500

# Dictionary to store last message from each user and timestamp
last_messages = {}

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.json
        user_input = data.get("message", "").strip()
        original_input = user_input
        
        # Get user ID from session if available, otherwise use session ID
        user_id = session.get("user_id")
        session_id = session.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            session["session_id"] = session_id
        
        # Use user_id or session_id as the user identifier
        user_identifier = user_id if user_id else session_id
        
        # Check for duplicate messages within 3 seconds
        current_time = time.time()
        if user_identifier in last_messages:
            last_message, timestamp = last_messages[user_identifier]
            # If the same message was sent within 3 seconds, return the original response without processing
            if last_message == user_input and current_time - timestamp < 3:
                return jsonify({
                    "botReply": "I'm processing your request. Please wait a moment before sending the same message again.",
                    "type": "duplicate_warning",
                    "needsTypingIndicator": False
                })
        
        # Update the last message record
        last_messages[user_identifier] = (user_input, current_time)
            
        # Define user language - priority from request param, fallback to user preference or default
        requested_language = data.get("language")
        if requested_language:
            user_language = requested_language
        else:
            user_language = get_user_language_preference(user_id) if user_id else "en"
        
        # When language is set via the UI, also update user preferences
        if requested_language and user_id:
            set_user_language_preference(user_id, requested_language)
            session["preferred_language"] = requested_language
            session.modified = True
        
        # Detect input language and translate to English if needed
        detected_language = detect_language(user_input)
        should_translate = detected_language != "en" and detected_language != "unknown"
        
        # Set default sentiment if TextBlob isn't available
        sentiment = 0
        if have_textblob:
            try:
                sentiment = analyze_sentiment(user_input)
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
        
        # Determine if we should simulate typing for a more natural conversation
        # Long responses or complex questions will appear to take longer to answer
        should_simulate_typing = len(user_input) > 20 or "?" in user_input
        
        # Translate input to English if needed for processing
        if should_translate:
            # Preserve original language for later response
            try:
                user_input = translate_text(user_input, "en", source_language=detected_language)
            except Exception as translate_error:
                print(f"Translation error: {str(translate_error)}")
                # Continue with original text if translation fails
        
        # Analyze user intent and extract entities using our NLP function
        intent_analysis = analyze_user_intent(user_input)
        intent = intent_analysis["intent"]
        entities = intent_analysis["entities"]
        priority_keywords = intent_analysis["priority_keywords"]
        
        # Extract detailed skincare entities for better context
        skincare_entities = extract_skincare_entities(user_input)
        
        # Save the user's message
        save_conversation_message("user", original_input, user_id or session_id)
        
        # Load limited conversation history from database (just enough for context)
        conversation_history = get_conversation_history(user_id or session_id, limit=10, offset=0)
        
        # Handle special case: clear chat request
        if user_input.lower() in ["clear chat", "start new chat", "reset chat"]:
            clear_conversation_history(user_id or session_id)
            session["conversation_state"] = {}
            session.modified = True
            
            response_text = "I've cleared our conversation history. How can I help you today?"
            
            # Translate response if needed
            if user_language != "en":
                response_text = translate_text(response_text, user_language)
            
            return jsonify({
                "botReply": response_text,
                "type": "clear_confirmation",
                "needsTypingIndicator": False
            })
        
        # Handle feedback submission
        if user_input.startswith("/feedback"):
            parts = user_input.split(" ", 3)
            if len(parts) >= 3:
                message_id = parts[1]
                rating = int(parts[2])
                feedback_text = parts[3] if len(parts) > 3 else ""
                
                # Save feedback
                user_identifier = user_id if user_id else session_id
                save_chatbot_feedback(user_identifier, message_id, rating, feedback_text)
                
                return jsonify({
                    "botReply": "Thank you for your feedback! It helps me improve.",
                    "type": "feedback_confirmation",
                    "needsTypingIndicator": False
                })
        
        # If user asks about skin analysis or diagnosis, redirect them
        if any(term in user_input.lower() for term in ["analyze my skin", "diagnose my skin", "skin diagnosis", "analyze photo", "check my face", "analyze my face", "skin disease", "what disease", "what condition"]):
            # Save the user message
            save_conversation_message("user", user_input, user_id or session_id)
            
            # Generate and save the bot response
            bot_reply = "I can't analyze skin conditions or diagnose skin diseases through chat. Please use our dedicated Skin Analysis tool or Skin Disease Prediction tool from the main menu, or consult with a dermatologist for a professional diagnosis."
            
            # Translate if needed
            if user_language != "en":
                bot_reply = translate_text(bot_reply, user_language)
                
            save_conversation_message("assistant", bot_reply, user_id or session_id)
            
            return jsonify({
                "botReply": bot_reply,
                "type": "tool_suggestion",
                "tools": ["skin_analysis", "skin_disease_prediction"],
                "needsTypingIndicator": should_simulate_typing
            })
        
        # Handle appointment flow states
        if session.get("conversation_state", {}).get("awaiting_date"):
            try:
                parsed_date = dateparser.parse(user_input)
                if parsed_date:
                    session["conversation_state"]["date"] = parsed_date.strftime("%Y-%m-%d %H:%M")
                    session["conversation_state"]["awaiting_date"] = False
                    session["conversation_state"]["awaiting_reason"] = True
                    session.modified = True
                    
                    # Save the user message
                    save_conversation_message("user", user_input, user_id or session_id)
                    
                    # Generate and save the bot response
                    bot_reply = f"Great! Your appointment is set for {session['conversation_state']['date']}. Now, please describe the reason for your appointment."
                    save_conversation_message("assistant", bot_reply, user_id or session_id)
                    
                    return jsonify({
                        "botReply": bot_reply,
                        "type": "appointment_flow"
                    })
                else:
                    return jsonify({
                        "botReply": "I couldn't understand that date format. Please try again with a valid date (e.g., 'Next Monday at 3 PM').",
                        "type": "error"
                    })
            except Exception as date_error:
                print(f"Date parsing error: {str(date_error)}")
                return jsonify({
                    "botReply": "There was an issue processing the date. Please try again with a different format.",
                    "type": "error"
                })
                
        if session.get("conversation_state", {}).get("awaiting_reason"):
            try:
                reason = user_input
                user = get_user(session.get("username"))
                
                if not user:
                    return jsonify({"botReply": "Please log in to schedule an appointment.", "type": "error"})
                
                survey_data = get_survey_response(user["id"])
                if not survey_data:
                    return jsonify({"botReply": "Please complete your profile survey first.", "type": "error"})
                
                survey_data = dict(survey_data)
                appointment_id = insert_appointment_data(
                    name=survey_data["name"],
                    email=user["username"],
                    date=session["conversation_state"]["date"],
                    skin=survey_data["skin_type"],
                    phone=survey_data.get("phone", ""),
                    age=survey_data["age"],
                    address=reason,
                    status=False,
                    username=user["username"]
                )
                
                # Save the user message
                save_conversation_message("user", user_input, user_id or session_id)
                
                # Generate and save the bot response
                bot_reply = f"Your appointment has been successfully scheduled for {session['conversation_state']['date']} with the reason: {reason}. Your reference ID is APPT-{appointment_id}."
                save_conversation_message("assistant", bot_reply, user_id or session_id)
                
                # Reset conversation state
                session["conversation_state"] = {}
                session.modified = True
                
                return jsonify({
                    "botReply": bot_reply,
                    "type": "appointment_confirmation",
                    "appointmentId": appointment_id
                })
            except Exception as appointment_error:
                print(f"Appointment creation error: {str(appointment_error)}")
                return jsonify({
                    "botReply": "There was an issue scheduling your appointment. Please try again later or contact support.",
                    "type": "error"
                })
            
        # Handle appointment request
        if "make an appointment" in user_input.lower():
            session["conversation_state"]["awaiting_date"] = True
            session.modified = True
            
            # Save the user message
            save_conversation_message("user", user_input, user_id or session_id)
            
            # Generate and save the bot response
            bot_reply = "When would you like to schedule your appointment? (e.g., 'March 10 at 3 PM')"
            save_conversation_message("assistant", bot_reply, user_id or session_id)
            
            return jsonify({
                "botReply": bot_reply,
                "type": "appointment_flow"
            })
        
        # Handle general conversation
        # Analyze sentiment of user input
        sentiment = analyze_sentiment(user_input)
        
        # Ensure sentiment is used correctly based on return type
        sentiment_value = sentiment if isinstance(sentiment, (int, float)) else 0
        
        # Translate user input to English if it's in another language
        original_input = user_input
        if detected_language != "en" and detected_language != user_language:
            # Keep original for response but translate for processing
            user_input_for_ai = translate_text(user_input, "en")
        else:
            user_input_for_ai = user_input
            
        # Save the user message first
        save_conversation_message("user", original_input, user_id or session_id)
        
        # Build the conversation prompt with more context from our NLP analysis
        prompt = build_conversation_prompt(conversation_history, user_input_for_ai)
        
        # Add entity information to guide the model if we extracted any
        if entities or any(skincare_entities.values()):
            prompt += "\n\nContext from user query:"
            
            if "skin_condition" in entities:
                prompt += f"\n- User mentioned skin condition: {entities['skin_condition']}"
            
            if skincare_entities["skin_concerns"]:
                prompt += f"\n- Skin concerns mentioned: {', '.join(skincare_entities['skin_concerns'])}"
                
            if skincare_entities["ingredients"]:
                prompt += f"\n- Ingredients mentioned: {', '.join(skincare_entities['ingredients'])}"
                
            if skincare_entities["product_types"]:
                prompt += f"\n- Product types mentioned: {', '.join(skincare_entities['product_types'])}"
                
            if priority_keywords:
                prompt += f"\n- Priority concerns: {', '.join(priority_keywords)}"
        
        # Special handling for certain intents
        if intent == "product_recommendation":
            prompt += "\nProvide specific product recommendations with ingredients to look for. Be specific but not overwhelming."
        elif intent == "routine_help":
            prompt += "\nProvide a simple, step-by-step routine with core products needed."
        elif intent == "ingredient_question":
            prompt += "\nExplain the benefits and potential concerns about the ingredients mentioned."
        
        try:
            # Set a timeout for the API call
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
            bot_reply = response.text
            
            # Process the reply only if we got something back
            if bot_reply:
                try:
                    # Avoid overprocessing if the response is already good
                    if len(bot_reply.split()) < 30 or bot_reply[-1] not in ".!?":
                        bot_reply = langchain_summarize(bot_reply, max_length=100, min_length=40)
                    
                    # Ensure the reply is complete
                    bot_reply = complete_answer_if_incomplete(bot_reply)
                    
                    # Enhance the response with our NLP function
                    bot_reply = enhance_response_with_nlp(bot_reply, user_input, sentiment_value)
                    
                except Exception as process_error:
                    print(f"Response processing error: {str(process_error)}")
                    # If post-processing fails, still return the original response
                    pass
            else:
                bot_reply = "I'm not quite sure how to answer that. Could you rephrase your question about your skin concern? I'd love to help!"
            
            # Translate the response if needed
            original_reply = bot_reply
            if user_language != "en":
                bot_reply = translate_text(bot_reply, user_language)
                
            # Save the bot response
            save_conversation_message("assistant", bot_reply, user_id or session_id)
            
            # Create message ID for feedback
            message_id = str(uuid.uuid4())
            
            # Create rich response with additional UI elements
            rich_response = create_rich_response(bot_reply, conversation_history, original_input, sentiment_value)
            rich_response["needsTypingIndicator"] = should_simulate_typing
            rich_response["messageId"] = message_id
            
            # Add intent and entity information for advanced clients if available
            if intent != "general_question":
                rich_response["intent"] = intent
            
            if entities:
                rich_response["detected_entities"] = entities
            
            # Add original English response for debugging only when explicitly enabled
            if user_language != "en" and app.config.get('DEBUG_TRANSLATIONS', False):
                rich_response["originalEnglishText"] = original_reply
                
            return jsonify(rich_response)
        except Exception as gemini_error:
            print(f"Generative AI error: {str(gemini_error)}")
            error_message = "Looks like I'm having a moment! I can't access my skincare knowledge right now. Could you try again in a sec? I'm eager to help with your skin questions!"
            
            # Translate error message if needed
            if user_language != "en":
                error_message = translate_text(error_message, user_language)
                
            return jsonify({
                "botReply": error_message, 
                "type": "error",
                "needsTypingIndicator": False
            })
    
    except Exception as e:
        import traceback
        print(f"Chatbot route error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "botReply": "Oops! Something went wrong on my end. Let's try that again - I'm here to help with your skincare questions!", 
            "type": "error",
            "needsTypingIndicator": False
        }), 500

@app.route("/chatbot_feedback", methods=["POST"])
def submit_chatbot_feedback():
    try:
        data = request.get_json()
        message_id = data.get("messageId")
        rating = data.get("rating")
        feedback_text = data.get("feedback", "")
        
        if not message_id or rating is None:
            return jsonify({"error": "Missing required fields"}), 400
            
        # Get user ID if logged in, otherwise use session ID
        user_identifier = None
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                user_identifier = user["id"]
        
        if not user_identifier:
            user_identifier = session.get('session_id', str(uuid.uuid4()))
            session['session_id'] = user_identifier
            
        # Save the feedback
        save_chatbot_feedback(user_identifier, message_id, rating, feedback_text)
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error submitting feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_conversation_history", methods=["GET"])
def get_conversation_history_route():
    try:
        # Get pagination parameters
        limit = request.args.get('limit', 5, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get conversation from database if user is logged in
        user_id = None
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                user_id = user["id"]
        
        # Ensure session_id exists for non-logged in users
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Get limited history from database
        history = get_conversation_history(user_id or session_id, limit, offset)
        
        # Get total message count for pagination
        with get_db_connection() as conn:
            total_count = conn.execute(
                "SELECT COUNT(*) as count FROM conversations WHERE user_id = ?", 
                (user_id or session_id,)
            ).fetchone()['count']
        
        return jsonify({
            "messages": history,
            "total_count": total_count,
            "has_more": total_count > (offset + limit)
        })
    except Exception as e:
        print("Error fetching conversation history:", str(e))
        return jsonify({"messages": [], "total_count": 0, "has_more": False}), 500

@app.route("/load_more_history", methods=["GET"])
def load_more_history():
    try:
        # Get pagination parameters
        limit = request.args.get('limit', 10, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get user ID or session ID
        user_id = None
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                user_id = user["id"]
        
        # Ensure session_id exists for non-logged in users
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Get older messages from database
        history = get_conversation_history(user_id or session_id, limit, offset)
        
        return jsonify({
            "messages": history,
            "offset": offset + limit
        })
    except Exception as e:
        print("Error loading more history:", str(e))
        return jsonify({"messages": [], "offset": offset}), 500

@app.route("/set_language", methods=["POST"])
def set_language():
    try:
        data = request.get_json()
        language_code = data.get("language")
        
        if not language_code:
            return jsonify({"error": "Missing language code"}), 400
            
        # Save user preference if logged in
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                set_user_language_preference(user["id"], language_code)
        
        # Store in session for non-logged in users
        session["preferred_language"] = language_code
        session.modified = True
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error setting language: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot_page")
def chatbot_page():
    # Get user language preference
    user_language = "en"  # Default
    
    if "username" in session:
        user = get_user(session.get("username"))
        if user:
            user_language = get_user_language_preference(user["id"])
    elif "preferred_language" in session:
        user_language = session.get("preferred_language")
    
    # Supported languages for the UI
    ui_supported_languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Español"},
        {"code": "fr", "name": "Français"},
        {"code": "de", "name": "Deutsch"},
        {"code": "it", "name": "Italiano"},
        {"code": "pt", "name": "Português"},
        {"code": "ru", "name": "Русский"},
        {"code": "zh-cn", "name": "中文"},
        {"code": "ja", "name": "日本語"},
        {"code": "ko", "name": "한국어"},
        {"code": "ar", "name": "العربية"},
        {"code": "hi", "name": "हिंदी"},
        {"code": "ta", "name": "தமிழ்"}
    ]
    
    return render_template(
        "chatbot.html", 
        current_language=user_language,
        supported_languages=ui_supported_languages
    )

@app.route("/clear_conversation", methods=["POST"])
def clear_conversation():
    try:
        # Get user ID or session ID
        user_id = None
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                user_id = user["id"]
        
        # Ensure session_id exists for non-logged in users
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Clear conversation in database
        clear_conversation_history(user_id or session_id)
        
        # Clear conversation state in session (keep this small)
        session["conversation_state"] = {}
        session.modified = True
        
        return jsonify({"success": True})
    except Exception as e:
        print("Error clearing conversation:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/translate_to_tamil", methods=["POST"])
def translate_to_tamil():
    """Enhanced Tamil translation with special handling for Tamil script"""
    try:
        data = request.get_json()
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Use the smart translation system for Tamil
        result = smart_translate(text, "ta", prepare_for_speech=True)
        translated = result.get("translated", "")
        
        # Verify we have Tamil characters in result
        tamil_regex = re.compile(r'[\u0B80-\u0BFF]')  # Unicode range for Tamil
        if not tamil_regex.search(translated) and translated != text:
            return jsonify({"error": "Tamil translation produced invalid characters"}), 500
        
        return jsonify({"translated": translated})
    except Exception as e:
        print(f"Tamil translation API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/translate_to_hindi", methods=["POST"])
def translate_to_hindi():
    """Enhanced Hindi translation with special handling for Hindi script"""
    try:
        data = request.get_json()
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Use the smart translation system for Hindi
        result = smart_translate(text, "hi", prepare_for_speech=True)
        translated = result.get("translated", "")
        
        # Verify we have Hindi characters in result
        hindi_regex = re.compile(r'[\u0900-\u097F]')  # Unicode range for Hindi
        if not hindi_regex.search(translated) and translated != text:
            return jsonify({"error": "Hindi translation produced invalid characters"}), 500
        
        return jsonify({"translated": translated})
    except Exception as e:
        print(f"Hindi translation API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/translate_text", methods=["POST"])
def translate_text_endpoint():
    """Generic endpoint to translate text to any supported language with intelligent processing"""
    try:
        data = request.get_json()
        text = data.get("text")
        target_language = data.get("language", "en")
        prepare_for_speech = data.get("prepare_for_speech", False)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if target_language not in supported_languages.keys():
            return jsonify({"error": f"Unsupported language: {target_language}"}), 400
        
        # Special handling for Tamil and Hindi can now use the same smart translation
        # system, removing the need for separate endpoints
        result = smart_translate(text, target_language, prepare_for_speech=prepare_for_speech)
        
        if not result.get("translated"):
            return jsonify({"error": "Translation failed"}), 500
            
        return jsonify(result)
    except Exception as e:
        print(f"Translation API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/text_to_speech", methods=["POST"])
def text_to_speech_endpoint():
    """Generate speech from text with advanced options and multiple engines"""
    try:
        data = request.get_json()
        text = data.get("text")
        language = data.get("language", "en")
        auto_translate = data.get("auto_translate", True)
        
        # New advanced parameters
        voice_settings = data.get("voice_settings", {})
        emotion = data.get("emotion")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # First translate if needed
        if auto_translate and language != "en":
            translation_result = smart_translate(text, language, prepare_for_speech=True)
            text_for_speech = translation_result.get("translated", text)
        else:
            text_for_speech = text
        
        # Generate speech using advanced TTS
        tts_result = advanced_text_to_speech(
            text_for_speech, 
            language=language,
            voice_settings=voice_settings,
            emotion=emotion
        )
        
        if tts_result.get("success", False):
            result = {
                "audio": tts_result["audio"],
                "format": tts_result["format"],
                "text": tts_result["text"],
                "engine": tts_result["engine"]
            }
            
            # Include additional metadata for client
            if emotion:
                result["emotion"] = emotion
                
            return jsonify(result)
        else:
            return jsonify({"error": tts_result.get("error", "Unknown error")}), 500
            
    except Exception as e:
        print(f"Text-to-speech API error: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route("/stream_speech", methods=["GET"])
def stream_speech():
    """Stream the audio file directly with advanced TTS options"""
    try:
        # Get parameters from request
        text = request.args.get("text", "")
        language = request.args.get("language", "en")
        auto_translate = request.args.get("auto_translate", "true").lower() == "true"
        
        # Get advanced parameters
        emotion = request.args.get("emotion")
        voice = request.args.get("voice", "default")
        rate = float(request.args.get("rate", "1.0"))
        pitch = float(request.args.get("pitch", "1.0"))
        volume = float(request.args.get("volume", "1.0"))
        engine = request.args.get("engine", "auto")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # URL decoding might have created issues with SSML tags
        # Fix common issues with malformed SSML from URL parameters
        text = re.sub(r'<break\s+time="([^">]+)">', r'<break time="\1"/>', text)
        text = re.sub(r'<break\s+time=\\"([^"]+)\\"', r'<break time="\1"', text)
        text = re.sub(r'<break\s+time="<([^>]+)>"', r'<break time="0.5s"', text)
        
        # Fix say-as tags from URL parameters
        text = re.sub(r'<say-as\s+interpret-as="([^">]*)"([^>/]*)>', r'<say-as interpret-as="\1">', text)
        text = re.sub(r'<say-as\s+interpret-as=\\"([^"]+)\\"', r'<say-as interpret-as="\1"', text)
        
        # Specific fix for Hindi nested tags in break time attributes
        if language == "hi":
            # Find all break tags with complex content in time attribute
            text = re.sub(r'<break\s+time=[\'"]<[^>]*>[^<]*</[^>]*>[^<]*<[^>]*>[^<]*</[^>]*>s?[\'"]/?>', 
                         r'<break time="0.5s"/>', text)
            
            # Don't translate numbers in Hindi for SSML - they cause problems
            text = re.sub(r'<say-as\s+interpret-as=[\'"]cardinal[\'"]>(\d+)</say-as>', r'\1', text)
        
        # Translate if needed
        if auto_translate and language != "en":
            translation_result = smart_translate(text, language, prepare_for_speech=True)
            text_for_speech = translation_result.get("translated", text)
        else:
            text_for_speech = text
        
        # Preprocess text to ensure SSML is properly formatted
        # This function fixes malformed SSML tags
        text_for_speech = preprocess_text_for_speech(text_for_speech, language, emotion)
        
        # Log the final SSML for debugging
        print(f"Final SSML for TTS: {text_for_speech}")
        
        # Configure voice settings
        voice_settings = {
            "engine": engine,
            "voice": voice,
            "rate": rate,
            "pitch": pitch,
            "volume": volume
        }
        
        # Generate speech using advanced function
        tts_result = advanced_text_to_speech(
            text_for_speech, 
            language=language,
            voice_settings=voice_settings,
            emotion=emotion
        )
        
        if not tts_result.get("success", False):
            return jsonify({"error": tts_result.get("error", "Failed to generate speech")}), 500
        
        # Decode base64 audio data
        audio_data = base64.b64decode(tts_result["audio"])
        
        # Create BytesIO object for sending
        audio_fp = io.BytesIO(audio_data)
        audio_fp.seek(0)
        
        # Set correct MIME type based on format
        if tts_result["format"] == "mp3":
            mimetype = "audio/mpeg"
        elif tts_result["format"] == "wav":
            mimetype = "audio/wav"
        else:
            mimetype = "application/octet-stream"
        
        # Return audio file
        return send_file(
            audio_fp,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"speech.{tts_result['format']}"
        )
    except Exception as e:
        print(f"Stream speech error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# Image Processing and Optimization Functions
# -----------------------------------------------------------------------------
def process_image_for_prediction(img_data, target_size=(224, 224)):
    """Process an image from various input types for prediction."""
    if isinstance(img_data, str):  
        # If it's a file path
        img = image.load_img(img_data, target_size=target_size)
    elif isinstance(img_data, np.ndarray):  
        # If it's a numpy array (from CV2 or camera)
        img = Image.fromarray(img_data)
        img = img.resize(target_size)
    elif isinstance(img_data, bytes):  
        # If it's raw bytes
        img = Image.open(io.BytesIO(img_data))
        img = img.resize(target_size)
    else:
        raise ValueError("Unsupported image data type")
    
    # Convert to array and prepare for model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    
    return img_array

def predict_from_array(model, img_array):
    """Make a prediction from a preprocessed image array."""
    predictions = model.predict(img_array)
    probabilities = np.round(predictions[0] * 100, 2)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = CLASSES[predicted_class_idx]
    formatted_probabilities = {CLASSES[i]: f"{probabilities[i]:.2f}%" for i in range(len(CLASSES))}
    return predicted_label, formatted_probabilities

def save_temp_image(img_data, prefix="upload"):
    """Save image data to a temporary file."""
    upload_folder = os.path.join("static", "skin_uploads")
    os.makedirs(upload_folder, exist_ok=True)
    
    if isinstance(img_data, np.ndarray):
        # Convert numpy array to image
        img = Image.fromarray(img_data)
        filename = f"{prefix}_{uuid.uuid4()}.jpg"
        file_path = os.path.join(upload_folder, filename)
        img.save(file_path)
    elif isinstance(img_data, bytes):
        # Save bytes directly
        filename = f"{prefix}_{uuid.uuid4()}.jpg"
        file_path = os.path.join(upload_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(img_data)
    elif isinstance(img_data, str) and os.path.exists(img_data):
        # It's already a file path, just return it
        return img_data
    else:
        raise ValueError("Unsupported image data type")
    
    return file_path

# -----------------------------------------------------------------------------
# Enhanced NLP Functions for Chatbot
# -----------------------------------------------------------------------------
def analyze_user_intent(text):
    """
    Analyzes the user's message to determine their intent and key entities.
    Returns a dictionary with intent classification and extracted entities.
    """
    result = {
        "intent": "general_question",
        "entities": {},
        "sentiment": 0,
        "priority_keywords": []
    }
    
    # Skip processing for very short texts
    if len(text.split()) < 3:
        return result
    
    # Basic sentiment analysis with TextBlob
    if have_textblob:
        try:
            blob = TextBlob(text)
            result["sentiment"] = blob.sentiment.polarity
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
    
    # Check for skincare-specific intents
    intent_patterns = {
        "product_recommendation": r"recommend|suggest|what\s+(products?|should I use)|good\s+products?",
        "routine_help": r"routine|regimen|steps|morning|night|order",
        "skin_concern": r"acne|pimple|redness|irritation|dry|oily|sensitive|wrinkle|aging|dark spot|hyperpigmentation",
        "ingredient_question": r"ingredient|contain|retinol|vitamin c|hyaluronic|niacinamide|acid|spf",
        "comparison": r"(difference|vs|versus|compare|better)[^.]*?between",
        "how_to": r"how\s+to|how\s+do\s+I|how\s+can\s+I",
        "side_effect": r"side\s+effect|irritate|burn|sting|react|allergy"
    }
    
    # Check for matches in each intent pattern
    for intent, pattern in intent_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            result["intent"] = intent
            break
    
    # Extract entities using spaCy if available
    if nlp and is_full_spacy_model:
        try:
            doc = nlp(text)
            
            # Extract product types
            product_types = ["cleanser", "moisturizer", "serum", "toner", "sunscreen", 
                            "exfoliant", "mask", "oil", "cream", "lotion"]
            for token in doc:
                if token.text.lower() in product_types:
                    result["entities"]["product_type"] = token.text.lower()
            
            # Extract skin conditions
            skin_conditions = ["acne", "rosacea", "eczema", "psoriasis", "dermatitis", 
                              "hyperpigmentation", "wrinkles", "dryness", "oiliness"]
            for ent in doc.ents:
                if ent.text.lower() in skin_conditions:
                    result["entities"]["skin_condition"] = ent.text.lower()
            
            # Extract ingredients of interest
            ingredients = ["retinol", "vitamin c", "hyaluronic acid", "niacinamide", 
                          "salicylic acid", "benzoyl peroxide", "azelaic acid", "glycolic acid"]
            for ingredient in ingredients:
                if ingredient in text.lower():
                    if "ingredients" not in result["entities"]:
                        result["entities"]["ingredients"] = []
                    result["entities"]["ingredients"].append(ingredient)
        
        except Exception as e:
            print(f"Entity extraction error: {e}")
    else:
        # Fallback entity extraction using regex when spaCy is not available
        try:
            # Extract product types with regex
            product_types = ["cleanser", "moisturizer", "serum", "toner", "sunscreen", 
                            "exfoliant", "mask", "oil", "cream", "lotion"]
            for product in product_types:
                if re.search(r'\b' + product + r'\b', text, re.IGNORECASE):
                    result["entities"]["product_type"] = product.lower()
                    break
            
            # Extract skin conditions with regex
            skin_conditions = ["acne", "rosacea", "eczema", "psoriasis", "dermatitis", 
                              "hyperpigmentation", "wrinkles", "dryness", "oiliness"]
            for condition in skin_conditions:
                if re.search(r'\b' + condition + r'\b', text, re.IGNORECASE):
                    result["entities"]["skin_condition"] = condition.lower()
                    break
            
            # Extract ingredients with regex
            ingredients = ["retinol", "vitamin c", "hyaluronic acid", "niacinamide", 
                          "salicylic acid", "benzoyl peroxide", "azelaic acid", "glycolic acid"]
            matched_ingredients = []
            for ingredient in ingredients:
                if re.search(r'\b' + ingredient.replace(" ", r"\s+") + r'\b', text, re.IGNORECASE):
                    matched_ingredients.append(ingredient.lower())
            
            if matched_ingredients:
                result["entities"]["ingredients"] = matched_ingredients
        except Exception as e:
            print(f"Fallback entity extraction error: {e}")
    
    # Extract priority keywords for better response focus
    skincare_priorities = [
        "urgent", "help", "painful", "severe", "worse", "allergic", 
        "reaction", "irritation", "burning", "dermatologist"
    ]
    for word in skincare_priorities:
        if word in text.lower():
            result["priority_keywords"].append(word)
    
    return result

def enhance_response_with_nlp(response, user_input, sentiment=0):
    """
    Enhances the chatbot response using NLP techniques to make it more
    professional, direct, and helpful.
    """
    # Skip enhancement for very short responses
    if len(response) < 20:
        return response
    
    try:
        # Remove overly casual openings first
        response = re.sub(r'^Oh honey,?\s+', '', response, flags=re.IGNORECASE)
        
        # Adjust tone based on detected sentiment
        user_sentiment = sentiment
        if have_textblob and sentiment == 0:  # Only recalculate if not provided and TextBlob is available
            try:
                user_sentiment = TextBlob(user_input).sentiment.polarity
            except Exception:
                pass
                
        # Replace casual terms with more professional language
        casual_phrases = {
            "Oh honey": "",
            "so frustrating": "uncomfortable",
            "the worst": "problematic",
            "so uncomfortable": "uncomfortable",
            "big improvement": "improvement",
            "amazing": "effective",
            "game changer": "beneficial",
            "works wonders": "is effective",
            "really helps": "helps",
            "annoying": "uncomfortable",
            "You've got this": "",
            "Hope that helps": "",
            "Give it a try": "",
            "Remember, consistency is key": "Consistency is important"
        }
        
        for casual, professional in casual_phrases.items():
            response = re.sub(r'\b' + re.escape(casual) + r'\b', professional, response, flags=re.IGNORECASE)
        
        # Replace overly direct phrases with more professional ones
        clinical_phrases = {
            "is usually because of": "is typically caused by",
            "try adding": "incorporate",
            "add in": "include",
            "works really well": "is effective",
            "should help a lot": "should provide relief",
            "about": "approximately",
            "Using this routine": "Following this regimen",
            "check-up": "evaluation",
            "seeing a dermatologist might be a good idea": "consulting with a dermatologist is recommended"
        }
        
        for casual, professional in clinical_phrases.items():
            response = re.sub(r'\b' + re.escape(casual) + r'\b', professional, response, flags=re.IGNORECASE)
        
        # Keep some contractions for a balance of professional yet readable tone
        response = response.replace("it is ", "it's ")
        response = response.replace("that is ", "that's ")
                
        # Clean up any remnants of casual language structure and extra exclamation points
        response = response.replace("!", ".")
        response = response.replace("...", ".")
        response = response.replace("…", ".")
        
        # Add appropriate professional opening for negative sentiment - doing this after cleaning up casual language
        if user_sentiment < -0.3 and not re.match(r'^(Dry skin|This condition|Your symptoms|The symptoms)', response, re.IGNORECASE):
            professional_starts = [
                "Dry skin can be uncomfortable. ",
                "Your symptoms are common. ",
                "This is a frequent concern. ",
                "These symptoms can be addressed. ",
                "This condition is treatable. "
            ]
            response = random.choice(professional_starts) + response
        
        # Clean up any awkward grammar from replacements
        response = response.replace("is an is effective", "is effective")
        response = response.replace("is a is effective", "is effective")
        response = response.replace("is is effective", "is effective")
        response = response.replace("because it has", "due to its")
        response = response.replace(", , ", ", ")
        response = response.replace("uncomfortable. It sounds uncomfortable", "uncomfortable")
        response = response.replace("Dry skin can be uncomfortable. dry", "Dry skin can be uncomfortable. Dry")
        response = response.replace("Dry skin can be uncomfortable. Dry, flaky skin is uncomfortable", "Dry, flaky skin is uncomfortable")
        response = response.replace("Your symptoms are common. dry", "Your symptoms are common. Dry")
        response = response.replace("Your symptoms are common. Dry, flaky skin is uncomfortable", "Dry, flaky skin is a common concern")
        response = re.sub(r'([Ii]t sounds really uncomfortable\.)\s+', '', response)
        
        # Final cleanup - fix any punctuation or spacing issues
        response = response.replace(" . ", ". ")
        response = response.replace("..", ".")
        response = response.replace("  ", " ")
        
        # Clean up repeated sentences
        sentences = response.split(". ")
        unique_sentences = []
        seen_phrases = set()
        
        for s in sentences:
            s = s.strip()
            if not s:
                continue
                
            # Skip sentences that are too similar to what we've already included
            key_phrase = re.sub(r'\W+', '', s.lower())
            if key_phrase in seen_phrases or any(s.lower() in existing.lower() for existing in unique_sentences):
                continue
                
            seen_phrases.add(key_phrase)
            
            # Capitalize the first letter of each sentence
            if len(s) > 0:
                s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            unique_sentences.append(s)
                
        response = ". ".join(unique_sentences)
        
        # Ensure the response ends with a period
        if response and not response.endswith((".", "!", "?")):
            response += "."
        
    except Exception as e:
        print(f"Response enhancement error: {e}")
    
    return response

def extract_skincare_entities(text):
    """
    Extracts skincare-specific entities from text using pattern matching and NLP.
    Returns a dictionary of entity types and values.
    """
    entities = {
        "product_types": [],
        "skin_concerns": [],
        "ingredients": [],
        "brands": []
    }
    
    # Common product types
    product_patterns = [
        r'\b(cleanser|facial wash|face wash)\b',
        r'\b(moisturizer|moisturiser|cream|lotion)\b',
        r'\b(serum|essence|ampoule)\b',
        r'\b(toner|astringent)\b',
        r'\b(sunscreen|spf|sun block|sun protection)\b',
        r'\b(mask|sheet mask|clay mask|face mask)\b',
        r'\b(exfoliant|scrub|peel|exfoliator)\b'
    ]
    
    # Common skin concerns
    concern_patterns = [
        r'\b(acne|pimple|breakout|zit|blemish)\b',
        r'\b(wrinkle|fine line|aging|anti aging|anti-aging)\b',
        r'\b(dry|dryness|dehydrat(ed|ion))\b',
        r'\b(oily|oil|sebum|shine)\b',
        r'\b(sensitive|sensitivity|irritation|irritated|redness)\b',
        r'\b(dark spot|hyperpigmentation|discoloration|melasma)\b',
        r'\b(rosacea|eczema|dermatitis|psoriasis)\b'
    ]
    
    # Common skincare ingredients
    ingredient_patterns = [
        r'\b(retinol|retin-a|tretinoin|retinoid)\b',
        r'\b(vitamin c|ascorbic acid|l-ascorbic)\b',
        r'\b(hyaluronic acid|ha|sodium hyaluronate)\b',
        r'\b(niacinamide|vitamin b3)\b',
        r'\b(salicylic acid|bha)\b',
        r'\b(glycolic acid|aha|lactic acid)\b',
        r'\b(benzoyl peroxide|bp)\b',
        r'\b(ceramide|ceramides)\b',
        r'\b(peptide|peptides)\b',
        r'\b(spf|sun protection factor)\b'
    ]
    
    # Popular skincare brands
    brand_patterns = [
        r'\b(cerave|cetaphil|la roche-posay|neutrogena|clinique)\b',
        r'\b(the ordinary|inkey list|paula\'s choice|drunk elephant)\b',
        r'\b(skinceuticals|tatcha|kiehl\'s|estee lauder|lancome)\b'
    ]
    
    # Extract all entities using regex patterns
    for pattern in product_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["product_types"].extend([m.lower() for m in matches])
    
    for pattern in concern_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["skin_concerns"].extend([m.lower() for m in matches])
    
    for pattern in ingredient_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["ingredients"].extend([m.lower() for m in matches])
    
    for pattern in brand_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["brands"].extend([m.lower() for m in matches])
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

# -----------------------------------------------------------------------------
# Advanced Text-to-Speech Functions
# -----------------------------------------------------------------------------
def advanced_text_to_speech(text, language="en", voice_settings=None, emotion=None):
    """
    Enhanced text-to-speech function with multiple engine support and advanced features
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code
        voice_settings (dict): Voice customization parameters
        emotion (str): Emotion to convey (neutral, happy, sad, excited)
        
    Returns:
        dict: Result containing audio data and metadata
    """
    # Default settings if none provided
    if voice_settings is None:
        voice_settings = {
            "engine": "auto",
            "voice": "default",
            "rate": 1.0,
            "pitch": 1.0,
            "volume": 1.0
        }
    
    # Prepare result structure
    result = {
        "audio": None,
        "format": "mp3",
        "engine": voice_settings.get("engine", "auto"),
        "text": text,
        "language": language,
        "success": False
    }
    
    # Check if text has SSML - if not, preprocess it
    has_ssml = text.strip().startswith('<speak>') and text.strip().endswith('</speak>')
    if not has_ssml:
        # Add preprocessing for text - but skip if already has SSML
        text = preprocess_text_for_speech(text, language, emotion)
    
    try:
        # Try Google Cloud TTS first (highest quality)
        if have_google_tts and (result["engine"] in ["auto", "google_cloud"]):
            try:
                audio_data = generate_speech_google_cloud(
                    text, 
                    language, 
                    voice=voice_settings.get("voice", "default"),
                    pitch=voice_settings.get("pitch", 1.0),
                    speaking_rate=voice_settings.get("rate", 1.0)
                )
                
                # Update result
                audio_fp = io.BytesIO(audio_data)
                result["engine"] = "google_cloud"
                result["format"] = "mp3"
            except Exception as google_error:
                print(f"Google Cloud TTS error: {str(google_error)}")
                raise Exception(str(google_error))
        
        # Fallback to pyttsx3 for local TTS
        elif have_pyttsx3 and (result["engine"] in ["auto", "pyttsx3"]):
            # Local TTS has limited language support
            # For non-English, we might need to adjust voice
            audio_data = generate_speech_pyttsx3(
                text,
                rate=voice_settings.get("rate", 1.0),
                volume=voice_settings.get("volume", 1.0),
                voice=voice_settings.get("voice", "default")
            )
            
            # Update result
            audio_fp = io.BytesIO(audio_data)
            result["engine"] = "pyttsx3"
            result["format"] = "wav"
        
        # Final fallback to gTTS
        else:
            # Clean SSML tags for gTTS
            if has_ssml:
                # Remove SSML tags for gTTS as it doesn't support them
                clean_text = re.sub(r'<[^>]+>', '', text)
                text = clean_text.replace('<speak>', '').replace('</speak>', '')
            
            # For certain languages, we need special handling
            # Note: gTTS splits text into chunks, which can cause issues with longer texts
            audio_fp = io.BytesIO()
            tts = gTTS(text=text, lang=language[:2], slow=False)
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            result["format"] = "mp3"
    
        # Apply audio post-processing if available and needed
        if have_audio_processing and (emotion or voice_settings.get("pitch") != 1.0):
            audio_fp = post_process_audio(
                audio_fp, 
                result["format"],
                emotion=emotion,
                pitch_shift=voice_settings.get("pitch", 1.0)
            )
        
        # Encode audio data for return
        audio_fp.seek(0)
        audio_data = base64.b64encode(audio_fp.read()).decode('utf-8')
        result["audio"] = audio_data
        result["success"] = True
        
        return result
        
    except Exception as e:
        print(f"Advanced TTS error: {str(e)}")
        # Fallback to simple gTTS if all else fails
        try:
            audio_fp = io.BytesIO()
            # Remove any SSML tags for gTTS
            clean_text = re.sub(r'<[^>]+>', '', text)
            tts = gTTS(text=clean_text, lang='en', slow=False)
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_data = base64.b64encode(audio_fp.read()).decode('utf-8')
            return {
                "audio": audio_data,
                "format": "mp3",
                "engine": "gtts_fallback",
                "text": clean_text,
                "language": "en",
                "success": True,
                "error": str(e)
            }
        except Exception as fallback_error:
            return {
                "success": False,
                "error": f"TTS generation failed: {str(fallback_error)}"
            }

def preprocess_text_for_speech(text, language, emotion=None):
    """Preprocess text to improve speech quality with better SSML"""
    # First check if text already contains SSML tags
    has_ssml = '<speak>' in text or '</speak>' in text or '<break' in text or '<say-as' in text
    
    # Fix malformed SSML before anything else
    if '<' in text and '>' in text:
        # Fix nested tags in break time attributes - major issue in Hindi
        text = re.sub(r'<break\s+time=[\'"]<[^>]*>[^<]*</[^>]*>[^<]*<[^>]*>[^<]*</[^>]*>s?[\'"]/?>', r'<break time="0.5s"/>', text)
        
        # Fix nested quotes in attributes
        text = re.sub(r'<break\s+time=\\"([^"]+)\\"', r'<break time="\1"', text)
        text = re.sub(r'<break\s+time="<([^>]+)>"', r'<break time="0.5s"', text)
        
        # Fix common say-as tag issues
        text = re.sub(r'<say-as\s+interpret-as=\\"([^"]+)\\"', r'<say-as interpret-as="\1"', text)
        text = re.sub(r'<say-as\s+interpret-as="([^"]*)"([^>]*)>', r'<say-as interpret-as="\1">', text)
        
        # Fix incorrectly nested tags
        text = re.sub(r'<say-as[^>]*>[^<]*<break[^>]*>[^<]*</say-as>', 
                     lambda m: m.group(0).replace('<break', '</say-as><break').replace('>', '></say-as>'), text)
    
    # If it contains SSML, ensure it's properly wrapped in <speak> tags
    if has_ssml:
        # Clean up any malformed SSML tags
        text = re.sub(r'<([^>]+)"([^>]+)"([^>]*)>', r'<\1\'\2\'\3>', text)  # Replace double quotes with single quotes inside tags
        
        # Ensure text is wrapped in <speak> tags
        if not text.startswith('<speak>'):
            text = '<speak>' + text
        if not text.endswith('</speak>'):
            text = text + '</speak>'
            
        return text
    
    # If no SSML, apply regular processing
    # Clean up any existing SSML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Add pauses after punctuation
    text = text.replace('. ', '. <break time="0.3s"/> ')
    text = text.replace('? ', '? <break time="0.3s"/> ')
    text = text.replace('! ', '! <break time="0.3s"/> ')
    
    # Handle numbers and abbreviations
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1 \2 \3', text)  # Dates
    
    # Add emotion-specific modifications
    if emotion:
        if emotion == "happy":
            text = text.replace('!', '! ')
            if not text.endswith(('!', '.')):
                text += '!'
        elif emotion == "sad":
            text = text.replace('.', '... ')
        elif emotion == "excited":
            text = text.replace('!', '!! ')
    
    # Language-specific preprocessing
    if language == "en":
        # Common English abbreviations
        abbreviations = {
            "Dr.": "Doctor ",
            "Mr.": "Mister ",
            "Mrs.": "Misses ",
            "Ms.": "Miss ",
            "etc.": "etcetera ",
            "e.g.": "for example ",
            "i.e.": "that is "
        }
        for abbr, expanded in abbreviations.items():
            text = text.replace(abbr, expanded)
    
    # Wrap in speak tags
    text = '<speak>' + text + '</speak>'
    return text

def generate_speech_pyttsx3(text, rate=1.0, volume=1.0, voice="default"):
    """Generate speech using pyttsx3 with temp file output"""
    if not have_pyttsx3:
        raise ImportError("pyttsx3 is not installed")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', int(rate * 180))  # Default rate is ~200 words per minute
    engine.setProperty('volume', volume)
    
    # Set voice if not default
    if voice != "default":
        voices = engine.getProperty('voices')
        for v in voices:
            if voice.lower() in v.name.lower():
                engine.setProperty('voice', v.id)
                break
    
    # Use a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    
    # Read the file into memory and return the data
    with open(temp_path, 'rb') as f:
        audio_data = f.read()
    
    # Clean up the temporary file
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    
    return audio_data

def generate_speech_google_cloud(text, language, voice="default", pitch=1.0, speaking_rate=1.0):
    """Generate high-quality speech using Google Cloud Text-to-Speech"""
    if not have_google_tts:
        raise ImportError("Google Cloud Text-to-Speech library is not installed")
    
    try:
        # Explicitly create client with environment credentials
        client = texttospeech.TextToSpeechClient()
        
        # Debug - print current credentials status
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print(f"Using credentials file: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        
        # Language code mapping
        lang_map = {
            "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", 
            "it": "it-IT", "zh-cn": "cmn-CN", "ja": "ja-JP", "ko": "ko-KR",
            "hi": "hi-IN", "ta": "ta-IN", "ar": "ar-XA"
        }
        
        # Determine language and voice settings
        language_code = lang_map.get(language, "en-US")
        
        # Set the SSML voice gender
        if voice.lower() == "male":
            ssml_gender = texttospeech.SsmlVoiceGender.MALE
        elif voice.lower() == "female":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
        else:
            # Default to neutral
            ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL
        
        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=pitch * 10 - 10,  # Convert 0.5-1.5 range to -5 to +5
            speaking_rate=speaking_rate  # 1.0 is normal speed
        )
        
        # Set the text input - check for proper SSML
        is_ssml = text.strip().startswith('<speak>') and text.strip().endswith('</speak>')
        
        if is_ssml:
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)
        
        # Select voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=ssml_gender
        )
        
        # Generate speech
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content
    except Exception as e:
        print(f"Google Cloud TTS error details: {str(e)}")
        raise

def post_process_audio(audio_fp, format_type, emotion=None, pitch_shift=1.0):
    """Apply audio post-processing for better voice quality"""
    if not have_audio_processing:
        return audio_fp  # Return original if processing libs not available
    
    # First convert to floating point audio
    audio_fp.seek(0)
    audio_data = audio_fp.read()
    
    with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_in:
        temp_in_path = temp_in.name
        temp_in.write(audio_data)
    
    try:
        # Load audio with librosa to enable processing
        y, sr = librosa.load(temp_in_path, sr=None)
        
        # Apply pitch shift if needed
        if pitch_shift != 1.0:
            # Convert to semitones (logarithmic scale)
            n_steps = (pitch_shift - 1.0) * 12
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        
        # Apply emotion-based effects
        if emotion:
            if emotion == "happy":
                # Make slightly faster and brighter
                y = librosa.effects.time_stretch(y, rate=1.05)
                # Boost high frequencies
                b, a = signal.butter(4, 0.6, btype='high', analog=False)
                y = signal.filtfilt(b, a, y) * 1.2 + y * 0.8
            
            elif emotion == "sad":
                # Make slightly slower and darker
                y = librosa.effects.time_stretch(y, rate=0.95)
                # Boost low frequencies
                b, a = signal.butter(4, 0.6, btype='low', analog=False)
                y = signal.filtfilt(b, a, y) * 1.2 + y * 0.8
            
            elif emotion == "excited":
                # Faster and more dynamic
                y = librosa.effects.time_stretch(y, rate=1.1)
                # Add more dynamics
                y = np.sign(y) * np.power(np.abs(y), 0.8)
        
        # Export processed audio
        with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_out:
            temp_out_path = temp_out.name
        
        sf.write(temp_out_path, y, sr)
        
        # Read processed file back into memory
        with open(temp_out_path, 'rb') as f:
            processed_audio = f.read()
        
        # Clean up temp files
        os.unlink(temp_in_path)
        os.unlink(temp_out_path)
        
        # Return processed audio
        result = io.BytesIO(processed_audio)
        result.seek(0)
        return result
        
    except Exception as e:
        print(f"Audio post-processing error: {str(e)}")
        # Return original on error
        os.unlink(temp_in_path)
        audio_fp.seek(0)
        return audio_fp

@app.route("/available_voices", methods=["GET"])
def available_voices():
    """Get information about available TTS voices and engines"""
    language = request.args.get("language", "all")
    
    # Initialize result with basic capabilities
    result = {
        "available_engines": ["gtts"],  # gTTS is always available
        "voices": {
            "gtts": {
                "info": "Google Translate TTS (limited control)",
                "languages": list(supported_languages.keys()),
                "voice_options": ["default"],
                "features": ["basic_tts"]
            }
        },
        "languages": supported_languages
    }
    
    # Add pyttsx3 if available
    if have_pyttsx3:
        result["available_engines"].append("pyttsx3")
        
        # Get available voices from pyttsx3
        try:
            engine = pyttsx3.init()
            pyttsx3_voices = []
            
            for voice in engine.getProperty('voices'):
                voice_info = {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": voice.languages,
                    "gender": "unknown"
                }
                
                # Try to detect gender from name
                if "female" in voice.name.lower() or "f" in voice.id.lower():
                    voice_info["gender"] = "female"
                elif "male" in voice.name.lower() or "m" in voice.id.lower():
                    voice_info["gender"] = "male"
                
                pyttsx3_voices.append(voice_info)
            
            result["voices"]["pyttsx3"] = {
                "info": "Local TTS engine with voice control",
                "voices": pyttsx3_voices,
                "features": ["offline_tts", "voice_selection", "rate_control", "volume_control"]
            }
        except Exception as e:
            print(f"Error getting pyttsx3 voices: {str(e)}")
            result["voices"]["pyttsx3"] = {
                "info": "Local TTS engine (error getting voices)",
                "error": str(e),
                "features": ["offline_tts"]
            }
    
    # Add Google Cloud TTS if available
    if have_google_tts:
        result["available_engines"].append("google_cloud")
        
        # Set up standard voices list for Google Cloud
        google_voices = {
            "en": [
                {"name": "en-US-Neural2-A", "gender": "male"},
                {"name": "en-US-Neural2-C", "gender": "female"},
                {"name": "en-US-Neural2-D", "gender": "male"},
                {"name": "en-US-Neural2-F", "gender": "female"}
            ],
            "es": [
                {"name": "es-ES-Neural2-A", "gender": "female"},
                {"name": "es-ES-Neural2-B", "gender": "male"}
            ],
            "fr": [
                {"name": "fr-FR-Neural2-A", "gender": "female"},
                {"name": "fr-FR-Neural2-B", "gender": "male"}
            ],
            "de": [
                {"name": "de-DE-Neural2-A", "gender": "female"},
                {"name": "de-DE-Neural2-B", "gender": "male"}
            ],
            "ja": [
                {"name": "ja-JP-Neural2-B", "gender": "female"},
                {"name": "ja-JP-Neural2-C", "gender": "male"}
            ],
            "ko": [
                {"name": "ko-KR-Neural2-A", "gender": "female"},
                {"name": "ko-KR-Neural2-B", "gender": "male"}
            ],
            "hi": [
                {"name": "hi-IN-Neural2-A", "gender": "female"},
                {"name": "hi-IN-Neural2-B", "gender": "male"}
            ],
            "ta": [
                {"name": "ta-IN-Neural2-A", "gender": "female"},
                {"name": "ta-IN-Neural2-B", "gender": "male"}
            ]
        }
        
        # Filter by language if specified
        if language != "all" and language in google_voices:
            filtered_voices = {language: google_voices[language]}
            google_voices = filtered_voices
        
        result["voices"]["google_cloud"] = {
            "info": "Premium cloud TTS with advanced voice quality",
            "voices": google_voices,
            "features": ["high_quality", "neural_voices", "pitch_control", "speaking_rate", "ssml_support"]
        }
    
    # Information about audio processing capabilities
    if have_audio_processing:
        result["audio_processing"] = {
            "available": True,
            "features": ["pitch_shift", "emotion_effects", "time_stretching", "equalization"]
        }
    else:
        result["audio_processing"] = {
            "available": False
        }
    
    return jsonify(result)

@app.route("/tts_demo", methods=["GET"])
def tts_demo():
    """Render a demo page for testing the advanced TTS capabilities"""
    return render_template(
        "tts_demo.html",
        supported_languages=supported_languages,
        have_pyttsx3=have_pyttsx3,
        have_google_tts=have_google_tts,
        have_audio_processing=have_audio_processing
    )

@app.route("/configure_tts", methods=["POST", "GET"])
def configure_tts():
    """Configure Google Cloud TTS credentials"""
    if request.method == "GET":
        # Check if credentials are already configured
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            return jsonify({"status": "configured", "credentials_file": credentials_file})
        else:
            return jsonify({"status": "not_configured"})
    
    try:
        data = request.get_json() if request.is_json else {}
        
        # If api_key provided directly (simple method)
        if "api_key" in data:
            os.environ["GOOGLE_API_KEY"] = data["api_key"]
            return jsonify({"success": True, "message": "TTS configured with API key"})
        
        # Use the credentials file that we've created
        credentials_file = os.path.join(os.getcwd(), "google_cloud_credentials.json")
        if os.path.exists(credentials_file):
            # Set environment variable to point to the file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
            
            # Test if credentials work
            try:
                if have_google_tts:
                    client = texttospeech.TextToSpeechClient()
                    # Simple test request
                    response = client.list_voices(language_code="en-US")
                    return jsonify({
                        "success": True, 
                        "message": "TTS configured with service account credentials",
                        "voices_available": len(response.voices)
                    })
                else:
                    return jsonify({
                        "success": True, 
                        "warning": "Credentials configured, but google-cloud-texttospeech is not installed"
                    })
            except Exception as e:
                return jsonify({"success": False, "error": f"Credentials file found but test failed: {str(e)}"})
        else:
            return jsonify({"success": False, "error": "Credentials file not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Check if Google Cloud TTS is available
have_google_tts = False
try:
    from google.cloud import texttospeech
    have_google_tts = True
    
    # Credentials should already be set at the top of the file
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print(f"✅ Google Cloud TTS will use credentials from: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    elif os.environ.get("GOOGLE_API_KEY"):
        print(f"✅ Google Cloud TTS will use API key from environment variable")
    else:
        print("⚠️ Google Cloud TTS credentials not found. Using fallback TTS engines.")
        print("  - To configure: POST to /configure_tts with {\"api_key\": \"your_api_key\"}")
        print("  - Or set GOOGLE_API_KEY environment variable")
        print("  - Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
except ImportError:
    print("Google Cloud Text-to-Speech not available. Install with: pip install google-cloud-texttospeech")

# Check if Google Cloud Translate is available
have_google_translate = False
try:
    import google.cloud.translate_v2 as translate
    
    # Test if credentials are working
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY"):
        try:
            client = translate.Client()
            languages = client.get_languages()
            have_google_translate = True
            print(f"✅ Google Cloud Translation API initialized with {len(languages)} supported languages")
        except Exception as e:
            print(f"⚠️ Google Cloud Translation API initialization error: {str(e)}")
            print("  - Translation will fall back to free services")
    else:
        print("⚠️ Google Cloud Translation credentials not found. Using fallback translation services.")
except ImportError:
    print("Google Cloud Translate not available. Install with: pip install google-cloud-translate")

def detect_skin_conditions(image_path, output_path, config=None):
    """
    Detect skin conditions from an image and generate an annotated image
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the annotated image
        config: Configuration options
        
    Returns:
        Dictionary with detected classes and other analysis results
    """
    try:
        # Default configuration
        if config is None:
            config = {
                'confidence': 15,
                'overlap': 30
            }
        
        # Run skin condition detection with roboflow model
        skin_result = model_skin.predict(image_path, 
                                         confidence=config.get('confidence', 15), 
                                         overlap=config.get('overlap', 30)).json()
        
        predictions = skin_result.get("predictions", [])
        if not predictions:
            return {
                'detected_classes': [],
                'message': "No skin conditions detected in the image"
            }
        
        # Extract skin labels
        skin_labels = [pred["class"] for pred in predictions]
        unique_classes = set(skin_labels)
        
        # Run oiliness detection
        try:
            custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
            with CLIENT.use_configuration(custom_configuration):
                oiliness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")
                
            if not oiliness_result.get("predictions"):
                unique_classes.add("dry skin")
            else:
                oiliness_classes = [
                    class_mapping.get(pred["class"], pred["class"])
                    for pred in oiliness_result.get("predictions", [])
                    if pred.get("confidence", 0) >= 0.3
                ]
                unique_classes.update(oiliness_classes)
        except Exception as e:
            print(f"Error in oiliness detection: {str(e)}")
            # Continue with just the skin conditions
        
        # Generate annotated image
        try:
            img_cv = cv2.imread(image_path)
            detections = sv.Detections.from_inference(skin_result)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            
            annotated_image = box_annotator.annotate(scene=img_cv, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            
            cv2.imwrite(output_path, annotated_image)
        except Exception as e:
            print(f"Error creating annotated image: {str(e)}")
            # Continue without annotation
        
        return {
            'detected_classes': list(unique_classes),
            'raw_predictions': predictions,
            'message': "Successfully detected skin conditions"
        }
    except Exception as e:
        print(f"Error in detect_skin_conditions: {str(e)}")
        traceback.print_exc()
        return {
            'detected_classes': [],
            'error': str(e),
            'message': "Failed to detect skin conditions"
        }

if __name__ == "__main__":
    create_tables()  # Create tables if they don't exist
    ensure_database_compatibility()  # Ensure database is compatible with current code
    
    # Check if the credentials file exists but isn't being recognized
    credentials_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_cloud_credentials.json")
    if os.path.exists(credentials_file) and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print(f"Credentials file found at {credentials_file} but not loaded - setting up environment...")
        # Use the credentials file that was found
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    
    # Start the app
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")