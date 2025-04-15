import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Allow HTTP for local development
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sqlite3
import uuid
import cv2
import pandas as pd
import requests
import traceback
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, session, url_for, jsonify, abort
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
Hey, could you please summarize the text below in a simple, friendly way?
Keep it short—between {min_length} and {max_length} words.
Include a conclusive statement or practical tip at the end.
Make sure your summary is self-contained and COMPLETE, not requiring additional context.

-----------------------------------
{text}
-----------------------------------

Always end with a clear, complete thought. Never leave the response hanging or incomplete.
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
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user = get_user(session["username"])
    if not user:
        return jsonify({"error": "Access denied"}), 403
        
    appointment_id = request.form.get("appointment_id")
    update_appointment_status(appointment_id, 1)  # For example, setting status 1 to confirm
    return jsonify({"message": "Appointment status updated after face analysis."})

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
                "annotated_image": f"/{annotated_image_path}"
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
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
        if not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
            return jsonify({"error": "Invalid file type. Only JPG and PNG are allowed."}), 400
        upload_folder = os.path.join("static", "skin_uploads")
        os.makedirs(upload_folder, exist_ok=True)
        filename = secure_filename(str(uuid.uuid4()) + "_" + image_file.filename)
        file_path = os.path.join(upload_folder, filename)
        image_file.save(file_path)
        print(f"Saved file: {file_path}")
        if best_model is None:
            return jsonify({"error": "Model file not found. Please ensure the model is correctly downloaded and placed in the 'model' directory."}), 500
        try:
            predicted_label, prediction_probs = predict_disease(best_model, file_path)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": f"An error occurred during prediction: {e}"}), 500
        print(f"Prediction: {predicted_label}")
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
    return render_template("skin_disease_prediction.html")

# Redirect old route to new one
@app.route("/skin_predict", methods=["GET", "POST"])
def skin_predict():
    if request.method == "POST":
        return redirect(url_for("skin_disease_prediction"), code=307)  # 307 preserves POST data
    return redirect(url_for("skin_disease_prediction"))

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

def build_conversation_prompt(history, user_input):
    """Builds a conversation prompt for the chatbot, including conversation history and user input."""
    prompt = """You are KnowYourSkins AI Assistant, an expert skincare advisor. Follow these guidelines:

1. ACCURACY: Provide accurate skincare information based on dermatological science. Never invent facts.
2. CONCISENESS: Keep answers brief, focused, and to the point (1-3 sentences when possible). No fluff or unnecessary explanations.
3. VALUE: Prioritize actionable advice and practical recommendations over general information.
4. TONE: Professional but conversational and empathetic. Acknowledge skin concerns with understanding.
5. CAUTION: For skin concerns that may require medical attention, recommend professional consultation with a dermatologist.
6. PERSONALIZATION: Use context from previous messages to tailor your advice.

Your strengths include general skincare advice, recommending routines, explaining ingredients, and suggesting daily skincare habits.

LIMITATIONS: Do NOT attempt to diagnose skin conditions, analyze skin images, or provide medical treatment recommendations. Always refer users to consult with a dermatologist for diagnosis of skin conditions.

If unsure, acknowledge limitations rather than guessing. Skin health is important - accuracy matters.

"""
    
    # Add conversation history
    if history:
        prompt += "Previous conversation:\n"
        for msg in history:
            if msg['role'] == 'user':
                prompt += f"User: {msg['text']}\n"
            else:
                prompt += f"Assistant: {msg['text']}\n"
    
    # Add current user input
    prompt += f"\nCurrent question: {user_input}\n\nYour concise, accurate response:"
    
    return prompt

def refine_response(response):
    """Post-processes the chatbot response to ensure it's concise, valuable and well-structured."""
    if not response:
        return "I'm not sure about that. Please consider consulting with a dermatologist for personalized advice."
    
    # Trim any leading/trailing whitespace or common AI-generated fillers
    response = response.strip()
    
    # Remove common verbose AI introductions
    filler_phrases = [
        "As an AI assistant, I",
        "As a skincare assistant, I",
        "Based on the information provided,",
        "I'd be happy to help with that.",
        "Thank you for your question.",
        "That's a great question.",
        "I'd like to provide some information about",
        "I can certainly help with that.",
        "Let me answer that for you.",
    ]
    
    for phrase in filler_phrases:
        if response.startswith(phrase):
            response = response[len(phrase):].strip()
            # Remove additional connecting words if they exist
            for connector in [", ", ". ", "! "]:
                if response.startswith(connector):
                    response = response[len(connector):].strip()
    
    # Capitalize first letter if needed
    if response and not response[0].isupper():
        response = response[0].upper() + response[1:]
    
    # If response is too long (over 500 chars), try to summarize with the Gemini API
    if len(response) > 500:
        try:
            headers = {"Content-Type": "application/json"}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            
            summarize_prompt = f"Summarize this skincare advice in 2-3 concise, valuable sentences, keeping all important recommendations and warnings: {response}"
            
            summary_response = requests.post(
                url, 
                headers=headers, 
                json={"contents": [{"parts": [{"text": summarize_prompt}]}]}
            )
            
            if summary_response.status_code == 200:
                data = summary_response.json()
                summary = (data.get("candidates", [{}])[0]
                           .get("content", {})
                           .get("parts", [{}])[0]
                           .get("text", ""))
                
                if summary and len(summary) < len(response):
                    return summary.strip()
        except Exception:
            # If summarization fails, return the original but slightly truncated
            pass
            
    return response

def complete_answer_if_incomplete(answer):
    """Checks if the chatbot's answer is complete and continues it if necessary."""
    # Check if answer ends with proper punctuation
    if not answer.rstrip().endswith(('.', '!', '?', ':', ';', '"', ')', ']', '}')):
        # Answer appears incomplete, so continue it
        continuation_prompt = f"Continue the following answer: {answer}"
        
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
    
    return answer

def get_conversation_history(user_id=None):
    """Gets the conversation history for a user or session."""
    with get_db_connection() as conn:
        if user_id:
            history = conn.execute(
                "SELECT role, text FROM conversations WHERE user_id = ? ORDER BY timestamp", 
                (user_id,)
            ).fetchall()
        else:
            # For non-logged-in users, use session ID
            session_id = session.get('session_id')
            if not session_id:
                return []
                
            history = conn.execute(
                "SELECT role, text FROM conversations WHERE user_id = ? ORDER BY timestamp", 
                (session_id,)
            ).fetchall()
        
        return [dict(h) for h in history]

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

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json()
        user_input = data.get("userInput")
        
        if not user_input:
            return jsonify({"error": "No user input provided."}), 400
        
        # Get user ID for persistent storage
        user_id = None
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                user_id = user["id"]
        
        # Initialize session variables if they don't exist
        if "conversation_state" not in session:
            session["conversation_state"] = {}
        conversation_state = session["conversation_state"]
        
        # Load conversation history from database if available, otherwise use session
        conversation_history = []
        if user_id:
            conversation_history = get_conversation_history(user_id)
        elif "conversation_history" in session:
            conversation_history = session["conversation_history"]
        else:
            session["conversation_history"] = []
            conversation_history = []
        
        # Handle special case: clear chat request
        if user_input.lower() in ["clear chat", "start new chat", "reset chat"]:
            if user_id:
                clear_conversation_history(user_id)
            session["conversation_history"] = []
            session["conversation_state"] = {}
            session.modified = True
            return jsonify({
                "botReply": "I've cleared our conversation history. How can I help you today?",
                "type": "clear_confirmation"
            })
        
        # If user asks about skin analysis or diagnosis, redirect them
        if any(term in user_input.lower() for term in ["analyze my skin", "diagnose my skin", "skin diagnosis", "analyze photo", "check my face", "analyze my face", "skin disease", "what disease", "what condition"]):
            # Save the user message
            conversation_history.append({"role": "user", "text": user_input})
            if user_id:
                save_conversation_message("user", user_input, user_id)
            
            # Generate and save the bot response
            bot_reply = "I can't analyze skin conditions or diagnose skin diseases through chat. Please use our dedicated Skin Analysis tool or Skin Disease Prediction tool from the main menu, or consult with a dermatologist for a professional diagnosis."
            conversation_history.append({"role": "assistant", "text": bot_reply})
            if user_id:
                save_conversation_message("assistant", bot_reply, user_id)
            
            # Update session
            session["conversation_history"] = conversation_history
            
            return jsonify({
                "botReply": bot_reply,
                "type": "general_response"
            })
        
        # Handle appointment flow states
        if conversation_state.get("awaiting_date"):
            try:
                parsed_date = dateparser.parse(user_input)
                if parsed_date:
                    conversation_state["date"] = parsed_date.strftime("%Y-%m-%d %H:%M")
                    conversation_state["awaiting_date"] = False
                    conversation_state["awaiting_reason"] = True
                    session.modified = True
                    
                    # Save the user message
                    conversation_history.append({"role": "user", "text": user_input})
                    if user_id:
                        save_conversation_message("user", user_input, user_id)
                    
                    # Generate and save the bot response
                    bot_reply = f"Great! Your appointment is set for {conversation_state['date']}. Now, please describe the reason for your appointment."
                    conversation_history.append({"role": "assistant", "text": bot_reply})
                    if user_id:
                        save_conversation_message("assistant", bot_reply, user_id)
                    
                    # Update session if needed
                    session["conversation_history"] = conversation_history
                    
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
                
        if conversation_state.get("awaiting_reason"):
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
                    date=conversation_state["date"],
                    skin=survey_data["skin_type"],
                    phone=survey_data.get("phone", ""),
                    age=survey_data["age"],
                    address=reason,
                    status=False,
                    username=user["username"]
                )
                
                # Save the user message
                conversation_history.append({"role": "user", "text": user_input})
                if user_id:
                    save_conversation_message("user", user_input, user_id)
                
                # Generate and save the bot response
                bot_reply = f"Your appointment has been successfully scheduled for {conversation_state['date']} with the reason: {reason}. Your reference ID is APPT-{appointment_id}."
                conversation_history.append({"role": "assistant", "text": bot_reply})
                if user_id:
                    save_conversation_message("assistant", bot_reply, user_id)
                
                # Reset conversation state
                session["conversation_history"] = conversation_history
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
            conversation_state["awaiting_date"] = True
            session.modified = True
            
            # Save the user message
            conversation_history.append({"role": "user", "text": user_input})
            if user_id:
                save_conversation_message("user", user_input, user_id)
            
            # Generate and save the bot response
            bot_reply = "When would you like to schedule your appointment? (e.g., 'March 10 at 3 PM')"
            conversation_history.append({"role": "assistant", "text": bot_reply})
            if user_id:
                save_conversation_message("assistant", bot_reply, user_id)
            
            # Update session
            session["conversation_history"] = conversation_history
            
            return jsonify({
                "botReply": bot_reply,
                "type": "appointment_flow"
            })
        
        # Handle general conversation
        # Save the user message first
        conversation_history.append({"role": "user", "text": user_input})
        if user_id:
            save_conversation_message("user", user_input, user_id)
        
        # Generate response
        prompt = build_conversation_prompt(conversation_history, user_input)
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
                except Exception as process_error:
                    print(f"Response processing error: {str(process_error)}")
                    # If post-processing fails, still return the original response
                    pass
            else:
                bot_reply = "I apologize, but I couldn't generate a response. Could you please rephrase your question?"
            
            # Save the bot response
            conversation_history.append({"role": "assistant", "text": bot_reply})
            if user_id:
                save_conversation_message("assistant", bot_reply, user_id)
            
            # Update session
            session["conversation_history"] = conversation_history
            
            return jsonify({"botReply": bot_reply, "type": "general_response"})
        except Exception as gemini_error:
            print(f"Generative AI error: {str(gemini_error)}")
            error_message = "I'm having trouble connecting to my knowledge base right now. Could you try again in a moment?"
            return jsonify({"botReply": error_message, "type": "error"})
    
    except Exception as e:
        import traceback
        print(f"Chatbot route error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"botReply": "I encountered an error while processing your message. Please try again.", "type": "error"}), 500

@app.route("/clear_conversation", methods=["POST"])
def clear_conversation():
    try:
        # Clear conversation in database if user is logged in
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                clear_conversation_history(user["id"])
        
        # Clear conversation in session
        session["conversation_history"] = []
        session["conversation_state"] = {}
        session.modified = True
        
        return jsonify({"success": True})
    except Exception as e:
        print("Error clearing conversation:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/get_conversation_history", methods=["GET"])
def get_conversation_history_route():
    try:
        # Get conversation from database if user is logged in
        if "username" in session:
            user = get_user(session.get("username"))
            if user:
                history = get_conversation_history(user["id"])
                return jsonify(history)
        
        # Otherwise return from session
        if "conversation_history" in session:
            return jsonify(session["conversation_history"])
        
        return jsonify([])
    except Exception as e:
        print("Error fetching conversation history:", str(e))
        return jsonify([]), 500

@app.route("/chatbot_page")
def chatbot_page():
    return render_template("chatbot.html")

if __name__ == "__main__":
    create_tables()  # Create tables if they don't exist
    ensure_database_compatibility()  # Add any missing columns
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")