#!/usr/bin/env python
"""
Simple test script for the enhanced chatbot.
This allows testing the NLP-enhanced responses without running the full web application.
"""
import os
import sys
import json
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables")
    print("Please run setup_nlp.py first to configure your API key")
    print("Or manually add your Gemini API key to the .env file")
    sys.exit(1)

# Import from app.py (add parent directory to path if needed)
try:
    from app import analyze_user_intent, enhance_response_with_nlp, extract_skincare_entities, have_textblob, nlp
except ImportError:
    print("Error importing from app.py. Make sure you're running this from the project root.")
    sys.exit(1)

def generate_gemini_response(prompt):
    """Generate a response using Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        print("Sending request to Gemini API...")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        print(f"Response received in {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            text = (data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", ""))
            return text
        else:
            print(f"Error {response.status_code}: {response.text}")
            if response.status_code == 403:
                print("\nThis might be due to one of these reasons:")
                print("1. The API key is invalid")
                print("2. The API key doesn't have permission to access the Gemini API")
                print("3. The API key hasn't been set up with billing")
                print("\nPlease visit https://ai.google.dev/ to check your API key.")
            return f"Error {response.status_code}: API request failed. Check console for details."
    except Exception as e:
        print(f"Exception while calling Gemini API: {str(e)}")
        return f"Exception: {str(e)}"

def test_chatbot(user_input):
    """Test the chatbot with a user input."""
    print(f"\n{'='*60}")
    print(f"USER: {user_input}")
    print(f"{'='*60}")
    
    # Step 1: Analyze the intent and extract entities
    intent_analysis = analyze_user_intent(user_input)
    intent = intent_analysis["intent"]
    entities = intent_analysis["entities"]
    
    # Step 2: Extract detailed skincare entities
    skincare_entities = extract_skincare_entities(user_input)
    
    # Print analysis results
    print(f"\nINTENT: {intent}")
    print(f"ENTITIES: {json.dumps(entities, indent=2)}")
    print(f"SKINCARE ENTITIES: {json.dumps(skincare_entities, indent=2)}")
    print(f"SENTIMENT: {intent_analysis['sentiment']}")
    
    # Step 3: Create a prompt for Gemini
    prompt = f"""
You are Aura, a friendly and empathetic skincare assistant. 
Please respond to this skincare question in a helpful, warm manner:

{user_input}

Context from analysis:
- Intent: {intent}
- Entities: {json.dumps(entities)}
- Skin concerns: {', '.join(skincare_entities['skin_concerns']) if skincare_entities['skin_concerns'] else 'None mentioned'}
- Ingredients: {', '.join(skincare_entities['ingredients']) if skincare_entities['ingredients'] else 'None mentioned'}
- Product types: {', '.join(skincare_entities['product_types']) if skincare_entities['product_types'] else 'None mentioned'}

Your response should be friendly, empathetic, and provide specific, actionable advice.
End with a gentle encouragement.
"""
    
    # Step 4: Get a response from Gemini
    print("\nGenerating response...")
    raw_response = generate_gemini_response(prompt)
    
    # If we got an error response, skip enhancement
    if raw_response.startswith(("Error", "Exception")):
        print(f"\nERROR RESPONSE:\n{'-'*60}\n{raw_response}\n{'-'*60}")
        return raw_response
    
    # Step 5: Enhance the response with NLP
    try:
        enhanced_response = enhance_response_with_nlp(raw_response, user_input, intent_analysis["sentiment"])
        
        # Print responses
        print(f"\nRAW RESPONSE:\n{'-'*60}\n{raw_response}\n{'-'*60}")
        print(f"\nENHANCED RESPONSE:\n{'-'*60}\n{enhanced_response}\n{'-'*60}")
        
        return enhanced_response
    except Exception as e:
        print(f"Error enhancing response: {e}")
        return raw_response

def main():
    """Run test cases or take user input."""
    print("Skincare Chatbot Test Tool")
    print("Enter 'q' to quit")
    
    # Check if we have all the NLP components
    nlp_status = []
    if not have_textblob:
        nlp_status.append("TextBlob missing - sentiment analysis limited")
    if nlp is None:
        nlp_status.append("spaCy missing - entity extraction limited")
    
    if nlp_status:
        print("\nWarning: Some NLP components are not available:")
        for status in nlp_status:
            print(f"- {status}")
        print("Run setup_nlp.py to install missing components.")
    
    # List of test questions
    test_questions = [
        "What products do you recommend for acne?",
        "My skin is very dry, what should I do?",
        "Is retinol good for wrinkles?",
        "What's the difference between AHA and BHA?",
        "How do I layer skincare products?",
    ]
    
    print("\nSample questions:")
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. {question}")
    
    while True:
        choice = input("\nEnter question number or type your own (q to quit): ")
        
        if choice.lower() == 'q':
            break
        
        try:
            # Check if it's a number for a test question
            if choice.isdigit() and 1 <= int(choice) <= len(test_questions):
                test_chatbot(test_questions[int(choice) - 1])
            else:
                # User typed their own question
                test_chatbot(choice)
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"Traceback: {sys.exc_info()}")

if __name__ == "__main__":
    main() 