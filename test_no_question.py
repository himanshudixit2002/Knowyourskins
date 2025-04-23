#!/usr/bin/env python
"""
Test script to verify the chatbot doesn't include the user's question in responses.
"""
import os
import sys
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

def test_chatbot_response():
    """Test that the chatbot doesn't prepend the user's question in responses."""
    
    # Test question
    test_question = "I'm struggling with persistent dryness and flakiness on my cheeks, especially during winter. My usual moisturizer isn't cutting it anymore."
    
    # Simulate a POST request to the chatbot endpoint
    try:
        response = requests.post(
            "http://localhost:5000/chatbot",  # Adjust URL as needed
            headers={"Content-Type": "application/json"},
            json={"message": test_question, "language": "en"}
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("botReply", "")
            
            print("\nUSER QUESTION:")
            print("="*60)
            print(test_question)
            print("\nBOT RESPONSE:")
            print("="*60)
            print(bot_reply)
            
            # Check if the question is included in the response
            if test_question in bot_reply or "User's question:" in bot_reply:
                print("\nWARNING: The user's question appears to be included in the response!")
            else:
                print("\nSUCCESS: The response does not include the user's question.")
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_chatbot_response() 