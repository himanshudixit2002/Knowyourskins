#!/usr/bin/env python
"""
Test script for the professional chatbot responses.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from app.py
try:
    from app import enhance_response_with_nlp, refine_response
except ImportError:
    print("Error importing from app.py. Make sure you're running this from the project root.")
    sys.exit(1)

def test_professional_response():
    """Test the professional enhancements with sample responses."""
    
    # Example of a conversational/casual response to transform
    casual_response = "Oh honey, dry, flaky skin is so frustrating! It sounds really uncomfortable. The key is gentle hydration and barrier repair. Try switching to a creamy cleanser, like CeraVe Hydrating Facial Cleanser, followed by a rich moisturizer such as La Roche-Posay Toleriane Double Repair Face Moisturizer. Avoid harsh scrubs. Focus on gentle hydration throughout the day. You'll see a difference! Remember, consistency is key – give it a few weeks to see improvement Hope that helps! . You've got this! Further assessment by a dermatologist may be indicated for annoying symptoms."
    
    # User input that triggered the response
    user_input = "I'm struggling with annoying dryness and flaking around my nose and cheeks. It feels tight and uncomfortable, especially in the winter. What can I help?"
    
    # Test negative sentiment
    sentiment = -0.5
    
    # Apply our enhancement functions
    enhanced = enhance_response_with_nlp(casual_response, user_input, sentiment)
    refined = refine_response(enhanced)
    
    # Print results
    print("\nORIGINAL CASUAL RESPONSE:")
    print("="*60)
    print(casual_response)
    print("\nENHANCED PROFESSIONAL RESPONSE:")
    print("="*60)
    print(enhanced)
    print("\nREFINED PROFESSIONAL RESPONSE:")
    print("="*60)
    print(refined)
    print("="*60)

if __name__ == "__main__":
    test_professional_response() 