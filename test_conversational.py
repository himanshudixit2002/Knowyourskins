#!/usr/bin/env python
"""
Simple test script for the enhanced conversational chatbot responses.
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

def test_conversational_response():
    """Test the conversational enhancements with sample responses."""
    
    # Example of a formal/clinical response
    clinical_response = "Dry, flaky skin is typically caused by a compromised skin barrier. It is recommended to incorporate a hydrating serum with hyaluronic acid, followed by a moisturizer with ceramides. Gentle cleansing is essential to avoid further irritation. CeraVe Moisturizing Cream is an effective option due to its ceramide content. Application of this regimen both morning and night should yield optimal results within approximately two weeks. Further evaluation by a dermatologist is recommended if symptoms persist or worsen."
    
    # User input that triggered the response
    user_input = "My skin is so dry and flaky lately, especially around my cheeks. It feels tight and uncomfortable. What can I help?"
    
    # Test negative sentiment
    sentiment = -0.5
    
    # Apply our enhancement functions
    enhanced = enhance_response_with_nlp(clinical_response, user_input, sentiment)
    refined = refine_response(enhanced)
    
    # Print results
    print("\nORIGINAL CLINICAL RESPONSE:")
    print("="*60)
    print(clinical_response)
    print("\nENHANCED CONVERSATIONAL RESPONSE:")
    print("="*60)
    print(enhanced)
    print("\nREFINED CONVERSATIONAL RESPONSE:")
    print("="*60)
    print(refined)
    print("="*60)

if __name__ == "__main__":
    test_conversational_response() 