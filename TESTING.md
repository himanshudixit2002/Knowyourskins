# Testing the Enhanced Chatbot

This guide will help you test the enhanced chatbot functionality with NLP capabilities.

## Setup

1. Make sure you've installed all required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the setup script to download NLP components:
   ```
   python setup_nlp.py
   ```

3. Ensure your `.env` file contains a valid Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

## Testing Options

### Option 1: Using the Standalone Test Script

The test script allows you to test the chatbot's NLP capabilities and response enhancement without running the full web application:

```
python test_chatbot.py
```

This script will:
1. Parse your input with NLP to extract intent and entities
2. Generate a response using the Gemini API
3. Enhance the response with friendly language, emoji, and other improvements
4. Show you both the raw and enhanced responses for comparison

### Option 2: Testing with the Full Web Application

To test with the full web application:

1. Start the web server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000/chatbot_page
   ```

3. Interact with the chatbot through the web interface

## Test Cases

Here are some test cases to try that demonstrate the enhanced capabilities:

### 1. Product Recommendations
Try: "What products do you recommend for acne?"
- Should identify "acne" as a skin concern
- Should provide specific product recommendations with ingredients
- Should have a friendly, encouraging tone

### 2. Skincare Routine Help
Try: "My skin is very dry, what should I do?"
- Should identify "dry" as a skin concern
- Should provide a routine with multiple steps
- Should include empathetic acknowledgment of the concern

### 3. Ingredient Questions
Try: "Is retinol good for wrinkles?"
- Should identify "retinol" as an ingredient and "wrinkles" as a concern
- Should provide balanced pros and cons about retinol
- Should include safety information

### 4. Comparative Questions
Try: "What's the difference between AHA and BHA?"
- Should identify both as chemical exfoliants
- Should explain the differences clearly
- Should suggest when to use each

### 5. Procedural Questions
Try: "How do I layer skincare products?"
- Should provide a clear step-by-step guide
- Should explain the reasoning behind the order
- Should be conversational rather than clinical

## Evaluating Results

When evaluating the chatbot responses, look for:

1. **Friendliness**: Does the tone feel warm and supportive?
2. **Empathy**: Does it acknowledge the user's concerns before giving advice?
3. **Specificity**: Does it provide specific, actionable recommendations?
4. **Accuracy**: Is the skincare information correct and science-based?
5. **Personalization**: Does it tailor advice to the specific concerns mentioned?

## Troubleshooting

If you encounter issues:

1. Check that all NLP components were successfully installed
2. Verify that your Gemini API key is valid
3. Look at the console output for any error messages
4. Try running `setup_nlp.py` again if components are missing 