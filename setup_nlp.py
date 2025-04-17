#!/usr/bin/env python
"""
Setup script for NLP components used by the chatbot.
Run this script before starting the application to ensure all necessary NLP resources are available.
"""
import os
import sys
import nltk
import subprocess
import requests
import importlib.util
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_package_installed(package_name):
    """Check if a Python package is installed."""
    return importlib.util.find_spec(package_name) is not None

def setup_nltk():
    """Download required NLTK resources."""
    print("Setting up NLTK resources...")
    try:
        # Check if NLTK is installed
        if not check_package_installed("nltk"):
            print("× NLTK is not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "nltk"], check=True)
            print("✓ NLTK installed successfully")
        
        # Download required NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✓ NLTK resources downloaded successfully")
    except Exception as e:
        print(f"× Error setting up NLTK: {str(e)}")
        print("  You can manually download NLTK resources with:")
        print("  >>> import nltk")
        print("  >>> nltk.download('punkt')")
        print("  >>> nltk.download('wordnet')")
        print("  >>> nltk.download('vader_lexicon')")
        return False
    return True

def setup_spacy():
    """Install spaCy models."""
    print("Setting up spaCy...")
    try:
        # Check if spaCy is installed
        if not check_package_installed("spacy"):
            print("× spaCy is not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)
            print("✓ spaCy installed successfully")
        
        # Install the English model if not already installed
        try:
            # Try to import the model to see if it's installed
            import en_core_web_sm
            print("✓ spaCy model 'en_core_web_sm' already installed")
        except ImportError:
            print("Installing spaCy model 'en_core_web_sm'...")
            try:
                # Try using spacy download command
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                print("✓ spaCy model installed successfully")
            except Exception as e:
                print(f"× Error installing spaCy model: {str(e)}")
                print("Trying alternative installation method...")
                # Try using pip if spacy download command fails
                subprocess.run([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl"], check=True)
                print("✓ spaCy model installed successfully via pip")
    except Exception as e:
        print(f"× Error setting up spaCy: {str(e)}")
        print("You can manually install the required model with: python -m spacy download en_core_web_sm")
        return False
    return True

def setup_textblob():
    """Set up TextBlob."""
    print("Setting up TextBlob...")
    try:
        # Check if TextBlob is installed
        if not check_package_installed("textblob"):
            print("× TextBlob is not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "textblob"], check=True)
            print("✓ TextBlob installed successfully")
        
        # Download corpora
        try:
            import textblob
            textblob.download_corpora(quiet=True)
            print("✓ TextBlob corpora downloaded successfully")
        except Exception as e:
            print(f"× Error downloading TextBlob corpora: {str(e)}")
            print("  You can manually download TextBlob corpora with:")
            print("  >>> python -m textblob.download_corpora")
            # Continue anyway as this is not critical
    except Exception as e:
        print(f"× Error setting up TextBlob: {str(e)}")
        return False
    return True

def verify_and_fix_gemini_api():
    """Verify the Gemini API key is set and working, with interactive fixing if needed."""
    print("Checking for Gemini API key...")
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("× Gemini API key not found in environment variables")
        print("\nPlease ensure you have a valid Gemini API key in your .env file:")
        print("1. Create or edit the .env file in the project root")
        print("2. Add the line: GEMINI_API_KEY=your_key_here")
        print("3. Save the file and run this script again")
        
        # Offer to create/update the .env file
        try:
            choice = input("\nWould you like to enter your Gemini API key now? (y/n): ")
            if choice.lower() == 'y':
                key = input("Enter your Gemini API key: ").strip()
                # Check if .env file exists
                env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
                
                if os.path.exists(env_path):
                    # Read existing content
                    with open(env_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Replace or add the API key
                    key_found = False
                    for i, line in enumerate(lines):
                        if line.startswith('GEMINI_API_KEY='):
                            lines[i] = f'GEMINI_API_KEY={key}\n'
                            key_found = True
                            break
                    
                    if not key_found:
                        lines.append(f'\nGEMINI_API_KEY={key}\n')
                    
                    # Write back
                    with open(env_path, 'w') as f:
                        f.writelines(lines)
                else:
                    # Create new .env file
                    with open(env_path, 'w') as f:
                        f.write(f'GEMINI_API_KEY={key}\n')
                
                print("✓ API key saved to .env file")
                print("Reloading environment variables...")
                load_dotenv(override=True)
                api_key = os.environ.get("GEMINI_API_KEY")
                print("✓ Environment variables reloaded")
            else:
                return False
        except Exception as e:
            print(f"× Error updating .env file: {str(e)}")
            return False
    
    if not api_key:
        print("× Still couldn't load the Gemini API key after update attempt")
        return False
        
    print("✓ Gemini API key found. Testing connection...")
    
    try:
        # Test the API with a minimal request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": "Hello world"}]}]}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✓ Gemini API connection successful!")
            return True
        else:
            print(f"× Gemini API returned error: {response.status_code}")
            print(f"  Response: {response.text}")
            
            if response.status_code == 403:
                print("\nThis might be due to one of these reasons:")
                print("1. The API key is invalid")
                print("2. The API key doesn't have permission to access the Gemini API")
                print("3. The API key hasn't been set up with billing")
                print("\nPlease visit https://ai.google.dev/ to check your API key.")
            
            return False
    except Exception as e:
        print(f"× Error testing Gemini API: {str(e)}")
        return False

def setup_emoji():
    """Set up emoji package."""
    print("Setting up emoji package...")
    try:
        # Check if emoji is installed
        if not check_package_installed("emoji"):
            print("× emoji package is not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "emoji"], check=True)
            print("✓ emoji package installed successfully")
        else:
            print("✓ emoji package already installed")
    except Exception as e:
        print(f"× Error setting up emoji package: {str(e)}")
        return False
    return True

def main():
    """Run all setup steps."""
    print("\n==================================================")
    print("     Setting up NLP components for the chatbot     ")
    print("==================================================\n")
    
    # Set up all components
    results = {
        "NLTK": setup_nltk(),
        "spaCy": setup_spacy(),
        "TextBlob": setup_textblob(),
        "emoji": setup_emoji(),
        "Gemini API": verify_and_fix_gemini_api()
    }
    
    # Print summary
    print("\n==================================================")
    print("                    Summary                       ")
    print("==================================================")
    for component, success in results.items():
        status = "✓ Installed" if success else "× Failed"
        print(f"{component}: {status}")
    
    # Check if all critical components are installed
    critical_components = ["NLTK", "TextBlob", "Gemini API"]
    critical_success = all(results[component] for component in critical_components)
    
    if critical_success:
        print("\n✓ Critical components installed successfully!")
        if not results["spaCy"] or not results["emoji"]:
            print("  Some non-critical components had issues, but the chatbot can still function.")
            print("  Certain enhanced features may be limited.")
    else:
        print("\n× Setup completed with errors in critical components.")
        print("  The chatbot may not function correctly without these components.")
        print("  Please fix the issues before starting the application.")
    
    print("\nTo run the application: python app.py")

if __name__ == "__main__":
    main() 