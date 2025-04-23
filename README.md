# Knowyourskins 🧬✨

![Project Banner](static/assets/bg5.webp)

A comprehensive skincare analysis and recommendation system combining computer vision with skincare science.

## Features ✨
- 🧑⚕️ AI-powered skin condition analysis using YOLOv8 object detection
- 💄 Personalized product recommendations
- 📸 Image-based skin assessment
- 📹 Live camera skin disease prediction
- 🤖 AI chatbot for skincare advice
- 📅 Appointment booking system
- 👤 User authentication & profile management
- 🌐 Multilingual support with translation features
- 🔊 Text-to-speech capabilities

## Tech Stack 🛠️
- **Backend**: Python Flask (app.py)
- **ML Framework**: TensorFlow/Keras
- **Object Detection**: Ultralytics YOLOv8 (yolov8n.pt)
- **AI Integration**: Google Gemini API, LangChain
- **Computer Vision**: OpenCV, Pillow, Supervision
- **NLP Components**: spaCy, NLTK, TextBlob
- **Database**: SQLite
- **Frontend**: HTML5/CSS3 + Jinja2 templating
- **Text-to-Speech**: Google Cloud TTS, pyttsx3, gTTS
- **Translation**: Google Translate, Deep Translator

## Installation ⚙️

```bash
# Clone repository
git clone https://github.com/yourusername/Knowyourskins.git
cd Knowyourskins

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies (if needed)
pip install inference-sdk==0.36.1
```

## Environment Setup 🔐
Create a `.env` file in the root directory with the following variables:
```
FLASK_SECRET_KEY=your_secret_key
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_google_credentials.json
```

## Usage 🚀

1. Start Flask development server:
```bash
python app.py
```

2. Access web interface at `http://localhost:5000`

3. Key paths:
- `/skin_disease_prediction` - Skin disease analysis (with camera support)
- `/face_analysis` - Skin image analysis
- `/survey` - Skin questionnaire
- `/recommendations` - Product suggestions
- `/chatbot_page` - AI skincare assistant
- `/profile` - User profile and skincare routine

## Using the Camera-based Skin Disease Prediction 📹

Our real-time camera feature allows users to:

1. Access the skin disease prediction page at `/skin_disease_prediction`
2. Click on the "Live Camera" tab
3. Grant camera permissions when prompted
4. View real-time skin disease predictions as you move the camera
5. Capture a specific image for detailed analysis with AI-generated recommendations

## AI Chatbot Assistant 🤖

The integrated chatbot provides:
- Personalized skincare advice
- Product recommendations
- Multilingual support
- Text-to-speech capabilities
- Context-aware responses
- Historical conversation tracking

## Datasets and Models 🔢
- Custom CNN for skin analysis (model/final_model.h5)
- YOLOv8n for lesion detection (yolov8n.pt)
- Product database with 10,000+ skincare products

## Multilingual Support 🌐
The application supports multiple languages including:
- English
- Spanish
- French
- German
- Italian
- Portuguese
- Russian
- Chinese
- Hindi
- Tamil

## Advanced Text-to-Speech 🔊
The system includes multiple TTS options:
- Google Cloud Text-to-Speech
- gTTS (Google Text-to-Speech)
- pyttsx3 (offline TTS)
- Audio processing for emotion and pitch adjustment

## Deployment 🚀

AWS EC2 Deployment Guide:
1. Launch EC2 instance with Ubuntu 22.04 LTS
2. Configure security groups to allow HTTP/HTTPS traffic
3. Connect via SSH and install requirements:
```bash
sudo apt update && sudo apt install python3-pip nginx
pip install -r requirements.txt
```
4. Configure Nginx reverse proxy for Flask app
5. Set up production WSGI server:
```bash
gunicorn -w 4 app:app
```
6. Configure domain in Nginx

Access live at: [https://knowyourskins.info](https://knowyourskins.com)

## Development Status 🧪
This project is currently in active development with continuous improvements to:
- AI model accuracy
- User experience
- Multilingual support
- Mobile responsiveness

## Contributing 🤝
Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.
