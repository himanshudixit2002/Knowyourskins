# Knowyourskins 🧬✨

![Project Banner](static/assets/bg5.webp)

A intelligent skincare analysis and recommendation system combining computer vision with skincare science.

## Features ✨
- 🧑⚕️ AI-powered skin condition analysis using YOLOv8 object detection
- 💄 Personalized product recommendations
- 📸 Image-based skin assessment
- 📹 Live camera skin disease prediction
- 📅 Appointment booking system
- 👤 User authentication & profile management

## Tech Stack 🛠️
- **Backend**: Python Flask (app.py)
- **ML Framework**: TensorFlow/Keras (final_model.h5)
- **Object Detection**: Ultralytics YOLOv8 (yolov8n.pt)
- **Computer Vision**: OpenCV, Pillow for real-time image processing
- **Database**: SQLite (app.db)
- **Frontend**: HTML5/CSS3 + Jinja2 templating

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

## Using the Camera-based Skin Disease Prediction 📹

Our new real-time camera feature allows users to:

1. Access the skin disease prediction page at `/skin_disease_prediction`
2. Click on the "Live Camera" tab
3. Grant camera permissions when prompted
4. View real-time skin disease predictions as you move the camera
5. Capture a specific image for detailed analysis with AI-generated recommendations

The system uses:
- WebRTC for camera access in the browser
- Optimized backend processing for real-time prediction
- TensorFlow model trained on skin disease dataset
- Canvas API for image capturing and processing

## Dataset 🔢
- `dataset/cosmetics.csv`: 10,000+ skincare products with ingredients
- `dataset/updated_skincare_products.csv`: Curated product recommendations
- `model/skin_disease_model.h5`: Trained model for skin disease classification

## Model Architecture 🧠
- Custom CNN for skin analysis (model/final_model.h5)
- YOLOv8n for lesion detection (runs/train32/ weights)
- EfficientNetV2 preprocessing for skin disease classification

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
6. Configure domain `himanshudixit.info` in Nginx
7. Enable automatic restart with systemd service

Access live at: [https://knowyourskins.info](https://knowyourskins.com)

## Camera Access Requirements 🎥

For the camera-based feature to work:
- Use a modern browser (Chrome, Firefox, Edge)
- Allow camera permissions when prompted
- Access the site via HTTPS in production environments (required for camera access)
- Adequate lighting for best prediction accuracy
