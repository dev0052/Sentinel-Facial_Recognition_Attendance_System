# Sentinel Facial Recognition Attendance System

## Overview
The **Sentinel Facial Attendance System** is a real-time facial recognition-based attendance management system. It eliminates the need for manual attendance tracking by leveraging **computer vision** and **machine learning** techniques to automatically mark attendance. The system is designed for **schools, colleges, and corporate offices** to enhance accuracy and efficiency.

This system uses **OpenCV's pre-trained Haar Cascade Classifier** to detect faces from a live camera feed and compare them against stored images to identify individuals. It records attendance and captures images of unauthorized persons separately. The system allows **adding new students and retraining the model** dynamically.

---
## Key Features
- **Real-Time Facial Recognition**: Identifies individuals from live video streams.
- **Attendance Marking**: Logs attendance efficiently and accurately.
- **Intruder Detection**: Captures and stores images of unknown persons and sends email alerts.
- **Automated Data Logging**: Stores attendance data in CSV format for easy analysis.
- **Training and Retraining**: Supports adding new individuals and updating the model.
- **Easy Setup**: Minimal configuration required.
- **Customizability**: Modify training images and detection settings as needed.

---
## Technology Stack
- **Programming Language**: Python 3
- **Computer Vision Library**: OpenCV
- **Machine Learning Algorithm**: Haar Cascade Classifier
- **Data Storage**: CSV files for attendance logs
- **Email Alerts**: SMTP for sending intruder detection alerts
- **Environment Management**: Virtual Environment (venv)

---
## Understanding Haar Cascade Classifier
The **Haar Cascade Classifier** is a machine learning object detection algorithm used to detect objects in images or video. It works by analyzing patterns in an image and comparing them to pre-trained patterns stored in an XML file.

### How It Works
1. **Feature Extraction**: Haar-like features (patterns) are extracted from an image.
2. **Integral Image Calculation**: The algorithm quickly calculates feature values.
3. **Adaboost Training**: Weak classifiers are combined into a strong classifier.
4. **Cascade Structure**: The classifier works in multiple stages, rejecting non-face regions early to optimize speed.

---
## Comparison with Alternative Methods

| Method | Accuracy | Speed | Complexity | Requires Training? |
|--------|---------|-------|------------|------------------|
| **Haar Cascade (Used in Sentinel)** | High | Fast | Low | No (Pre-trained Model) |
| **HOG + SVM** | Medium | Medium | Medium | Yes |
| **Deep Learning (CNNs, YOLO, etc.)** | Very High | Slow | High | Yes |

### Why Haar Cascade?
- **Fast**: Uses a cascade approach for quick processing.
- **Lightweight**: Requires minimal computational power.
- **Pre-trained**: Works out of the box with minimal setup.

---
## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Sentinel-Facial-Attendance-System.git
cd Sentinel-Facial-Attendance-System
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Face Detection Model
Ensure that the **Haar Cascade XML** file (`haarcascade_frontalface_default.xml`) is in the project root directory.
If not, download it from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the project directory.

---
## Usage Guide

### 1. Adding New Students
- Place training images in the `TrainingImage/` folder.
- Use the following naming convention for images:
  ```
  <StudentID>_<StudentName>.jpg
  ```
- Update `StudentDetails.csv` in the `StudentDetails/` folder.

### 2. Running the Attendance System
```bash
python main.py
```

### 3. Viewing Attendance Records
- Attendance records are stored in the `Attendance/` folder as **CSV files**.
- Each file is named with the **current date** and contains the following details:
  - Student ID
  - Name
  - Date
  - Time

### 4. Intruder Detection and Email Alerts
When an unknown face is detected:
1. The system captures the image and saves it in the `Intruder/` folder.
2. An **email alert** is sent to the administrator with the detected image attached.

#### Setting Up Email Alerts
- Configure the SMTP server in the `config.py` file:
  ```python
  SMTP_SERVER = "smtp.gmail.com"
  SMTP_PORT = 587
  EMAIL_SENDER = "your_email@gmail.com"
  EMAIL_PASSWORD = "your_password"
  EMAIL_RECEIVER = "admin_email@gmail.com"
  ```
- Ensure that **less secure apps access** is enabled on your email account.

---
## Troubleshooting
### 1. `haarcascade_frontalface_default.xml` Not Found
Ensure the file is in the project root directory. If missing, download and place it manually.

### 2. `cv2.CascadeClassifier` AttributeError
If OpenCV is not installed correctly, reinstall it:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

### 3. Camera Not Detected
Check if your webcam is properly connected and accessible.

### 4. No Face Detected
- Adjust **camera angle** and **lighting conditions**.
- Ensure the face is **clearly visible** to the camera.

### 5. Email Alerts Not Working
- Verify SMTP server settings.
- Check if your email account allows **less secure apps**.
- Ensure the internet connection is stable.

---
## Contributing
We welcome contributions! Fork the repository, make your changes, and submit a pull request.

---
## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---
## Acknowledgements
- **OpenCV** for providing robust computer vision tools.
- **Python** for its simplicity and power.
- **Haar Cascade Classifier** for efficient face detection.

---
## Author
- **Dev Sharma** - [GitHub Profile](https://github.com/dev0052)

