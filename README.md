# Sentinel Facial Attendance System

## Overview
The Sentinel Facial Attendance System is an innovative, real-time facial recognition-based attendance management system. It eliminates the need for manual attendance tracking by leveraging computer vision and machine learning techniques to automatically mark attendance. The system is designed to enhance accuracy and efficiency, especially in environments like schools, colleges, and corporate offices.

By utilizing OpenCV's pre-trained Haar Cascade Classifier, the system detects faces from live camera feeds and compares them against stored images to identify individuals. It records the attendance of authorized personnel and also identifies intruders by saving their images separately. The system supports adding new students and retraining the model with updated image data.

## Key Features
- **Real-Time Facial Recognition**: Automatically identifies individuals from live video streams.
- **Attendance Marking**: Efficient and accurate logging of attendance records.
- **Intruder Detection**: Captures and stores images of unauthorized persons.
- **Automated Data Logging**: Maintains attendance data in CSV format for easy analysis.
- **Training and Retraining**: Supports adding new individuals and retraining the model on the go.
- **Easy Setup**: Minimal configuration with a virtual environment.
- **Customizability**: Modify the training images and detection settings as needed.

## Technology Stack
- **Programming Language**: Python 3
- **Computer Vision Library**: OpenCV
- **Machine Learning Algorithm**: Haar Cascade Classifier
- **Data Storage**: CSV files for attendance logs and student details
- **Environment Management**: Virtualenv for isolated package management

## Directory Structure
```
Sentinel-Facial-Attendance-System/
├── Attendance/         # Contains attendance logs (CSV files)
├── Intruder/           # Stores images of detected intruders
├── myenv/              # Virtual environment directory
├── StudentDetails/     # Stores student information files
├── TrainingImage/      # Contains images for training the model
├── TrainingImageLabel/ # Stores labeled images for training
├── LICENSE             # License file for the project
├── README.md           # Project documentation
├── haarcascade_frontalface_default.xml # Pre-trained face detection model
├── main.py             # Main script for running the system
├── requirements.txt    # Python dependencies
└── StudentDetails.csv  # CSV file containing student data
```

## Installation Instructions
### Prerequisites
- Python 3.x installed on your system
- Git for cloning the repository
- Virtualenv for creating an isolated environment

### Installation Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Sentinel-Facial-Attendance-System.git
   cd Sentinel-Facial-Attendance-System
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models (if not included)**
   - Place the Haar Cascade XML file in the root directory.

## Usage Instructions
### Adding New Students
1. Place training images in the `TrainingImage/` folder with clear, frontal images of the individual.
2. Use the following naming convention:
   ```
   <StudentID>_<StudentName>.jpg
   ```
3. Update the student details CSV file in the `StudentDetails/` folder.

### Running the System
Execute the main script as follows:
```bash
python main.py
```

### Viewing Attendance Records
Attendance records are stored in the `Attendance/` folder as CSV files. Each file is named with the current date and contains the following columns:
- Student ID
- Name
- Date
- Time

### Intruder Detection
When an unknown face is detected, the system saves the image in the `Intruder/` folder with a timestamped filename.

## Troubleshooting
- **Camera Not Detected**: Ensure that your webcam is properly connected and accessible.
- **No Face Detected**: Adjust the camera angle or lighting conditions.
- **Performance Issues**: Reduce image resolution or use grayscale to speed up processing.

## Contributing
We welcome contributions to improve the project! Feel free to fork the repository, make changes, and submit a pull request. Please follow the project's coding standards and write meaningful commit messages.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements
- OpenCV for providing robust computer vision libraries
- Python for its simplicity and power
- Haar Cascade Classifier for efficient face detection

---
## Author

- **Dev Sharma** - [GitHub Profile](https://github.com/dev0052)

---