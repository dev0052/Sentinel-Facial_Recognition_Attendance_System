
# Sentinel Facial Recognition Attendance System

Sentinel is a robust and efficient facial recognition-based attendance system. It leverages advanced image processing techniques and a graphical user interface (GUI) to automate attendance tracking, ensuring accuracy and ease of use for educational institutions, offices, or any organization requiring a secure and reliable attendance solution.

---

## Features

- **Face Detection and Recognition**: Uses OpenCV and Haarcascade to detect and recognize faces in real time.
- **Automated Attendance Logging**: Saves attendance data to an Excel file with timestamps.
- **Password Protection**: Restricts access to sensitive features through a secure password mechanism.
- **Email Notifications**: Automatically sends the attendance Excel file via email.
- **Graphical User Interface**: Intuitive GUI for managing attendance, training data, and settings.
- **Multi-user Support**: Handles multiple registered users seamlessly.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/dev0052/Sentinel-Facial_Recognition_Attendance_System.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Sentinel-Facial_Recognition_Attendance_System
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the following files are present:
   - `haarcascade_frontalface_default.xml`
   - A valid SMTP email configuration in `main.py`.

5. Run the application:
   ```bash
   python main.py
   ```

---

## Usage

1. **Register Users**: Use the "Take Images" feature to register users by capturing their face data.
2. **Train Model**: Train the facial recognition model using the "Train Images" feature.
3. **Track Attendance**: Start real-time face recognition using the "Track Attendance" feature. Attendance is automatically logged.
4. **Email Reports**: Attendance files are emailed automatically to the configured recipient.

---

## Screenshots



---

## Contributing

Contributions are welcome! If you wish to improve the project:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

If you encounter issues or have suggestions, feel free to reach out:
- **Email**: realdev0052@gmail.com
- **GitHub**: [dev0052](https://github.com/dev0052)

---

## Acknowledgements

- [OpenCV](https://opencv.org/) for image processing.
- The Python community for amazing libraries and support.

---
## Author

- **Dev Sharma** - [GitHub Profile](https://github.com/dev0052)