# Facial-Recognition_Attendance-System


## Description
The **Facial-Recognition_Attendance-System** is a robust application for managing attendance using face recognition. It leverages machine learning, computer vision, and cloud storage to ensure secure, efficient, and real-time attendance tracking. This system is ideal for educational institutions, offices, or any organization that requires seamless attendance monitoring.

---

## Features
- **Face Recognition**: Uses advanced facial recognition to mark attendance.
- **Cloud-Based Storage**: Stores attendance records securely in the cloud for accessibility and reliability.
- **Admin Panel**: Features password-protected admin access for managing data.
- **Real-Time Monitoring**: Tracks attendance in real-time with a user-friendly GUI.
- **Excel Export**: Automatically saves attendance data in Excel format for easy reporting.
- **User Count Display**: Shows the total number of registered users in the system.



---

## Installation

### Prerequisites
- Python 3.8+
- A webcam or external camera
- Internet connection for cloud integration (if applicable)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/dev0052/Facial-Recognition_Attendance-System.git
   cd Facial-Recognition_Attendance-System
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # For Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```

---

## Usage
1. **Take Images**: Register a new user by entering an ID and name, then capture their images.
2. **Train Images**: Train the model with captured images.
3. **Track Attendance**: Start the face recognition system to mark attendance.
4. **Admin Options**:
   - Change password.
   - Monitor attendance records.

---

## File Structure
```
Facial-Recognition_Attendance-System/
├── main.py                 # Main application script
├── requirements.txt        # Project dependencies
├── TrainingImage/          # Captured images for training
├── TrainingImageLabel/     # Trained model data
├── StudentDetails/         # Registered user details (CSV)
├── Attendance/             # Attendance records (Excel)
└── README.md               # Project documentation
```

---

## Technologies Used
- **Python**: Core programming language
- **OpenCV**: For face recognition
- **Tkinter**: GUI framework
- **Pandas**: Data manipulation and storage
- **Pillow**: Image processing

---

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
For any inquiries or assistance, please contact the project owner:
- **GitHub**: [dev0052](https://github.com/dev0052)
- **Email**: realdev0052@gmail.com

---


Happy Coding! 🚀
