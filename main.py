import tkinter as tk
from tkinter import messagebox as mess, simpledialog as tsd
import os, cv2, csv
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import time

# ----------------------- Helper Functions ----------------------- #
def assure_path_exists(path):
    """Ensure the provided directory exists, or create it."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_attendance_to_excel(attendance, folder="Attendance"):
    """Save the attendance data to an Excel file."""
    assure_path_exists(folder)
    date = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(folder, f"Attendance_{date}.xlsx")

    df_new = pd.DataFrame(attendance, columns=['Id', 'Name', 'Date', 'Time'])
    if os.path.isfile(file_path):
        df_existing = pd.read_excel(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(file_path, index=False)
    else:
        df_new.to_excel(file_path, index=False)

# ------------------------- Password Functions ------------------------- #
PASSWORD_FILE = "TrainingImageLabel/psd.txt"
EMAIL = "realdev0052@gmail.com"

def save_pass(new_pass):
    assure_path_exists("TrainingImageLabel")
    with open(PASSWORD_FILE, 'w') as f:
        f.write(new_pass)

def load_pass():
    if os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE, 'r') as f:
            return f.read()
    return None

def prompt_password(is_first_time):
    saved_pass = load_pass()

    if is_first_time or not saved_pass:
        new_pass = tsd.askstring('Set Password', 'Set a new password:', show='*')
        if new_pass:
            save_pass(new_pass)
            mess.showinfo('Success', 'Password has been set successfully!')
        else:
            mess.showerror('Error', 'Password not set!')
            return False
    else:
        entered_pass = tsd.askstring('Enter Password', 'Enter your password:', show='*')
        if entered_pass == saved_pass:
            return True
        else:
            forgot_choice = mess.askyesno('Error', 'Incorrect password! Forgot password?')
            if forgot_choice:
                email_entered = tsd.askstring('Forgot Password', f'Enter email ({EMAIL}) to reset password:')
                if email_entered == EMAIL:
                    new_pass = tsd.askstring('Reset Password', 'Set a new password:', show='*')
                    if new_pass:
                        save_pass(new_pass)
                        mess.showinfo('Success', 'Password reset successfully!')
                        return True
                mess.showerror('Error', 'Invalid email!')
    return False

def change_pass():
    saved_pass = load_pass()
    if not saved_pass:
        mess.showerror('Error', 'No password set! Please set a password first.')
        return

    master = tk.Toplevel()
    master.title("Change Password")
    master.geometry("400x200")
    master.configure(bg="white")

    def save_new_pass():
        old_pass = old.get()
        new_pass = new.get()
        confirm_pass = nnew.get()

        if old_pass == saved_pass:
            if new_pass == confirm_pass:
                save_pass(new_pass)
                mess.showinfo('Success', 'Password updated successfully!')
                master.destroy()
            else:
                mess.showerror('Error', 'New passwords do not match!')
        else:
            mess.showerror('Error', 'Incorrect old password!')

    tk.Label(master, text='Old Password:', bg='white').place(x=10, y=20)
    old = tk.Entry(master, show='*', width=30)
    old.place(x=150, y=20)

    tk.Label(master, text='New Password:', bg='white').place(x=10, y=70)
    new = tk.Entry(master, show='*', width=30)
    new.place(x=150, y=70)

    tk.Label(master, text='Confirm Password:', bg='white').place(x=10, y=120)
    nnew = tk.Entry(master, show='*', width=30)
    nnew.place(x=150, y=120)

    tk.Button(master, text="Save", command=save_new_pass).place(x=150, y=160)
    tk.Button(master, text="Cancel", command=master.destroy).place(x=220, y=160)

# -------------------------- Face Recognition -------------------------- #
def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror('Error', 'Haarcascade file missing!')
        exit()

def take_images():
    Id = txt_id.get()
    name = txt_name.get()
    
    if not (Id and name.isalpha()):
        mess.showerror('Error', 'Invalid ID or Name!')
        return

    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    if not Id:
        mess.showerror('Error', 'ID cannot be empty!')
        return
    elif not all(part.isalpha() for part in name.split()):
        mess.showerror('Error', 'Name must only contain alphabets (letters) and spaces!')
        return

    sample_num = 0

    while True:
        ret, img = cam.read()
        if not ret:
            mess.showerror('Error', 'Failed to capture image!')
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"TrainingImage/{name}.{Id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capturing Images', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sample_num > 0:
        with open("StudentDetails/StudentDetails.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([Id, name])

        mess.showinfo('Info', f"Images saved for ID: {Id}")
    else:
        mess.showerror('Error', 'No images were captured!')

def train_images():
    check_haarcascadefile()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces, ids = get_images_and_labels("TrainingImage/")
    recognizer.train(faces, np.array(ids))
    assure_path_exists("TrainingImageLabel/")
    recognizer.save("TrainingImageLabel/Trainner.yml")

    mess.showinfo('Info', 'Training complete!')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, ids = [], []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        faces.append(np.array(img, 'uint8'))
        ids.append(int(os.path.split(image_path)[-1].split('.')[1]))

    return faces, ids

def track_images():
    check_haarcascadefile()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        mess.showerror('Error', 'No training data found! Please train the images first.')
        return

    recognizer.read("TrainingImageLabel/Trainner.yml")

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        mess.showerror('Error', 'Unable to access the camera!')
        return

    attendance = []

    while True:
        ret, img = cam.read()
        if not ret:
            mess.showerror('Error', 'Failed to capture image!')
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 50:
                with open("StudentDetails/StudentDetails.csv", 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if int(row[0]) == id_:
                            name = row[1]
                            break

                date = datetime.now().strftime('%Y-%m-%d')
                time_ = datetime.now().strftime('%H:%M:%S')

                if [id_, name, date, time_] not in attendance:
                    attendance.append([id_, name, date, time_])

                cv2.putText(img, f"{name} - {id_}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cam.release()
                cv2.destroyAllWindows()

                save_attendance_to_excel(attendance)
                mess.showinfo('Info', 'Attendance saved successfully!')
                return
            else:
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Tracking', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if not attendance:
        mess.showinfo('Info', 'No attendance recorded.')

# -------------------------- GUI Code -------------------------- #
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("800x500")
window.configure(bg='#2c3e50')

# Title
header = tk.Label(window, text="Face Recognition Attendance System", bg='#2c3e50', fg="white", font=('Helvetica', 16, 'bold'))
header.pack(pady=20)

# Help Button
def show_help():
    mess.showinfo('Help', f'For assistance, contact: {EMAIL}')

tk.Button(window, text="Help", command=show_help, bg="#34495e", fg="white").place(x=730, y=10)

# Input Frame
input_frame = tk.Frame(window, bg='#34495e')
input_frame.pack(pady=20)

lbl_id = tk.Label(input_frame, text="Enter ID:", bg='#34495e', fg="white")
lbl_id.grid(row=0, column=0, padx=10, pady=10)
txt_id = tk.Entry(input_frame)
txt_id.grid(row=0, column=1, padx=10, pady=10)

lbl_name = tk.Label(input_frame, text="Enter Name:", bg='#34495e', fg="white")
lbl_name.grid(row=1, column=0, padx=10, pady=10)
txt_name = tk.Entry(input_frame)
txt_name.grid(row=1, column=1, padx=10, pady=10)

# Buttons
def track_attendance():
    if prompt_password(is_first_time=False):
        track_images()

btn_take = tk.Button(input_frame, text="Take Images", command=take_images)
btn_take.grid(row=2, column=0, padx=10, pady=10)

btn_train = tk.Button(input_frame, text="Train Images", command=train_images)
btn_train.grid(row=2, column=1, padx=10, pady=10)

btn_track = tk.Button(window, text="Track Attendance", command=track_attendance)
btn_track.pack(pady=10)

btn_change_pass = tk.Button(window, text="Change Password", command=change_pass)
btn_change_pass.place(x=600, y=450)

btn_exit = tk.Button(window, text="Exit", command=window.destroy)
btn_exit.pack(pady=10)

window.mainloop()
