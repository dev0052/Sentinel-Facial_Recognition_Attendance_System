import tkinter as tk
from tkinter import messagebox as mess, simpledialog as tsd
from tkinter import ttk
import smtplib
from email.message import EmailMessage
import mimetypes
import os, cv2, csv
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "ubuntuvm22.alerts@gmail.com"
EMAIL_PASSWORD = "aemp hjhj ewsu geca"
EMAIL_RECEIVER = "dev.sharma2021@vitbhopal.ac.in"


def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def send_mail_with_image(image_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Intruder Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("An intruder has been detected. Please check the attached image.")

        with open(image_path, "rb") as img:
            img_data = img.read()
            mime_type, _ = mimetypes.guess_type(image_path)
            main_type, sub_type = mime_type.split("/")
            msg.add_attachment(img_data, maintype=main_type, subtype=sub_type, filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent successfully with image: {image_path}")
    except Exception as e:
        print(f"Error sending email: {e}")


def save_attendance_to_excel(attendance, folder="Attendance"):
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

    send_attendance_email(file_path)

recording_active = True


def stop_recording(cam, record_window):
    global recording_active
    recording_active = False
    cam.release()
    cv2.destroyAllWindows()
    record_window.destroy()


def get_registered_users_count():
    if os.path.exists("StudentDetails/StudentDetails.csv"):
        with open("StudentDetails/StudentDetails.csv", 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            return len(rows) - 1
    return 0


def update_registered_users_label():
    count = get_registered_users_count()
    lbl_registered_users.config(text=f"Registered Users: {count}")


def send_attendance_email(file_path):
    subject = "Attendance Report"
    body = "Please find the attached attendance file."
    message = MIMEMultipart()
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECEIVER
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with open(file_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
    message.attach(part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message.as_string())
        print("Attendance email sent successfully.")
    except Exception as e:
        print(f"Failed to send attendance email: {e}")


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


def prompt_password():
    saved_pass = load_pass()
    if not saved_pass:
        new_pass = tsd.askstring('Set Password', 'No password set. Please set a new password:', show='*')
        if new_pass:
            save_pass(new_pass)
            mess.showinfo('Success', 'Password has been set successfully!')
            return True
        mess.showerror('Error', 'Password not set!')
        return False

    entered_pass = tsd.askstring('Enter Password', 'Enter your password:', show='*')
    if entered_pass == saved_pass:
        return True
    forgot_choice = mess.askyesno('Error', 'Incorrect password! Forgot password?')
    if forgot_choice:
        email_entered = tsd.askstring('Forgot Password', 'Enter the email of admin to reset password:')
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
        update_registered_users_label()
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
                name = "Unknown"
                try:
                    with open("StudentDetails/StudentDetails.csv", 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            if row and str(row[0]).strip() == str(id_):
                                name = row[1].strip()
                                break
                except Exception as e:
                    mess.showerror('Error', f"Error reading StudentDetails.csv: {e}")
                    cam.release()
                    cv2.destroyAllWindows()
                    return

                date = datetime.now().strftime('%Y-%m-%d')
                time_ = datetime.now().strftime('%H:%M:%S')

                if [id_, name, date, time_] not in attendance:
                    attendance.append([id_, name, date, time_])

                cv2.putText(img, f"{name} - {id_}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Tracking', img)

        if attendance:
            save_attendance_to_excel(attendance)
            mess.showinfo('Info', 'Attendance saved successfully!')
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if not attendance:
        mess.showinfo('Info', 'No attendance recorded.')


def record_attendance():
    """Continuously record attendance and capture unknown intruders."""
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
    intruder_count = 0

    intruder_dir = "Intruder"
    assure_path_exists(intruder_dir)

    recording_active = True

    record_window = tk.Toplevel()
    record_window.title("Recording Attendance")

    lbl_video = tk.Label(record_window, text="Recording attendance... Close this window or press Stop to exit.")
    lbl_video.pack(pady=10)

    def stop_recording():
        nonlocal recording_active
        recording_active = False
        cam.release()
        cv2.destroyAllWindows()
        record_window.destroy()

    btn_stop = tk.Button(record_window, text="Stop", command=stop_recording)
    btn_stop.pack(pady=5)

    try:
        while recording_active:
            ret, img = cam.read()
            if not ret:
                mess.showerror('Error', 'Failed to capture image!')
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            print(f"Detected faces: {faces}")  # Debug: Check detected faces

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                id_, conf = recognizer.predict(gray[y:y + h, x:x + w])

                if conf < 50:
                    name = "Unknown"
                    try:
                        with open("StudentDetails/StudentDetails.csv", 'r') as f:
                            reader = csv.reader(f)
                            next(reader)
                            for row in reader:
                                if row and str(row[0]).strip() == str(id_):
                                    name = row[1].strip()
                                    break
                    except Exception as e:
                        mess.showerror('Error', f"Error reading StudentDetails.csv: {e}")
                        cam.release()
                        cv2.destroyAllWindows()
                        record_window.destroy()
                        return

                    date = datetime.now().strftime('%Y-%m-%d')
                    time_ = datetime.now().strftime('%H:%M:%S')

                    if [id_, name, date, time_] not in attendance:
                        attendance.append([id_, name, date, time_])

                    cv2.putText(img, f"{name} - {id_}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                else:
                    intruder_count += 1
                    intruder_filename = f"{intruder_dir}/Intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{intruder_count}.jpg"
                    cropped_face = img[y:y + h, x:x + w]
                    success = cv2.imwrite(intruder_filename, cropped_face)
                    print(f"Image saved successfully: {success} at {intruder_filename}")  # Debug: Image save status

                    if success:
                        print(f"Intruder image saved as {intruder_filename}")
                        send_mail_with_image(intruder_filename)
                    else:
                        print("Failed to save intruder image.")

                    cv2.putText(img, "Intruder", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('Recording Attendance', img)

            if attendance:
                save_attendance_to_excel(attendance)
                attendance.clear()

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        record_window.destroy()

    email_intruder_report()


def email_intruder_report():
    """Email attendance report and intruder images."""
    folder = "Attendance"
    files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]
    if not files:
        mess.showinfo('Info', 'No attendance report found to send.')
        return

    latest_report = max([os.path.join(folder, f) for f in files], key=os.path.getctime)
    intruder_files = [os.path.join("Intruder", f) for f in os.listdir("Intruder") if f.endswith(".jpg")]

    subject = "Attendance Report with Intruder Alert"
    body = "Attached is the attendance report along with detected intruder photos." if intruder_files else "Attached is the attendance report. No intruders were detected."

    message = MIMEMultipart()
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECEIVER
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with open(latest_report, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(latest_report)}")
    message.attach(part)

    for intruder_file in intruder_files:
        with open(intruder_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(intruder_file)}")
        message.attach(part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message.as_string())
        mess.showinfo('Success', 'Email sent successfully with intruder photos.')
    except Exception as e:
        mess.showerror('Error', f'Failed to send email: {e}')


def view_attendance():
    """View attendance records with password protection."""
    if not prompt_password():
        return

    folder = "Attendance"
    files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]

    if not files:
        mess.showinfo('Info', 'No attendance records found.')
        return

    view_window = tk.Toplevel()
    view_window.title("View Attendance")

    tree = ttk.Treeview(view_window)
    tree.pack(fill=tk.BOTH, expand=True)

    latest_file = max([os.path.join(folder, f) for f in files], key=os.path.getctime)
    df = pd.read_excel(latest_file)

    tree["columns"] = list(df.columns)
    tree.column("#0", width=0, stretch=tk.NO)
    for col in df.columns:
        tree.column(col, anchor=tk.W, width=100)
        tree.heading(col, text=col, anchor=tk.W)

    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))

    view_window.mainloop()


window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("800x500")
window.configure(bg='#2c3e50')

header = tk.Label(window, text="Face Recognition Attendance System", bg='#2c3e50', fg="white", font=('Helvetica', 16, 'bold'))
header.pack(pady=20)

lbl_registered_users = tk.Label(window, text="Registered Users: 0", bg='#2c3e50', fg="white", font=('Helvetica', 12))
lbl_registered_users.pack(pady=10)

def show_help():
    mess.showinfo('Help', f'For assistance, contact: {EMAIL}')

tk.Button(window, text="Help", command=show_help, bg="#34495e", fg="white").place(x=730, y=10)

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

def on_closing():
    global cam
    try:
        if 'cam' in globals() and cam.isOpened():
            cam.release()
    except NameError:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    window.destroy()

def capture_attendance():
    track_images()

btn_take = tk.Button(input_frame, text="Take Images", command=take_images)
btn_take.grid(row=2, column=0, padx=10, pady=10)

btn_train = tk.Button(input_frame, text="Train Images", command=train_images)
btn_train.grid(row=2, column=1, padx=10, pady=10)

button_frame = tk.Frame(window, bg='#2c3e50')
button_frame.pack(pady=10)

btn_capture = tk.Button(button_frame, text="Capture Attendance", command=capture_attendance)
btn_capture.pack(side=tk.LEFT, padx=10)

btn_record = tk.Button(button_frame, text="Record Attendance", command=record_attendance)
btn_record.pack(side=tk.LEFT, padx=10)

btn_view = tk.Button(window, text="View Attendance", command=view_attendance)
btn_view.pack(pady=10)

btn_change_pass = tk.Button(window, text="Change Password", command=change_pass)
btn_change_pass.place(x=600, y=450)

btn_exit = tk.Button(window, text="Exit", command=on_closing)
btn_exit.pack(pady=10)

window.protocol("WM_DELETE_WINDOW", on_closing)
update_registered_users_label()
window.mainloop()
