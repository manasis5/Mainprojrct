from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pygame
import threading

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("best_drowsiness_model.keras")

# Initialize pygame mixer for alarm
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alam.mp3")

# Load Haarcascade Classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

# Global variable to store drowsiness status
drowsiness_status = "Not Drowsy"

# Function to play alarm
def play_alarm():
    if not pygame.mixer.get_busy():
        alarm_sound.play(loops=-1)

# Function to stop alarm
def stop_alarm():
    alarm_sound.stop()

# Function to generate video frames
def generate_frames():
    global drowsiness_status
    cap = cv2.VideoCapture(0)
    score = 0
    alarm_on = False
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                left_eye = left_eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                right_eye = right_eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                
                eyes_closed = False
                eyes_detected = False
                
                for (ex, ey, ew, eh) in left_eye + right_eye:
                    eye = roi_color[ey:ey+eh, ex:ex+ew]
                    eye = cv2.resize(eye, (64, 64))
                    eye = np.expand_dims(eye, axis=0) / 255.0
                    prediction = model.predict(eye)
                    eyes_detected = True
                    if prediction < 0.5:
                        eyes_closed = True
                
                if eyes_detected:
                    if eyes_closed:
                        score += 1
                        if score >= 10 and not alarm_on:
                            threading.Thread(target=play_alarm).start()
                            alarm_on = True
                        drowsiness_status = "DROWSY!"  # Update status
                    else:
                        score = max(0, score - 1)
                        if score < 10 and alarm_on:
                            stop_alarm()
                            alarm_on = False
                        drowsiness_status = "Not Drowsy"  # Update status
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

# Route for the main UI
@app.route('/')
def index():
    return render_template('index.html')

# Route for live video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# New API route to fetch drowsiness status
@app.route('/drowsiness_status')
def get_drowsiness_status():
    return jsonify({"status": drowsiness_status})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Enable multi-threading for better performance
