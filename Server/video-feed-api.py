from flask import Flask, render_template, Response, send_from_directory
import cv2 as cv
import numpy as np
import os
import datetime
import time
import discord
from discord.ext import commands, tasks
import face_recognition


app = Flask(__name__, template_folder='templates')
video_feed = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
body_cascade = cv.CascadeClassifier('./cascades/haarcascade_upperbody.xml')
eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_eye.xml')
hand_cascade = cv.CascadeClassifier('./cascades/hand.xml')

# base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory where recorded videos will be saved
video_output_path = 'recorded_videos'
os.makedirs(video_output_path, exist_ok=True)


def generate_frames():
    frame_size = (int(video_feed.get(3)), int(
        video_feed.get(4)))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    detection = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 5
    while True:
        success, frame = video_feed.read()
        if not success:
            break

        grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        bodies = body_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=2, minNeighbors=1, minSize=(100, 100))
        # eyes = eye_cascade.detectMultiScale(
        #     grayscale_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        hands = hand_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

        for x, y, w, h in bodies:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # for x, y, w, h in eyes:
        #     cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

        for x, y, w, h in hands:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

        if len(faces) + len(bodies)+len(hands) > 0:
            if detection:
                timer_started = False
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out = cv.VideoWriter(
                    f"{video_output_path}/{current_time}.mp4", fourcc, 20.0, frame_size)
                print("Started recording!")
        elif detection:
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    detection = False
                    timer_started = False
                    out.release()
                    print("Stopped recording!")
            else:
                timer_started = True
                detection_stopped_time = time.time()
        if detection:
            out.write(frame)  # writes frame

        _, buffer = cv.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.route('/live-feed')
def live_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live-surveillance')
def index():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recordings')
def recordings():
    videos = []
    for filename in os.listdir(video_output_path):
        if filename.endswith('.mp4'):
            filepath = os.path.join(video_output_path, filename)
            datetime_str = filename.split('.')[0]
            videos.append({'filepath': filename, 'datetime': datetime_str})
# # Route to serve static video files
    return render_template('recordings.html', videos=videos)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
