import datetime
import os
import time
from threading import Thread

import pyramids
import heartrate
import preprocessing
import eulerian

import cv2
import numpy as np
from flask import Flask, render_template, Response, request

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0

freq_min = 1
freq_max = 1.8

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instantiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if face:
                frame = detect_face(frame)
            if grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if neg:
                frame = cv2.bitwise_not(frame)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if rec:
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "REC", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        # elif request.form.get('grey') == 'Grey':
        #     global grey
        #     grey = not grey
        # elif request.form.get('neg') == 'Negative':
        #     global neg
        #     neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if face:
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('./videos/video2.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif not rec:
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/heartbeat')
def heartbeat():
    # Preprocessing phase
    print("Reading + preprocessing video...")
    video_frames, frame_ct, fps = preprocessing.read_video("videos/video2.avi")

    # Build Laplacian video pyramid
    print("Building Laplacian video pyramid...")
    lap_video = pyramids.build_video_pyramid(video_frames)

    amplified_video_pyramid = []

    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video) - 1:
            continue

        # Eulerian magnification with temporal FFT filtering
        print("Running FFT and Eulerian magnification...")
        result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
        lap_video[i] += result

        # Calculate heart rate
        print("Calculating heart rate...")
        heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)

    # Collapse laplacian pyramid to generate final video
    # print("Rebuilding final video...")
    amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

    # Output heart rate and final video
    print("Heart rate: ", heart_rate, "bpm")
    return f"<h1>Your HeartBeat is: {round(heart_rate, 3)} <h1>"

    # for frame in amplified_frames:
    #     cv2.imshow("frame", frame)
    #     cv2.waitKey(20)



if __name__ == "__main__":
    app.run(debug=False)

camera.release()
cv2.destroyAllWindows()
