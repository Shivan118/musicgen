from flask import Flask, redirect, url_for, request, render_template, jsonify, Response
from flask import session, abort
import flask_login
from flask_login import current_user
#login code
import os
import pathlib
from datetime import datetime
import requests
from flask import Flask, session, abort, redirect, request, render_template
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
from flask_mail import Mail, Message
from dotenv import load_dotenv
load_dotenv()

# Read credentials from environment variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

app = Flask("SonicSerenity")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default_secret_key")

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")

# app.config.update(
#     SESSION_COOKIE_SECURE=True,
#     SESSION_COOKIE_HTTPONLY=True,
#     SESSION_COOKIE_SAMESITE='Lax',
# )

login_manager = flask_login.LoginManager()
login_manager.init_app(app)
# login_manager.login_view = 'login'
# login_manager.session_protection = "strong"

mail = Mail(app)

# Get the JSON file path from .env
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Firebase using the credentials file
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

db = firestore.client()
bucket = storage.bucket()

doc = db.collection("SIgnal").document("YVNeDHCXVFvg0G3wZc6c")
doc.set({"play":"true"})


os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Fetch OAuth Credentials
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_SCOPES = os.getenv("GOOGLE_SCOPES").split(",")


# Create OAuth flow using environment variables instead of client_secret.json
flow = Flow.from_client_config(
    {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [GOOGLE_REDIRECT_URI]
        }
    },
    scopes=GOOGLE_SCOPES,
)

# Fetch Firebase Credentials
config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
}



class User(flask_login.UserMixin): 
    def __init__(self, id, username, type):
        self.id = id
        self.type = type
        self.username = username

    def get_id(self):
        return self.id

@login_manager.user_loader
def user_loader(email):  
    # user = User()
    # user.id = email
    # return user
    if len(user_details):
        return User(user_details['email'], user_details['name'], type[0])
    return None

def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)  # Authorization required
        else:
            return function()

    return wrapper

steps = [False]
user_session_data = {}
therapists_session_data = []

type = [""]
@app.route("/google_login", methods = ['POST', 'GET'] )
def g_login():
    # if(not steps[0]):
    if request.method == 'POST':
        result = request.form
        a = dict(result)
        type[0] = str(a["type"])
    
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)

user_details= {}
user_data = {}

def check_user(email, type):
    try:
        users_ref = db.collection(type.capitalize())
        print(f"Checking email: {email} in collection: {type.capitalize()}")
        query = users_ref.where('email', '==', email).limit(1)
        results = query.get()
        return len(results) > 0
    except Exception as e:
        print(f'An error occurred: {e}')
        return False

# Function to add a new user if they don't exist
def add_user(user_data, type):
    users_ref = db.collection(type.capitalize())
    users_ref.add(user_data)

@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)

    # if not session["state"] == request.args["state"]:
    #     abort(500)  # State does not match!

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    print(id_info)
    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    session["profile_image"] = id_info.get("picture")
    user_details["name"] = id_info.get("name").split(" ")[0]
    user_details["full_name"] = id_info.get("name")
    user_details["picture"] = id_info.get("picture")
    user_details["email"] = id_info.get("email")
    user_details["custom_audio"] = {}
    user_data["email"] = id_info.get("email")
    user_data["picture"] = id_info.get("picture")
    user_data["name"] = id_info.get("name").split(" ")[0]
    user_data["full_name"] = id_info.get("name")
    user_data["custom_audio"] = {}

    if(type[0] == ""):
        type[0] = "users"

    user = User(user_data["email"], user_data["name"], type[0])
    flask_login.login_user(user)

    if check_user(user_data["email"], type[0]):
        print("User already exists")
    else:
        add_user(user_data, type[0])
        print("User added successfully")
    return redirect("/protected_area")

# @app.route("/logout")
# def logout():
#     # print(session)
#     # session.clear()
#     # # auth.current_user = None
#     # print(session)
#     # # return redirect("/")
#     return render_template("index.html",logged_in="no", user_details = user_details)

@app.route("/l", methods =["POST", "GET"])
def l():
    session.clear()
    user_details.clear()
    authenticated_user_type[0] = ""
    return render_template("index.html",flask_login = flask_login,logged_in="no", user_details = user_details)

@app.route("/protected_area")
@login_is_required
def protected_area():
    try : 
        if page == "index" :
            return render_template("index.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "about":
            return render_template("about_us.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "faq":
            return render_template("faq.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "services":
            return render_template("our_services.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "graph":
            return render_template("graph.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "contact":
            return render_template("contact_us.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "approach":
            return render_template("approach.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
        elif page == "therapists":
            get_auth_type()
            get_session_data()
            print("ghf",authenticated_user_type[0])
            return render_template("therapist_tab.html",flask_login = flask_login, logged_in="yes", user_details = user_details, authenticated_user_type = authenticated_user_type[0], therapists_session_data = therapists_session_data, user_session_data = user_session_data, length = 0, alert_msg = "")
        else:
            custom_audio = get_custom_audio(user_details["email"])
            custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
            return render_template("steps.html",flask_login = flask_login, steps="", logged_in="yes", user_details = user_details, id="", custom_audio = custom_audio, custom_audio_url = custom_audio_url)
    except Exception as e :
        print(e)
        custom_audio = get_custom_audio(user_details["email"])
        custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
        return render_template("steps.html",flask_login = flask_login, steps="", logged_in="yes", user_details = user_details, id="", custom_audio = custom_audio, custom_audio_url = custom_audio_url)

authenticated_user_type = [""]

def check_if_user_logged_in():
    if len(user_details) :
        users_ref = db.collection("Users")
        query = users_ref.where('email', '==', current_user.id)
        results = query.get()
        if(len(results)):
            for i in results:
                user_details.update(i.to_dict())
                print(user_details)
                authenticated_user_type[0] = "Users"
        else :
            users_ref = db.collection("Therapists")
            query = users_ref.where('email', '==', current_user.id)
            results = query.get()
            if(len(results)):
                for i in results:
                    user_details.update(i.to_dict())
                    print(user_details)
                    authenticated_user_type[0] = "Therapists"
        return True
    else:
        return False
    
def get_auth_type():
    if len(user_details) :
        users_ref = db.collection("Therapists")
        query = users_ref.where('email', '==', current_user.id)
        results = query.get()
        print(f"usern {current_user.type} id {current_user.id}")
        if(len(results) and current_user.type == "therapists"):
            for i in results:
                user_details.update(i.to_dict())
                print(user_details)
                print("theraps")
                authenticated_user_type[0] = "Therapists"
        else :
            users_ref = db.collection("Users")
            query = users_ref.where('email', '==', current_user.id)
            results = query.get()
            if(len(results)):
                for i in results:
                    user_details.update(i.to_dict())
                    print(user_details)
                    authenticated_user_type[0] = "Users"

@app.route('/')
def start_page():
    #if check_if_user_logged_in():
       # print(user_details["full_name"])
        #return render_template("index.html",flask_login = flask_login,logged_in="yes", user_details = user_details)
   # else:
        return render_template("index.html",flask_login = flask_login,logged_in="no", user_details = user_details)

@app.route("/index")
def index():
    if(len(user_details)):
        return render_template("index.html",flask_login = flask_login,logged_in="yes", user_details = user_details)
    return render_template("index.html",flask_login = flask_login,logged_in="no", user_details = user_details)

@app.route("/login", methods=["POST"])
def login_page():
    global page
    page = ""
    if(not steps[0]):
        if request.method == 'POST':
            result = request.form
            a = dict(result)
            page = a["page"]
    return render_template("login.html")

#therapy code

from multiprocessing import Process, Queue, Event
import cv2
import numpy as np
import librosa
import pyaudio
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.signal import find_peaks
import cProfile
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components
from bokeh.io.export import export_png
import io

f = open("abc.txt", "w")
f.close()
f2 = open("current.txt", "w")
f2.close()
f3 = open("history.txt", "w")
f3.close()
f4 = open("freqs.txt", "w")
f4.close()

stop_event = Event()
capture = [False]

#mapping the different emotions to different solfeggio frequencies
def emotion_to_frequency(emotion, dominant_frequency = None):
    emotion_frequencies = {
        'anger': [417, 741],
        'fear': [285, 396],
        'happy': [639, 852, 963], 
        'neutral': [dominant_frequency, dominant_frequency],
        'sadness': [174, 432]
    }
    return emotion_frequencies.get(emotion.lower())

#calculating how many semitones to shift a frequency to match another frequency
def calculate_pitch_shift_semitones(target_frequency, reference_frequency):
    return 12 * np.log2(target_frequency / reference_frequency)

def webcam_emotion_detection(start_time, frame_skip=5, stability_check=2):

    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model("improved_model.h5")
    print("load")
    with open("process.txt", "r") as f:
        data = f.read()
    emotion_labels = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
    frame_count = 0
    previous_weights = np.zeros(len(emotion_labels))
    dominant_emotion_index = -1
    timer_start = None

    while(data.find("Start") == -1):
        with open("process.txt", "r") as f:
            data = f.read()
        continue

    global stop_event
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break 

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                predictions = classifier.predict(roi)[0]
                current_dominant_index = np.argmax(predictions)
                if dominant_emotion_index != current_dominant_index:
                    timer_start = time.time()
                    dominant_emotion_index = current_dominant_index
                elif timer_start is not None and (time.time() - timer_start) >= stability_check:
                    current_time = time.time() - start_time
                    with open("abc.txt", "w") as f:
                        f.write(str(dominant_emotion_index))
                    with open("current.txt", "a") as f:
                        f.write(f'{current_time},{str(dominant_emotion_index)},{str(predictions[dominant_emotion_index])}\n')
                    with open('history.txt', "a") as f:
                        f.write(f'{current_time},{",".join(map(str, predictions))}\n')
                    timer_start = None
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{emotion_labels[dominant_emotion_index]}: {predictions[dominant_emotion_index] * 100:.2f}%"
                cv2.putText(frame, text, (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        fram = buffer.tobytes()
        yield (b'--frame\r\n'   
                b'Content-Type: image/jpeg\r\n\r\n' + fram + b'\r\n') 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    print("camera closed")
    cap.release()
    cv2.destroyAllWindows()

def base_freq(y, sr):
    nfft=2000  
    freqs = librosa.fft_frequencies(sr=sr,n_fft=nfft)
    
    #calculate fft 
    S = np.abs(librosa.stft(y,n_fft=nfft)) #stft faster algo, abs to remove neg, nfft should be twice of max freq u wanna detect 

    #mean of fft
    mean_S=np.mean(S,1) #60s divided into many chunks, mean of all chunks
    mean_S_trimmed = mean_S[np.where(freqs>150)]

    #find loval maxima peaks (indexes)
    peaks, t = find_peaks(mean_S, height=1)

    # find frequencies corresponding to the peak indexes
    peak_values=freqs[peaks]

    solfeggio_freqs = [174, 285, 396, 417, 432, 528, 639, 693, 741, 852, 963]

    #find closest solfeggio frequency to the peak frequencies
    diff = np.inf
    base_freq = None
    d=np.zeros([len(peak_values)*len(solfeggio_freqs),1])
    i=0

    for peak_value in peak_values:
        #print(peak_value)
        for target_freq in solfeggio_freqs:
            d= abs(peak_value - target_freq)
            if d < diff:
                #print(peak_value)
                diff = d
                base_freq = peak_value

    return base_freq

def base_freq_ps(y, sr, target):
    nfft=2000  
    freqs = librosa.fft_frequencies(sr=sr,n_fft=nfft)
    
    #calculate fft 
    S = np.abs(librosa.stft(y,n_fft=nfft)) #stft faster algo, abs to remove neg, nfft should be twice of max freq u wanna detect 

    #mean of fft
    mean_S=np.mean(S,1) #60s divided into many chunks, mean of all chunks
    mean_S_trimmed = mean_S[np.where(freqs>150)]

    #find loval maxima peaks (indexes)
    peaks, t = find_peaks(mean_S, height=1)

    # find frequencies corresponding to the peak indexes
    peak_values=freqs[peaks]

    solfeggio_freqs = [174, 285, 396, 417, 432, 528, 639, 693, 741, 852, 963]

    #find closest solfeggio frequency to the peak frequencies
    diff = np.inf
    base_freq = None
    d=np.zeros([len(peak_values)*len(solfeggio_freqs),1])
    i=0

    for peak_value in peak_values:
        #print(peak_value)
        d= abs(peak_value - target)
        if d < diff:
            #print(peak_value)
            diff = d
            base_freq = peak_value

    return base_freq

#modifies the sound in chunks (for a smooth transition between chunks)
def targeted_window(chunk, window_size_ratio=0.01):
    chunk_length = len(chunk)#chunk in sample length
    window_size = int(chunk_length * window_size_ratio) #size of the window for fading based on the length of the chunk
    window_size = max(window_size, 2) #window size is at least 2(samples), not too small
    
    #create a hanning window to apply a fade in and fade out effect to the chunk
    fade_window = np.hanning(window_size * 2)
    fade_in = fade_window[:window_size]#split the window in half for the fade in 
    fade_out = fade_window[window_size:]#and half for the fade out
    chunk[:window_size] = chunk[:window_size] * fade_in #apply fade in at the chunk start
    chunk[-window_size:] = chunk[-window_size:] * fade_out #and fade out at the chunk end
    return chunk

def get_audio_file(id, audio_file):
    blob = bucket.blob(f"{id}/custom_audio/{audio_file}")
    blob.download_to_filename(audio_file)
    
def audio_processing(audio_file, start_time, id, user_name):
    print("in")
    with open("process.txt", "r") as f:
        data = f.read()
    emotion_labels = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
    print("hii")
    solfeggio_freqs = [174, 285, 396, 417, 432, 528, 639, 693, 741, 852, 963]
    # "tibetan_R34TKLkz.wav"
    # y, sr = librosa.load(audio_file, duration = 60)
    # print("first load")
    # bf = base_freq(y, sr, solfeggio_freqs)
    # print("cal")
    default_audio = ["crystal.wav","heal.wav","marimba.mp3","nature.wav","om.wav","space.wav","tibetan.wav"]
    if audio_file in default_audio:
        blob = bucket.blob(f"Songs/{audio_file}")
        blob.download_to_filename(audio_file)
    else:
        get_audio_file(id, audio_file)
    y, sr = librosa.load(audio_file, sr=None)
    print("loaded")
    p = pyaudio.PyAudio()
    print("audio pyaudio")
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    print("open")
    try : 
        doc_ref = db.collection('Songs').document("OiCdogKheZX0bljWo1AG")
        doc_snapshot = doc_ref.get()
        if doc_snapshot.exists:
            print(doc_snapshot.to_dict())
            bf = doc_snapshot.to_dict()[audio_file]                                                                 
            print(bf)
    except:
        doc_ref = db.collection("Users").where("email","==",id)
        result = doc_ref.get()
        if result:
            for i in result:
                d = i.to_dict()
                bf = d["custom_audio"][audio_file]
                print(bf, type(bf))

    doc = db.collection("SIgnal").document("YVNeDHCXVFvg0G3wZc6c")

    chunk_size = int(sr * 7)
    last_stable_emotion_index = -1
    last_frequency = bf
    pitch_shift_amount = 0  

    global stop_event
    print("hello")
    while(data.find("Start") == -1):
        with open("process.txt", "r") as f:
            data = f.read()
        continue
    print("start")

    doc.set({"play" : "false"})
    time.sleep(1)

    for i in range(0, len(y), chunk_size):
        print(i)
        current_chunk = y[i:i + chunk_size]

        try:
            with open("abc.txt", "r") as f:  # read the latest emotion index from this file
                new_dominant_emotion_index = int(f.read().strip()) #and convert it to integer
                if new_dominant_emotion_index != last_stable_emotion_index: #if emotion has changes since the last chunk
                    last_stable_emotion_index = new_dominant_emotion_index
                    current_emotion = emotion_labels[new_dominant_emotion_index]
                    target_frequencies = emotion_to_frequency(current_emotion, last_frequency)
                    print('b_f',last_frequency)
                    print('curr emotion',current_emotion)
                    print('emotion frequencies:', target_frequencies)
                    p = [calculate_pitch_shift_semitones(fr, last_frequency) for fr in target_frequencies if fr is not None]
                    fifth = 7  #the interval of a perfect fifth in semitones
                    closest = min(p, key=lambda x: abs(x - fifth)) #find the shift closest to a perfect fifth
                    print('psa',closest)
                    target_index = p.index(closest) #get the index and actual frequency of this closest value
                    target = target_frequencies[target_index]
                    current_chunk = librosa.effects.pitch_shift(current_chunk, sr=sr, n_steps=closest)
                    current_chunk = targeted_window(current_chunk)
                    last_frequency = base_freq_ps(current_chunk, sr, target) #update last_frequency
                    print('t_f',target)

                    print('l_f update',last_frequency)
                    curr_time = time.time() - start_time
                    with open("freqs.txt", "a") as f:  #update the current time and last frequency in freqs.txt
                        f.write(f'{curr_time},{last_frequency}\n')
        except Exception as e:
            print(f"error: {e}")
    
        with open("process.txt", "r") as f:
            data = f.read()
        if(data == "True"):
            print("breaking")
            break

        stream.write(current_chunk.astype(np.float32).tobytes())

    stream.stop_stream() #close the stream and terminate the audio system
    stream.close()
    os.remove(audio_file)
    now = datetime.now()
    print(now)
    n = user_name
    formatted_now = now.strftime('%d-%m-%Y %H:%M:%S')
    uploading_audio_in_storage(formatted_now, id)
    graph_db(formatted_now, id, audio_file, n)
    p.terminate()  

def uploading_audio_in_storage(formatted_now, id):
    blob = bucket.blob(f"{id}/{formatted_now}/history.txt")
    blob.upload_from_filename("history.txt")
    blob = bucket.blob(f"{id}/{formatted_now}/freqs.txt")
    blob.upload_from_filename("freqs.txt")
    print("uploading done")

def graph_db(new, id, audio_name, name):
    try :
        users_ref = db.collection("History")
        query = users_ref.where('user_email', '==', id)
        results = query.get()
        if(results):
            for i in results :
                doc_ref = i.reference
                data = i.to_dict()
                sessions = data["sessions"]
                if(len(sessions) == 5):
                    # s = [j for j in sessions[0]][0]
                    # blobs = bucket.list_blobs(prefix=f'{id}/{s}/')
                    # for blob in blobs:
                    #     blob.delete()
                    sessions.pop(0)
                sessions.append({new : audio_name})
                doc_ref.update({"sessions" : sessions})
                print("data added")
        else:
            new_data = {"user_email" : id, "sessions" : [{new : audio_name}], "user_name" : name}
            users_ref.add(new_data)
            print("data added")
    except:
        users_ref = db.collection("History")
        new_data = {"user_email" : id, "sessions" : [[new, audio_name]], "user_name" : name}
        users_ref.add(new_data)

def upload_audio_processing(audio_file, start_time, id, user_name):
    print(audio_file)
    with open("process.txt", "r") as f:
        data = f.read()
    emotion_labels = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
    print("hii")
            
    # get_audio_file(id, audio_file)

    y, sr = librosa.load(audio_file, sr=None, duration = 60)
    print("first load")
    bf = base_freq(y, sr)
    print("cal")
    y, sr = librosa.load(audio_file, sr=None)
    print("loaded")
    p = pyaudio.PyAudio()
    print("audio pyaudio")
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    print("open")

    doc = db.collection("SIgnal").document("YVNeDHCXVFvg0G3wZc6c")

    doc_r = db.collection("Users").where("email","==",id)
    result = doc_r.get()
    if result:
        for i in result:
            doc_ref_ = i.reference
            d = i.to_dict()
            custom_audio = d["custom_audio"]
            custom_audio.clear()
            custom_audio.update({audio_file : bf})
            doc_ref_.update({"custom_audio":custom_audio})

    chunk_size = int(sr * 7)
    last_stable_emotion_index = -1
    last_frequency = bf
    pitch_shift_amount = 0  

    global stop_event
    print("hello")
    while(data.find("Start") == -1):
        with open("process.txt", "r") as f:
            data = f.read()
        continue
    print("start")

    doc.set({"play" : "false"})
    time.sleep(0.5)
    print("play")

    for i in range(0, len(y), chunk_size):
        print(i)
        current_chunk = y[i:i + chunk_size]

        try:
            with open("abc.txt", "r") as f:  # read the latest emotion index from this file
                new_dominant_emotion_index = int(f.read().strip()) #and convert it to integer
                if new_dominant_emotion_index != last_stable_emotion_index: #if emotion has changes since the last chunk
                    last_stable_emotion_index = new_dominant_emotion_index
                    current_emotion = emotion_labels[new_dominant_emotion_index]
                    target_frequencies = emotion_to_frequency(current_emotion, last_frequency)
                    print('b_f',last_frequency)
                    print('curr emotion',current_emotion)
                    print('emotion frequencies:', target_frequencies)
                    p = [calculate_pitch_shift_semitones(fr, last_frequency) for fr in target_frequencies if fr is not None]
                    fifth = 7  #the interval of a perfect fifth in semitones
                    closest = min(p, key=lambda x: abs(x - fifth)) #find the shift closest to a perfect fifth
                    print('psa',closest)
                    target_index = p.index(closest) #get the index and actual frequency of this closest value
                    target = target_frequencies[target_index]
                    current_chunk = librosa.effects.pitch_shift(current_chunk, sr=sr, n_steps=closest)
                    current_chunk = targeted_window(current_chunk)
                    last_frequency = base_freq_ps(current_chunk, sr, target) #update last_frequency
                    print('t_f',target)
                    print('l_f update',last_frequency)
                    curr_time = time.time() - start_time
                    with open("freqs.txt", "a") as f:  #update the current time and last frequency in freqs.txt
                        f.write(f'{curr_time},{last_frequency}\n')
        except Exception as e:
            print(f"error: {e}")
    
        with open("process.txt", "r") as f:
            data = f.read()
        if(data == "True"):
            print("breaking")
            break

        stream.write(current_chunk.astype(np.float32).tobytes())

    stream.stop_stream() #close the stream and terminate the audio system
    stream.close()
    os.remove(audio_file)
    now = datetime.now()
    print(now)
    n = user_name
    formatted_now = now.strftime('%d-%m-%Y %H:%M:%S')
    uploading_audio_in_storage(formatted_now, id)
    graph_db(formatted_now, id, audio_file, n)
    p.terminate()

def upload_audio_no_processing(audio_file, start_time, id, user_name):
    print("abnormal")
    print(audio_file)
    with open("process.txt", "r") as f:
        data = f.read()
    emotion_labels = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
    print("hii")

    # get_audio_file(id, audio_file)
    y, sr = librosa.load(audio_file, sr=None, duration = 60)
    print("first load")
    bf = base_freq(y, sr)
    print("cal")

    y, sr = librosa.load(audio_file, sr=None)
    print("loaded")
    p = pyaudio.PyAudio()
    print("audio pyaudio")
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
    print("open")

    doc = db.collection("SIgnal").document("YVNeDHCXVFvg0G3wZc6c")

    doc_r = db.collection("Users").where("email","==",id)
    result = doc_r.get()
    if result:
        for i in result:
            doc_ref_ = i.reference
            d = i.to_dict()
            custom_audio = d["custom_audio"]
            custom_audio.clear()
            custom_audio.update({audio_file : bf})
            doc_ref_.update({"custom_audio":custom_audio})

    chunk_size = int(sr * 7)
    last_stable_emotion_index = -1
    last_frequency = bf
    pitch_shift_amount = 0  

    global stop_event
    print("hello")
    while(data.find("Start") == -1):
        with open("process.txt", "r") as f:
            data = f.read()
        continue
    print("start")

    doc.set({"play" : "false"})
    time.sleep(0.5)
    print("play")

    for i in range(0, len(y), chunk_size):

        print(i)
        current_chunk = y[i:i + chunk_size]
        chunk = y[i:i + chunk_size]

        try:
            with open("abc.txt", "r") as f:  # read the latest emotion index from this file
                new_dominant_emotion_index = int(f.read().strip()) #and convert it to integer
                if new_dominant_emotion_index != last_stable_emotion_index: #if emotion has changes since the last chunk
                    last_stable_emotion_index = new_dominant_emotion_index
                    current_emotion = emotion_labels[new_dominant_emotion_index]
                    target_frequencies = emotion_to_frequency(current_emotion, last_frequency)
                    print('b_f',last_frequency)
                    print('curr emotion',current_emotion)
                    print('emotion frequencies:', target_frequencies)
                    p = [calculate_pitch_shift_semitones(fr, last_frequency) for fr in target_frequencies if fr is not None]
                    fifth = 7  #the interval of a perfect fifth in semitones
                    closest = min(p, key=lambda x: abs(x - fifth)) #find the shift closest to a perfect fifth
                    print('psa',closest)
                    target_index = p.index(closest) #get the index and actual frequency of this closest value
                    target = target_frequencies[target_index]
                    # current_chunk = librosa.effects.pitch_shift(current_chunk, sr=sr, n_steps=closest)
                    # current_chunk = targeted_window(current_chunk)
                    last_frequency = base_freq_ps(current_chunk, sr, target) #update last_frequency
                    print('t_f',target)
                    print('l_f update',last_frequency)
                    curr_time = time.time() - start_time
                    with open("freqs.txt", "a") as f:  #update the current time and last frequency in freqs.txt
                        f.write(f'{curr_time},{last_frequency}\n')

        except Exception as e:
            print(f"error: {e}")

        with open("process.txt", "r") as f:
            data = f.read()
        if(data == "True"):
            print("breaking")
            break

        stream.write(chunk)

    
    stream.stop_stream()
    stream.close()
    os.remove(audio_file)
    now = datetime.now()
    print(now)
    n = user_name
    formatted_now = now.strftime('%d-%m-%Y %H:%M:%S')
    uploading_audio_in_storage(formatted_now, id)
    graph_db(formatted_now, id, audio_file, n)
    p.terminate()

@app.route('/get_started', methods = ['POST', 'GET'])
def get_started():
    print(user_details)
    f = open("process.txt","w")
    f.close()
    if len(user_details):
        steps[0] = False
        id = user_details["email"]
        print(id)
        custom_audio = get_custom_audio(id)
        custom_audio_url = get_custom_audio_url(id, custom_audio)
        return render_template("steps.html",flask_login = flask_login, step = "", logged_in="yes", user_details = user_details, id= "", custom_audio = custom_audio, custom_audio_url = custom_audio_url)
    else:
        steps[0] = True
        return redirect("/google_login")
    
audio = []

def get_custom_audio(id_):
    doc = db.collection("Users").where("email", "==", id_)
    result = doc.get()
    if result : 
        for i in result:
            a = i.to_dict()
            return a["custom_audio"]
    else:
        return {}

def get_custom_audio_url(id, custom_audio):
    a = {}
    for i in custom_audio:
        s = f"{id}%2Fcustom_audio%2F{(i).replace(' ', '%20')}"
        image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{s}?alt=media"
        print(image_url)
        a.update({i : image_url})
    return a

@app.route('/load', methods = ['POST', 'GET'])
def load_on():
    if request.method == 'POST':
        result = request.form
        a = dict(result)
        print(result)
        global audio_file
        if(a["audio"] != "g"):
            audio_file = a["audio"]
        else:
            audio_file = audio[0]
        stop_event.clear()
        start_time = time.time()
        global audio_process
        global webcam_process
        if len(user_details) :
            id = user_details["email"]
            name = user_details["full_name"]
        else :
            id = ""
            name = ""
        audio_process = Process(target=audio_processing, args=(audio_file, start_time, id, name))
        audio_process.start() 
        webcam_process = Process(target=webcam_emotion_detection, args=(start_time,))
        webcam_process.start()
        custom_audio = get_custom_audio(user_details["email"])
        custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
    return render_template("steps.html",flask_login = flask_login, step = "1", logged_in="yes", user_details = user_details, id=id, custom_audio = custom_audio, custom_audio_url = custom_audio_url)

@app.route('/turn_camera_on', methods = ['POST', 'GET'])
def camera_on():
    if request.method == 'POST':
        result = request.form
        a = dict(result)
        capture[0] = True
        print(result) 
        start_time = time.time()
        custom_audio = get_custom_audio(user_details["email"])
        custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
    return render_template("steps.html",flask_login = flask_login, step = "2", logged_in="yes", user_details = user_details, id="", custom_audio = custom_audio, custom_audio_url = custom_audio_url)

a = [1]
@app.route('/turn_audio_on', methods = ['POST', 'GET'])
def audio_on():
    with open("process.txt", "a") as f:
            f.write("Start")
    with open("history.txt", "w") as f:  
        f.write("")
    with open("current.txt", "w") as f:  
        f.write("")
    print("started") 
    custom_audio = get_custom_audio(user_details["email"])
    custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
    audio_process.join()
    webcam_process.join()
    return render_template("steps.html",flask_login = flask_login, step = "3", logged_in="yes", user_details = user_details, id="", custom_audio=custom_audio, custom_audio_url = custom_audio_url)

@app.route('/stop_audio', methods = ['POST', 'GET'])
def stop_audio():  
    print("Stopping processes...")
    stop_event.set()
    with open("process.txt", "w") as f:
        f.write(str(True))
    custom_audio = get_custom_audio(user_details["email"])
    custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
    return render_template("steps.html",flask_login = flask_login, step = "4", logged_in="yes", user_details = user_details, id="", custom_audio=custom_audio, custom_audio_url = custom_audio_url)

@app.route('/show_graph', methods = ['POST', 'GET'])
def show_graph(): 
    try :   
        weights = pd.read_csv('history.txt', header=None)
        weights.columns = ['Time', 'Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
        freq = pd.read_csv('freqs.txt', header=None, names=['Time', 'Frequency'])

        output_file("plots.html")

        emotion_source = ColumnDataSource(data=dict(
            Time=weights['Time'],
            Anger=weights['Anger'],
            Fear=weights['Fear'],
            Happy=weights['Happy'],
            Neutral=weights['Neutral'],
            Sadness=weights['Sadness'],
        ))

        emotion_plot = figure(width=800, height=400, title="Emotion Percentages Over Time", x_axis_label='Time', y_axis_label='Emotion Percentages(Weights)')
        emotion_hover = HoverTool(tooltips=[
            ("Time", "@Time"),
            ("Anger", "@Anger{0.2f}"),
            ("Fear", "@Fear{0.2f}"),
            ("Happy", "@Happy{0.2f}"),
            ("Neutral", "@Neutral{0.2f}"),
            ("Sadness", "@Sadness{0.2f}"),
        ])
        emotion_plot.add_tools(emotion_hover)

        colors = ["orange", "red", "green", "blue", "purple"]
        for emotion, color in zip(['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness'], colors):
            emotion_plot.line('Time', emotion, source=emotion_source, color=color, legend_label=emotion)

        frequency_source = ColumnDataSource(data=dict(
            Time=freq['Time'],
            Frequency=freq['Frequency'],
        ))

        frequency_plot = figure(width=800, height=400, title="Frequency Over Time", x_axis_label='Time', y_axis_label='Frequency')
        frequency_hover = HoverTool(tooltips=[
            ("Time", "@Time"),
            ("Frequency", "@Frequency"),
        ])

        frequency_plot.add_tools(frequency_hover)

        frequency_plot.line('Time', 'Frequency', source=frequency_source, color="black", legend_label='Frequency')

        script, div = components(emotion_plot)
        sc, di = components(frequency_plot)
        return render_template('graph.html',flask_login = flask_login, script=script, div=div, sc=sc, di=di, user_details = user_details, logged="yes")
    except Exception as e:
        print(e)
        return render_template('graph.html',flask_login = flask_login, script="", div="", sc="", di="", user_details = user_details, logged="yes")

@app.route('/upload', methods=["POST", "GET"])
def upload_file():
    if len(user_details) :
        id = user_details["email"]
        name = user_details["full_name"]
    else :
        id = ""
        name = ""

    if 'audio_file' not in request.files:
        return 'No file part'
    
    print("sfydshnj")
    print(f"upload {request.form}")
    data = request.form
    print(data.to_dict())
    file = request.files['audio_file']

    if file.filename == '':
        print("No selected file")

    if file:
        file.save(file.filename)
        audio.append(str(file.filename))
        audio[0] = file.filename
        print("File uploaded successfully")
        doc_r = db.collection("Users").where("email","==",id)
        result = doc_r.get()
        if result:
            for i in result:
                doc_ref_ = i.reference
                d = i.to_dict()
                custom_audio = d["custom_audio"]
                if(len(custom_audio) >= 1) :
                    blobs = bucket.list_blobs(prefix=f'{id}/custom_audio/')
                    for blob in blobs:
                        blob.delete()
                blob = bucket.blob(f"{id}/custom_audio/{file.filename}")
                blob.upload_from_filename(file.filename)
        # s = f"{id}%2F{(file.filename).replace(" ", "%20")}"
        # image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{s}?alt=media"
        # print(image_url)

    stop_event.clear()
    start_time = time.time()
    global audio_process
    global webcam_process
    if(not data):
        a = "no"
        print("normal")
        audio_process = Process(target=upload_audio_processing, args=(audio[0], start_time, id, name))
    else:
        a = "yes"
        print("abnormal")
        audio_process = Process(target=upload_audio_no_processing, args=(audio[0], start_time, id, name))
    audio_process.start() 
    webcam_process = Process(target=webcam_emotion_detection, args=(start_time,))
    webcam_process.start()
    custom_audio = get_custom_audio(user_details["email"])
    custom_audio_url = get_custom_audio_url(user_details["email"], custom_audio)
    return render_template("steps.html",flask_login = flask_login, step = "5", logged_in="yes", user_details = user_details, checkbox= a, custom_audio=custom_audio, custom_audio_url = custom_audio_url)

@app.route('/services', methods=["POST", "GET"])
def services():
    if len(user_details):
        return render_template("our_services.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
    else:
        return render_template("our_services.html",flask_login = flask_login, logged_in="no", user_details = user_details)

@app.route('/about_us', methods=["POST", "GET"])
def about_us():
    if len(user_details):
        return render_template("about_us.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
    else:
        return render_template("about_us.html",flask_login = flask_login, logged_in="no", user_details = user_details)

@app.route('/contact_us', methods=["POST", "GET"])
def contact_us():
    if len(user_details):
        return render_template("contact_us.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
    else:
        return render_template("contact_us.html",flask_login = flask_login, logged_in="no", user_details = user_details)
    
def get_session_data():
    user_session_data.clear()
    users_ref = db.collection("History")
    query = users_ref.where('user_email', '==', user_details["email"]).limit(1)
    results = query.get()
    if(results):
        for i in results:
            user_session_data.update(i.to_dict())

    therapists_session_data.clear()
    therapists_ref = db.collection("Shared Sessions")
    t_query = therapists_ref.where('therapist_email', '==', user_details["email"])
    t_results = t_query.get()
    print(t_results)
    if(t_results):
        for i in t_results:
            therapists_session_data.append(i.to_dict())

@app.template_filter('enumerate')
def do_enumerate(iterable):
    return enumerate(iterable)

@app.route('/therapists', methods=["POST", "GET"])
def therapists():
    if len(user_details):
        get_auth_type()
        get_session_data()
        print(f"user_session data {user_session_data}")
        print(f"therpists {therapists_session_data}")
        print(f"auth{authenticated_user_type[0]}")
        l = 0
        if user_session_data:
            if(len(user_session_data["sessions"])):
                l = len(user_session_data["sessions"])
            else:
                l = 0
        return render_template("therapist_tab.html",flask_login = flask_login, logged_in="yes", user_details = user_details, authenticated_user_type = authenticated_user_type[0], therapists_session_data = therapists_session_data, user_session_data = user_session_data, length_ = l, alert_msg = "")
    else:
        return render_template("therapist_tab.html",flask_login = flask_login, logged_in="no", user_details = user_details, authenticated_user_type = authenticated_user_type[0], therapists_session_data = therapists_session_data, user_session_data = user_session_data, length_ = 0, alert_msg = "")

@app.route("/show_patient_sessions", methods=["POST", "GET"])
def show_patient_sessions():
    if request.method == 'POST':
        b = request.form
        patient_data = {}
        doc = db.collection("Shared Sessions").where("patient_email", "==", b["p_email"])
        result = doc.get()
        if result:
            for i in result : 
                patient_data.update(i.to_dict())
        return render_template("patient_data.html",flask_login = flask_login, user_details = user_details, patient_data = patient_data)

@app.route('/share_graphs', methods=["POST", "GET"])
def share_graphs():
    if request.method == 'POST':
        flag = False
        result = request.form
        a = dict(result)
        sessions = a["sessions"]
        sessions = sessions.split(",")
        sessions.pop()
        print(a)
        for i in range(len(sessions)):
            sessions[i] = {sessions[i].split("+")[0] : sessions[i].split("+")[1]}
        therapist_email = a["therapist_email"]
        doc = db.collection("Shared Sessions").where("patient_email", "==", a["user_email"]).where("therapist_email", "==", a["therapist_email"])
        result = doc.get()
        if(len(result)):
            for i in result : 
                result_doc = i.reference
                result_data = i.to_dict()
                existing_sessions = result_data["sessions"]
                for i in sessions:
                    if i in existing_sessions:
                        continue
                    else:
                        if(len(existing_sessions) == 10):
                            existing_sessions.pop(0)
                            blobs = bucket.list_blobs(prefix=f'{a["user_email"]}/{existing_sessions[0]}/')
                            for blob in blobs:
                                blob.delete()
                        existing_sessions.append(i)
                result_doc.update({"sessions" : existing_sessions})
        else: 
            doc = db.collection("Shared Sessions")
            data = {"patient_name" : a["user_name"], "patient_email" : a["user_email"], "sessions" : sessions, "therapist_email" : a["therapist_email"]}
            doc.add(data)
        recipient = therapist_email
        subject = "New Session Shared"
        body = f"{a['user_name'].capitalize()} shared session with you. Login to SonicSerenity to check out."
        msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient])
        msg.body = body

        try:
            mail.send(msg)
            print('Email sent successfully!')
            return render_template("therapist_tab.html",flask_login = flask_login, logged_in="yes", user_details = user_details, authenticated_user_type = authenticated_user_type[0], therapists_session_data = therapists_session_data, user_session_data = user_session_data, length_ = len(user_session_data["sessions"]), alert_msg = "success")
        except Exception as e:
            print(f'Failed to send email: {str(e)}')
            return render_template("therapist_tab.html",flask_login = flask_login, logged_in="yes", user_details = user_details, authenticated_user_type = authenticated_user_type[0], therapists_session_data = therapists_session_data, user_session_data = user_session_data, length_ = 0, alert_msg = "unsuccess")

@app.route('/see_graphs_t', methods=["POST", "GET"])
def see_graphs_t():
    if request.method == 'POST':
        result = request.form
        a = dict(result)
        session_name = a["session"]
        id = a["email"]
        name = a["name"]
        try : 
            blob = bucket.blob(f"{id}/{session_name}/history.txt")
            if(not os.path.exists("show_graph")):
                os.mkdir("show_graph")
            blob.download_to_filename("show_graph/history.txt")
            blob = bucket.blob(f"{id}/{session_name}/freqs.txt")
            if(not os.path.exists("show_graph")):
                os.mkdir("show_graph")
            blob.download_to_filename("show_graph/freqs.txt")

            weights = pd.read_csv('show_graph/history.txt', header=None)
            weights.columns = ['Time', 'Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
            freq = pd.read_csv('show_graph/freqs.txt', header=None, names=['Time', 'Frequency'])

            output_file("plots.html")

            emotion_source = ColumnDataSource(data=dict(
                Time=weights['Time'],
                Anger=weights['Anger'],
                Fear=weights['Fear'],
                Happy=weights['Happy'],
                Neutral=weights['Neutral'],
                Sadness=weights['Sadness'],
            ))

            emotion_plot = figure(width=800, height=400, title="Emotion Percentages Over Time", x_axis_label='Time', y_axis_label='Emotion Percentages(Weights)')
            emotion_hover = HoverTool(tooltips=[
                ("Time", "@Time"),
                ("Anger", "@Anger{0.2f}"),
                ("Fear", "@Fear{0.2f}"),
                ("Happy", "@Happy{0.2f}"),
                ("Neutral", "@Neutral{0.2f}"),
                ("Sadness", "@Sadness{0.2f}"),
            ])
            emotion_plot.add_tools(emotion_hover)

            colors = ["orange", "red", "green", "blue", "purple"]
            for emotion, color in zip(['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness'], colors):
                emotion_plot.line('Time', emotion, source=emotion_source, color=color, legend_label=emotion)

            frequency_source = ColumnDataSource(data=dict(
                Time=freq['Time'],
                Frequency=freq['Frequency'],
            ))

            frequency_plot = figure(width=800, height=400, title="Frequency Over Time", x_axis_label='Time', y_axis_label='Frequency')
            frequency_hover = HoverTool(tooltips=[
                ("Time", "@Time"),
                ("Frequency", "@Frequency"),
            ])

            frequency_plot.add_tools(frequency_hover)

            frequency_plot.line('Time', 'Frequency', source=frequency_source, color="black", legend_label='Frequency')

            script, div = components(emotion_plot)
            sc, di = components(frequency_plot)

            # Plotting Emotion Percentages Over Time
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

            # Plot Emotion Percentages
            for emotion, color in zip(['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness'], ["orange", "red", "green", "blue", "purple"]):
                ax1.plot(weights['Time'], weights[emotion], color=color, label=emotion)

            ax1.set_title("Emotion Percentages Over Time")
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Emotion Percentages(Weights)')
            ax1.legend()

            # Plot Frequency
            ax2.plot(freq['Time'], freq['Frequency'], color="black", label='Frequency')
            ax2.set_title("Frequency Over Time")
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Frequency')
            ax2.legend()

            # Super title for the combined plot
            fig.suptitle(f"{name.title()}'s {session_name} Session", fontsize=14)
            print(session_name)
            plt.tight_layout()
            plt.savefig('static/images/plots.png')
            plt.close()

            # blob = bucket.blob("Graph Image/plots.png")
            # blob.upload_from_filename("show_graph/plots.png")
            # image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/Graph%20Image%2Fplots.png?alt=media"

            # # Print the URL for testing
            # print(f"Download URL: {image_url}")

            return render_template('see_graph_therapist.html', script=script, div=div, sc=sc, di=di, user_details = user_details, logged="yes", patient_name = name)
        except Exception as e:
            print(e)
            return render_template('see_graph_therapist.html', script="", div="", sc="", di="", user_details = user_details, logged="yes", patient_name = name)

@app.route('/see_graphs', methods=["POST", "GET"])
def see_graphs():
    if request.method == 'POST':
        result = request.form
        a = dict(result)
        session_name = a["session"]
        id = user_details["email"]
        try : 
            blob = bucket.blob(f"{id}/{session_name}/history.txt")
            if(not os.path.exists("show_graph")):
                os.mkdir("show_graph")
            blob.download_to_filename("show_graph/history.txt")
            blob = bucket.blob(f"{id}/{session_name}/freqs.txt")
            if(not os.path.exists("show_graph")):
                os.mkdir("show_graph")
            blob.download_to_filename("show_graph/freqs.txt")

            weights = pd.read_csv('show_graph/history.txt', header=None)
            weights.columns = ['Time', 'Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']
            freq = pd.read_csv('show_graph/freqs.txt', header=None, names=['Time', 'Frequency'])

            output_file("plots.html")

            emotion_source = ColumnDataSource(data=dict(
                Time=weights['Time'],
                Anger=weights['Anger'],
                Fear=weights['Fear'],
                Happy=weights['Happy'],
                Neutral=weights['Neutral'],
                Sadness=weights['Sadness'],
            ))

            emotion_plot = figure(width=800, height=400, title="Emotion Percentages Over Time", x_axis_label='Time', y_axis_label='Emotion Percentages(Weights)')
            emotion_hover = HoverTool(tooltips=[
                ("Time", "@Time"),
                ("Anger", "@Anger{0.2f}"),
                ("Fear", "@Fear{0.2f}"),
                ("Happy", "@Happy{0.2f}"),
                ("Neutral", "@Neutral{0.2f}"),
                ("Sadness", "@Sadness{0.2f}"),
            ])
            emotion_plot.add_tools(emotion_hover)

            colors = ["orange", "red", "green", "blue", "purple"]
            for emotion, color in zip(['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness'], colors):
                emotion_plot.line('Time', emotion, source=emotion_source, color=color, legend_label=emotion)

            frequency_source = ColumnDataSource(data=dict(
                Time=freq['Time'],
                Frequency=freq['Frequency'],
            ))

            frequency_plot = figure(width=800, height=400, title="Frequency Over Time", x_axis_label='Time', y_axis_label='Frequency')
            frequency_hover = HoverTool(tooltips=[
                ("Time", "@Time"),
                ("Frequency", "@Frequency"),
            ])

            frequency_plot.add_tools(frequency_hover)

            frequency_plot.line('Time', 'Frequency', source=frequency_source, color="black", legend_label='Frequency')

            script, div = components(emotion_plot)
            sc, di = components(frequency_plot)

            return render_template('see_graph.html',flask_login = flask_login, script=script, div=div, sc=sc, di=di, user_details = user_details, logged="yes")
        except Exception as e:
            return render_template('see_graph.html',flask_login = flask_login, script="", div="", sc="", di="", user_details = user_details, logged="yes")

@app.route('/faq', methods=["POST", "GET"])
def faq():
    if len(user_details):
        return render_template("faq.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
    else:
        return render_template("faq.html",flask_login = flask_login, logged_in="no", user_details = user_details)

@app.route('/video_feed', methods=["POST", "GET"])
def video_feed():
    start_time = time.time()
    if capture[0]:
        return Response(webcam_emotion_detection(start_time=start_time), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return None
    
@app.route('/approach', methods=["POST", "GET"])
def approach():
    if len(user_details):
        return render_template("approach.html",flask_login = flask_login, logged_in="yes", user_details = user_details)
    else:
        return render_template("approach.html",flask_login = flask_login, logged_in="no", user_details = user_details)

@app.route('/contact_mail', methods=["POST","GET"])
def contact_mail():
    if request.method == 'POST':
        result = request.form
        a = dict(result)
        print(a)
        html = f"""Dear Admin Team,
<br><br>
We have received a new submission from the "Reach Out to Us" form on our website.<br> Below are the details:
<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;<b>User Information</b> : <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>- Name :</b> {a["fname"]} {a["lname"]}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>- Email Id :</b> {a["email"]}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>- Location :</b> {a["location"]}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>- Phone Number :</b> {a["phone_no"]}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>- Contact Choice :</b> {a["choice"]}<br>
    <br><br>
    &nbsp;&nbsp;&nbsp;&nbsp;<b>Message Details:</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>- Message :</b> {a["message"]}
    <br><br>
Please review the submission and take the necessary actions to address the user’s request.
<br><br>
Thank you,
The SonicSerenity Team
"""
        recipient = "sonicserenity.therapy@gmail.com"
        subject = "New Contact Us Form Submission"
        msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient])
        msg.html = html
        try:
            mail.send(msg)
        except Exception as e:
            print(e)
        if len(user_details):
            return render_template("contact_us.html",flask_login = flask_login, logged_in="yes", user_details = user_details, alert_msg = "yes")
        else:
            return render_template("contact_us.html",flask_login = flask_login, logged_in="no", user_details = user_details, alert_msg = "no")

if __name__ == '__main__':
    app.run(debug = True,  host='0.0.0.0')
