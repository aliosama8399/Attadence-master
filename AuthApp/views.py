from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import Subject, Hall
from django.views.decorators.csrf import csrf_exempt
from insightface.app import FaceAnalysis
import logging
import os
import cv2
import csv
import time
import numpy as np
from django.conf import settings

status = False
# Suppress debug logs
logging.getLogger('insightface').setLevel(logging.ERROR)

# Initialize the FaceAnalysis model
providers = ['CUDAExecutionProvider']
app = FaceAnalysis(providers=providers)
app.prepare(ctx_id=0, det_size=(224, 224))

# Store known face embeddings and names globally
known_face_embeddings = []
known_face_names = []

# Function to normalize embeddings
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# Background function to load training images
def load_training_images_in_background():
    global status
    training_folder = os.path.join(settings.BASE_DIR, 'training_images')

    if not os.path.exists(training_folder):
        logging.error("Training folder not found.")
        return

    try:
        if not status:
            for filename in os.listdir(training_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(training_folder, filename)
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    faces = app.get(img)
                    if faces:
                        for face in faces:
                            if face.embedding is not None:
                                normalized_embedding = normalize_embedding(face.embedding)
                                known_face_embeddings.append(normalized_embedding)
                                known_face_names.append(os.path.splitext(filename)[0])
        status = True
        logging.info("Training images loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading training images: {e}")

# Function for HTTP request to trigger loading images (optional)
@csrf_exempt
def load_training_images(request):
    try:
        global status
        if not status:
            load_training_images_in_background()
            return JsonResponse({"status": "success", "message": "Training images loaded successfully."})
    except Exception as e:
        return JsonResponse({"status": "failed", "message": f"Error loading training images: {e}"})

# Other functions (index, login, logout, takeAttendence) remain the same
# Function to save unique names to CSV
def save_unique_names_to_csv(names, csv_file_path):
    try:
        # Check if the file exists and load existing names to prevent duplication
        existing_names = set()
        # if os.path.exists(csv_file_path):
        #     with open(csv_file_path, mode='w') as file:
        #         reader = csv.reader(file)
        #         for row in reader:
        #             existing_names.add(row[0])  # Add the existing names to the set

        # Open the CSV file in append mode and write only unique names
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for name in names:
                if name not in existing_names:
                    writer.writerow([name])
                    existing_names.add(name)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

@login_required
def index(request):
    if request.user.is_authenticated:
        user = request.user
        subjects = Subject.objects.all()
        halls = Hall.objects.all()
        load_training_images(request)
        # print(subjects,halls);
        context = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'subjects': subjects,
            'halls': halls
        }
        return render(request, 'AuthApp/home.html',context)
    else:
        # print('hello world')
        return redirect('login')

def user_login(request):
    if request.method == 'POST':
        code = request.POST['code']
        password = request.POST['password']
        print(code,password)
        user = authenticate(request, username=code, password=password)

        if user is not None:
            login(request, user)
            return redirect('/')  # redirect to home or another page
        else:
            messages.error(request, 'Invalid credentials')

    # print('hello world')
    # status = load_training_images(request)
    # if status:
    return render(request, 'AuthApp/login.html')
    


@login_required
def user_logout(request):
    logout(request)
    return redirect('login')


@login_required
def takeAttendence(request):
        if request.method == 'POST':
            video_capture = None
            try:
                video_capture = cv2.VideoCapture(0)  # Access the default camera (0)
                if not video_capture.isOpened():
                    return JsonResponse({"status": "failed", "message": "Failed to access the camera."})

                start_time = time.time()
                recognized_names = set()  # Set to store unique recognized names

                # Capture images for 1 minute (60 seconds)
                while (time.time() - start_time) < 60:
                    ret, frame = video_capture.read()
                    if not ret:
                        return JsonResponse({"status": "failed", "message": "Failed to capture image from camera."})

                    faces = app.get(frame)

                    for face in faces:
                        if face.embedding is None:
                            continue  # Skip if embedding is None

                        embedding = normalize_embedding(face.embedding)
                        name = "Unknown"
                        min_dist = float("inf")

                        # Compare the face with known faces
                        for known_embedding, known_name in zip(known_face_embeddings, known_face_names):
                            dist = np.linalg.norm(embedding - known_embedding)
                            if dist < min_dist:
                                min_dist = dist
                                name = known_name if dist < 1.2 else "Unknown"

                        # Only add the name if it's not "Unknown" and it's not already in the set
                        if name != "Unknown":
                            recognized_names.add(name)

                # Save unique recognized names to CSV
                csv_file_path = os.path.join(settings.BASE_DIR, 'recognized_faces.csv')
                save_unique_names_to_csv(recognized_names, csv_file_path)

                return redirect('/')  # redirect to home or another page
                # return JsonResponse({"status": "success", "message": f"Unique recognized names saved to {csv_file_path}"})
            except Exception as e:
                return JsonResponse({"status": "failed", "message": f"Error recognizing faces: {e}"})
            finally:
                if video_capture:
                    video_capture.release()
                cv2.destroyAllWindows()

        return JsonResponse({"status": "failed", "message": "Invalid request method."})
