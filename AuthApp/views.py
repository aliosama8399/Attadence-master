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
import numpy as np
from django.conf import settings
from django.utils import timezone
from django.http import HttpResponse, FileResponse


status = False
# Suppress debug logs
logging.getLogger('insightface').setLevel(logging.ERROR)

# Initialize the FaceAnalysis model
providers = ['AzureExecutionProvider']
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

# Function to save unique names to CSV
def save_unique_names_to_csv(names_set, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Recognized Faces"])  # Write header
        for name in names_set:
            csv_writer.writerow([name])

@login_required
def index(request):
    if request.user.is_authenticated:
        user = request.user
        subjects = Subject.objects.all()
        halls = Hall.objects.all()

        context = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'subjects': subjects,
            'halls': halls
        }
        return render(request, 'AuthApp/home.html', context)
    else:
        return redirect('login')

def user_login(request):
    load_training_images(request)
    if request.method == 'POST':
        code = request.POST['code']
        password = request.POST['password']
        user = authenticate(request, username=code, password=password)

        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            messages.error(request, 'Invalid credentials')
    return render(request, 'AuthApp/login.html')

@login_required
def user_logout(request):
    logout(request)
    return redirect('login')

@login_required
def takeAttendence(request):
    if request.method == 'POST':
        try:
            uploaded_images = request.FILES.getlist('images')
            if not uploaded_images:
                return JsonResponse({"status": "failed", "message": "No images uploaded."})

            recognized_names = []  # List to store recognized names, allowing duplicates for "Unknown"
            unique_recognized_names = set()  # Set to keep track of unique recognized names

            for uploaded_image in uploaded_images:
                image_array = np.frombuffer(uploaded_image.read(), np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                faces = app.get(img)

                for face in faces:
                    if face.embedding is None:
                        continue

                    embedding = normalize_embedding(face.embedding)
                    name = "Unknown"
                    min_dist = float("inf")

                    for known_embedding, known_name in zip(known_face_embeddings, known_face_names):
                        dist = np.linalg.norm(embedding - known_embedding)
                        if dist < min_dist:
                            min_dist = dist
                            name = known_name if dist < 1.2 else "Unknown"

                    # Add recognized names
                    if name == "Unknown":
                        recognized_names.append(name)
                    elif name not in unique_recognized_names:
                        recognized_names.append(name)
                        unique_recognized_names.add(name)

            context = {
                'recognized_names': recognized_names,
            }
            return render(request, 'AuthApp/modify_recognized_faces.html', context)

        except Exception as e:
            return JsonResponse({"status": "failed", "message": f"Error recognizing faces: {e}"})

    return JsonResponse({"status": "failed", "message": "Invalid request method."})


@login_required
def save_modified_names(request):
    if request.method == 'POST':
        # Get modified names from the form submission
        modified_names = request.POST.getlist('modified_names')  # List that allows duplicates

        # Handle additional new names if entered by the user
        new_names = request.POST.get('new_names')
        if new_names:
            # Split and strip to handle multiple names entered in the 'new_names' field
            new_names_list = [name.strip() for name in new_names.split(',') if name.strip()]
            modified_names.extend(new_names_list)  # Extend list with new names, including duplicates of "Unknown"

        # Generate CSV filename with timestamp for the current user
        username = request.user.username
        current_date_time = timezone.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f"{username}_{current_date_time}.csv"
        csv_file_path = os.path.join(settings.BASE_DIR, csv_filename)

        # Save all modified names, allowing duplicates of "Unknown"
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Recognized Faces"])  # Header row
            for name in modified_names:
                csv_writer.writerow([name])  # Save each name as-is

        # Serve the CSV file as a downloadable response
        response = FileResponse(open(csv_file_path, 'rb'), as_attachment=True, filename=csv_filename)
        return response

    return JsonResponse({"status": "failed", "message": "Invalid request method."})