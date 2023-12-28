import math
import glob
import os
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
import shutil
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import warnings
import cv2
import random
import datetime
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import imutils
from colorama import init, Fore
from tkinter import *
import getpass
import requests
import copy
import argparse
import cv2 as cv
from tkinter import messagebox
from clint.textui import progress
from zipfile import ZipFile
from requests.structures import CaseInsensitiveDict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
init()
check_file = os.path.isfile("data/login/login.py")
if check_file == False:
    print(f"{Fore.YELLOW}Granting API Access")
    GH_PREFIX = "https://raw.githubusercontent.com"
    ORG = "ParadoxAI-Co"
    REPO = "login_data"
    BRANCH = "main"
    FILE = "data.prx"
    URL = GH_PREFIX + "/" + ORG + "/" + REPO + "/" + BRANCH + "/" + FILE
# Headers setup
    headers = CaseInsensitiveDict()
    headers["Authorization"] = "token " + "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"
# Execute and view status
    print(f"{Fore.GREEN}API Access Granted")
    print(f"{Fore.YELLOW}Downloading Program Data")
    resp = requests.get(URL, headers=headers)
    if resp.status_code == 200:
        path = 'data.prx'
        with open(path, 'wb') as f:
            total_length = int(resp.headers.get('content-length'))
            for chunk in progress.bar(resp.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()
    else:
        print(f"{Fore.RED}API access token is not in control anymore, please download the lasteset version or if you have the lastest version please message owner!")
        exit()
    with ZipFile("data.prx") as z:
        z.extractall()
    time.sleep(2)
    os.remove("data.prx")
    print(f"{Fore.GREEN}done")
os.system("python data/login/login.py")
def ccam():
    global camcapi
    cm = input("choose your camera mode(ip/internal) ==>> ")
    if cm == "ip":
        camcapi = input("write your ip ==>> ")
        camcapi = str(camcapi)
    elif cm == "internal":
        camcapi = input("write your camera number(write 0 if you don't know what to choose, 0 is default) ==>> ")
        camcapi = int(camcapi)
ccam()
def recognize(path, model_path):
    warnings.filterwarnings("ignore")
    def getFaceBox(net, frame, conf_threshold=0.7):
        warnings.filterwarnings("ignore")
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes
    faceProto = model_path + "/opencv_face_detector.pbtxt"
    faceModel = model_path + "/opencv_face_detector_uint8.pb"
    ageProto = model_path + "/age_deploy.prototxt"
    ageModel = model_path + "/age_net.caffemodel"
    genderProto = model_path + "/gender_deploy.prototxt"
    genderModel = model_path + "/gender_net.caffemodel"
    warnings.filterwarnings("ignore")
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    warnings.filterwarnings("ignore")
    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    warnings.filterwarnings("ignore")
    # Open a video file or an image file or a camera stream
    warnings.filterwarnings("ignore")
    cap = cv.VideoCapture(path)
    warnings.filterwarnings("ignore")
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            warnings.filterwarnings("ignore")
        for bbox in bboxes:
            warnings.filterwarnings("ignore")
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            warnings.filterwarnings("ignore")
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            warnings.filterwarnings("ignore")
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            warnings.filterwarnings("ignore")
            agePreds = ageNet.fowrward()
            age = ageList[agePreds[0].argmax()]
            print(f"{Fore.YELLOW}Gender : {gender}")
            print(f"{Fore.YELLOW}Age : {age}")
            print(f"{Fore.BLUE}--------------------------------------------------------------------------------------")
            label = "{},{}".format(gender, age)
            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
def fr():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    root2.withdraw()
    warnings.filterwarnings("ignore")
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
    warnings.filterwarnings("ignore")
    def submit():
        root3.withdraw()
        def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
            X = []
            y = []
            # Loop through each person in the training set
            for class_dir in os.listdir(train_dir):
                if not os.path.isdir(os.path.join(train_dir, class_dir)):
                    continue
                # Loop through each training image for the current person
                for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                    image = face_recognition.load_image_file(img_path)
                    face_bounding_boxes = face_recognition.face_locations(image)
                    if len(face_bounding_boxes) != 1:
                        # If there are no people (or too many people) in a training image, skip the image.
                        if verbose:
                            print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                    else:
                        # Add face encoding for current image to the training set
                        X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                        y.append(class_dir)
            # Determine how many neighbors to use for weighting in the KNN classifier
            if n_neighbors is None:
                n_neighbors = int(round(math.sqrt(len(X))))
                if verbose:
                    print("Chose n_neighbors automatically:", n_neighbors)
            # Create and train the KNN classifier
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
            knn_clf.fit(X, y)
            # Save the trained KNN classifier
            if model_save_path is not None:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f)
            return knn_clf
        def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
            if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
            # Load a trained KNN model (if one was passed in)
            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)
            global X_face_locations
            X_face_locations = face_recognition.face_locations(X_frame)
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
            # Find encodings for faces in the test image
            faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        def show_prediction_labels_on_image(frame, predictions):
            from PIL import Image, ImageDraw
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            for name, (top, right, bottom, left) in predictions:
                # enlarge the predictions for the full sized image.
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                key = cv2.waitKey(1)
                if ord('s') == key:
                    cropi = frame[top:bottom, left:right]
                    pi = Image.fromarray(cropi)
                    pi.show()
                    pi.save("database/saved faces/unknown face {}.jpg".format(random.random()))
                    n = input(f"{Fore.GREEN}Enter face name ==>> ")
                    print("face is encoded and you can see the face name when you restart the program[face name: {}]".format(n))
                    os.mkdir("database/recognized_faces/train/{}".format(n))
                    pi.save("database/recognized_faces/train/{}/{} {}.jpg".format(n, n, random.random()))
                elif ord('r') == key:
                    print(f"{Fore.YELLOW}[INFO] Getting requirements ready to retrain all models, this may take some minutes if you have 99+ trained faces pictures")
                    classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
                    print(f"{Fore.GREEN}All faces got Retrained!")
                elif ord('a') == key:
                    mod = os.path.isfile("data/ag/ag_models/gender_net.caffemodel")
                    mod2 = os.path.isfile("data/ag/ag_models/age_net.caffemodel")
                    if mod == False and mod2 == False:
                        moi = messagebox.askyesno("Age & Gender Recognition", "you have not installed age & gender recognition models, do you want to download them right now?")
                        if moi == True:
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/gender_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/age_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                    elif mod == False and mod2 == True:
                        moi = messagebox.askyesno("Age & Gender Recognition", "one of age & gender recognition models are missing from your program, do you want to download them right now?")
                        if moi == True:
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/gender_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                    elif mod == True and mod2 == False:
                        moi = messagebox.askyesno("Age & Gender Recognition", "one of age & gender recognition models are missing from your program, do you want to download them right now?")
                        if moi == True:
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/age_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                    else:
                        os.system("python data/ag/ag.prx")
                        current_time = datetime.datetime.now()
                        with open('data/ag/{}{}.prx'.format(current_time.day, current_time.hour), 'r') as f:
                            lines = f.readlines()
                            print(lines)
                elif key == ord('e'):
                    current_time = datetime.datetime.now()
                    cropi = frame[top - 20:bottom + 20, left - 20:right + 20]
                    pe = Image.fromarray(cropi)
                    pe.save("data/ed/{}{}.jpg".format(current_time.day, current_time.hour))
                    os.system("python data/ed/ed.prx")
                    with open('data/ed/{}{}.prx'.format(current_time.day, current_time.hour), 'r') as f:
                        lines = []
                        for line in f:
                            lines.append(line)
                            print(line)
                elif key == ord('o'):
                    os.system("python data/recog/recog.py {}".format(name))
                warnings.filterwarnings("ignore")
                # Draw a box around the face using the Pillow module
                if name != "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
                elif name == "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                warnings.filterwarnings("ignore")
                # There's a bug in Pillow where it blows up with non-UTF-8 text
                # when using the default bitmap font
                warnings.filterwarnings("ignore")
                # Draw a label with a name below the face
                text_width, text_height = draw.textsize(name)
                warnings.filterwarnings("ignore")
                if name != "unknown":
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
                elif name == "unknown":
                    draw.text((10, 50), 'warning!', fill=(0, 0, 255))
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                name = name.encode("UTF-8")
                nof = len(X_face_locations)
                if nof == None or nof == 0:
                    draw.text((10, 30), 'number of faces: 0', fill=(0, 255, 0))
                else:
                    draw.text((10, 30), 'number of faces:' + str(nof), fill=(0, 255, 0))
                warnings.filterwarnings("ignore")
                draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 255, 0))
                warnings.filterwarnings("ignore")
            # Remove the drawing library from memory as per the Pillow docs.
            del draw
            # Save image in open-cv format to be able to show it.
            opencvimage = np.array(pil_image)
            return opencvimage
        if __name__ == "__main__":
            train_a = messagebox.askyesno("paradox master face recognition", "do you want to retrain all faces again?(click yes if you have add new faces)")
            if train_a == True:
                print(f"{Fore.YELLOW}[INFO] Getting requirements ready to train all models, this may take some minutes if you have 99+ trained faces pictures")
                classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
            print(f"{Fore.GREEN}Training complete!")
            print(f"{Fore.WHITE} ")
            # process one frame in every 30 frames for speed
            process_this_frame = 29
            cap = cv2.VideoCapture(camcapi)
            while 1:
                ret, frame = cap.read()
                if ret:
                    # Different resizing options can be chosen based on desired program runtime.
                    # Image resizing for more stable streaming
                    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    process_this_frame = process_this_frame + 1
                    if process_this_frame % 30 == 0:
                        predictions = predict(img, model_path="data/prx_models/trained_faces.prx")
                    frame = show_prediction_labels_on_image(frame, predictions)
                    cv2.imshow('face recognition', frame)
                    if 27 == cv2.waitKey(1):
                        root3.deiconify()
                        cap.release()
                        cv2.destroyAllWindows()
                        break
    def submit2():
        root3.withdraw()
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        warnings.filterwarnings("ignore")
        def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
            X = []
            y = []
            # Loop through each person in the training set
            for class_dir in os.listdir(train_dir):
                if not os.path.isdir(os.path.join(train_dir, class_dir)):
                    continue
                # Loop through each training image for the current person
                for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                    image = face_recognition.load_image_file(img_path)
                    face_bounding_boxes = face_recognition.face_locations(image)
                    if len(face_bounding_boxes) != 1:
                        # If there are no people (or too many people) in a training image, skip the image.
                        if verbose:
                            print("[INFO] Image {} not suitable for training: {}".format(img_path, "[INFO] Didn't find a face" if len(face_bounding_boxes) < 1 else "[INFO] Found more than one face"))
                    else:
                        # Add face encoding for current image to the training set
                        X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                        y.append(class_dir)
            # Determine how many neighbors to use for weighting in the KNN classifier
            if n_neighbors is None:
                n_neighbors = int(round(math.sqrt(len(X))))
                if verbose:
                    print("[INFO] Chose n_neighbors automatically:", n_neighbors)
            # Create and train the KNN classifier
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
            knn_clf.fit(X, y)
            # Save the trained KNN classifier
            if model_save_path is not None:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f)
            return knn_clf
        def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
            if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
                raise Exception("[INFO] Invalid image path: {}".format(X_img_path))
            if knn_clf is None and model_path is None:
                raise Exception("[INFO] Must supply knn classifier either thourgh knn_clf or model_path")
            # Load a trained KNN model (if one was passed in)
            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)
            # Load image file and find face locations
            global X_img
            X_img = face_recognition.load_image_file(X_img_path)
            global X_face_locations
            X_face_locations = face_recognition.face_locations(X_img)
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
            # Find encodings for faces in the compare iamge
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
            # Use the KNN model to find the best matches for the compare face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        def show_prediction_labels_on_image(img_path, predictions):
            from PIL import Image, ImageDraw
            pil_image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_image)
            for name, (top, right, bottom, left) in predictions:
                # Draw a box around the face using the Pillow module
                if name != "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                if name == "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
                # There's a bug in Pillow where it blows up with non-UTF-8 text
                # when using the default bitmap font
                text_width, text_height = draw.textsize(name)
                if name != "unknown":
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                if name == "unknown":
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
                name = name.encode("UTF-8")
                draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 255, 0))
                nof = len(X_face_locations)
                draw.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
                warnings.filterwarnings("ignore")
            # Remove the drawing library from memory as per the Pillow docs
            del draw
            # Display the resulting image
            print(f"{Fore.GREEN}[INFO] founded {nof} face(s) in this photograph.")
            pil_image.show()
        if __name__ == "__main__":
            train_a = messagebox.askyesno("paradox master face recognition", "do you want to retrain all faces again?(click yes if you have add new faces)")
            if train_a == True:
                print(f"{Fore.YELLOW}[INFO] Getting requirements ready to train all models, this may take some minutes if you have 99+ trained faces pictures")
                classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
            print(f"{Fore.GREEN}[INFO] Training complete!")
            print(f"{Fore.BLUE}----------------------------------------------------------------------")
            print(f"{Fore.WHITE} ")
            warnings.filterwarnings("ignore")
            # STEP 2: Using the trained classifier, make predictions for unknown images
        from tkinter import filedialog as fd
        from tkinter.messagebox import showinfo
        root=tk.Tk()
        def disable_event():
            root3.deiconify()
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", disable_event)
        root.title('Choose Image File')
        root.geometry("440x275")
        name_var=tk.StringVar()
        def ret():
            print(f"{Fore.YELLOW}[INFO] Getting requirements ready to retrain all models, this may take some minutes if you have 99+ trained faces pictures")
            classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
            print(f"{Fore.GREEN}all faces got retrained!")
        def submit2():
            image_file = filename
            full_file_path = filename
            root.title(filename)
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(full_file_path, model_path="data/prx_models/trained_faces.prx")
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print(f"{Fore.BLUE}----------------------------------------------------------------------")
                print(f"{Fore.GREEN}[FACE FOUND] Found {name} at left: {left}, top: {top}, right: {right}, bottom: {bottom} In {image_file}")
                warnings.filterwarnings("ignore")
            # Display results overlaid on an image
            show_prediction_labels_on_image(image_file, predictions)
            warnings.filterwarnings("ignore")
        # Let's Create some buttons to help the user and make the program more optional & friendly for a beginner user!
        name_var.set("")
        def submit3():
            global name
            name=name_var.get()
            root.title(filename)
            # Load the jpg file into a numpy array
            image = face_recognition.load_image_file("{}".format(filename))
            image2 = face_recognition.load_image_file("{}".format(filename))
            image3 = face_recognition.load_image_file("{}".format(filename))
            # Find all facial features in all the faces in the image
            face_landmarks_list = face_recognition.face_landmarks(image)
            face_landmarks_list2 = face_recognition.face_landmarks(image3)
            face_locations = face_recognition.face_locations(image)
            face_locations2 = face_recognition.face_locations(image2)
            print(f"{Fore.GREEN}[INFO] founded {len(face_landmarks_list)} face(s) in this photograph.")
            # Create a PIL imagedraw object so we can draw on the picture
            from PIL import Image, ImageDraw
            pil_image = Image.fromarray(image)
            pil_image4 = Image.fromarray(image2)
            pil_image5 = Image.fromarray(image3)
            from PIL import Image, ImageDraw
            d = ImageDraw.Draw(pil_image)
            d2 = ImageDraw.Draw(pil_image4)
            d3 = ImageDraw.Draw(pil_image5)
            for (top, right, bottom, left) in face_locations:
                d.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                print(f"{Fore.BLUE}[INFO] A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}, saving face")
            for (top, right, bottom, left) in face_locations2:
                d2.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                nof = len(face_landmarks_list)
                d2.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
                face_image = image[top:bottom, left:right]
                from PIL import Image, ImageDraw
                pil_image3 = Image.fromarray(face_image)
                pil_image3.show()
                pil_image3.save("database/saved faces/{} cropped face {}.jpg".format(name, random.random()))
            for face_landmarks in face_landmarks_list2:
                for facial_feature in face_landmarks.keys():
                    d3.line(face_landmarks[facial_feature], width=4)
                    d3.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            for face_landmarks in face_landmarks_list:
                # Print the location of each facial feature in this image
                for facial_feature in face_landmarks.keys():
                    print(f"{Fore.YELLOW}The {facial_feature} in this face has the following points: {face_landmarks[facial_feature]}")
                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], width=4)
                    d.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            # Show the picture
            pil_image.show()
            pil_image4.show()
            pil_image5.show()
            pil_image4.save("database/saved faces/{} full image(detection only) {}.jpg".format(name, random.random()))
            pil_image.save("database/saved faces/{} full image(biometrics + detection) {}.jpg".format(name, random.random()))
            pil_image5.save("database/saved faces/{} full image(biometrics only) {}.jpg".format(name, random.random()))
        # lets do some definition for buttons that we just got created
        def submit4():
            global name
            name=name_var.get()
            root.title(filename)
            # Load the jpg file into a numpy array
            image = face_recognition.load_image_file("{}".format(filename))
            image2 = face_recognition.load_image_file("{}".format(filename))
            image3 = face_recognition.load_image_file("{}".format(filename))
            # Find all facial features in all the faces in the image
            face_landmarks_list = face_recognition.face_landmarks(image)
            face_landmarks_list2 = face_recognition.face_landmarks(image3)
            face_locations = face_recognition.face_locations(image)
            face_locations2 = face_recognition.face_locations(image2)
            print(f"{Fore.GREEN}[INFO] founded {len(face_landmarks_list)} face(s) in this photograph.")
            # Create a PIL imagedraw object so we can draw on the picture
            from PIL import Image, ImageDraw
            pil_image = Image.fromarray(image)
            pil_image4 = Image.fromarray(image2)
            pil_image5 = Image.fromarray(image3)
            from PIL import Image, ImageDraw
            d = ImageDraw.Draw(pil_image)
            d2 = ImageDraw.Draw(pil_image4)
            d3 = ImageDraw.Draw(pil_image5)
            for (top, right, bottom, left) in face_locations:
                d.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                nof = len(face_landmarks_list)
                d.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
                print(f"{Fore.BLUE}[INFO] A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}, saving face")
            for (top, right, bottom, left) in face_locations2:
                d2.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                d2.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
                face_image = image[top:bottom, left:right]
                from PIL import Image, ImageDraw
                pil_image3 = Image.fromarray(face_image)
                pil_image3.show()
            for face_landmarks in face_landmarks_list2:
                for facial_feature in face_landmarks.keys():
                    d3.line(face_landmarks[facial_feature], width=4)
                    d3.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            for face_landmarks in face_landmarks_list:
                # Print the location of each facial feature in this image
                for facial_feature in face_landmarks.keys():
                    print(f"{Fore.YELLOW}The {facial_feature} in this face has the following points: {face_landmarks[facial_feature]}")
                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], width=4)
            # Show the picture
            pil_image.show()
            pil_image4.show()
            pil_image5.show()
        def ag():
            mod = os.path.isfile("data/ag/ag_models/gender_net.caffemodel")
            mod2 = os.path.isfile("data/ag/ag_models/age_net.caffemodel")
            if mod == False and mod2 == False:
                moi = messagebox.askyesno("Age & Gender Recognition", "you have not installed age & gender recognition models, do you want to download them right now?")
                if moi == True:
                    url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
                    r = requests.get(url, stream=True)
                    path = 'data/ag/ag_models/gender_net.caffemodel'
                    with open(path, 'wb') as f:
                        total_length = int(r.headers.get('content-length'))
                        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                            if chunk:
                                f.write(chunk)
                                f.flush()
                    url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
                    r = requests.get(url, stream=True)
                    path = 'data/ag/ag_models/age_net.caffemodel'
                    with open(path, 'wb') as f:
                        total_length = int(r.headers.get('content-length'))
                        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                            if chunk:
                                f.write(chunk)
                                f.flush()
            elif mod == False and mod2 == True:
                moi = messagebox.askyesno("Age & Gender Recognition", "one of age & gender recognition models are missing from your program, do you want to download them right now?")
                if moi == True:
                    url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
                    r = requests.get(url, stream=True)
                    path = 'data/ag/ag_models/gender_net.caffemodel'
                    with open(path, 'wb') as f:
                        total_length = int(r.headers.get('content-length'))
                        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                            if chunk:
                                f.write(chunk)
                                f.flush()
            elif mod == True and mod2 == False:
                moi = messagebox.askyesno("Age & Gender Recognition", "one of age & gender recognition models are missing from your program, do you want to download them right now?")
                if moi == True:
                    url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
                    r = requests.get(url, stream=True)
                    path = 'data/ag/ag_models/age_net.caffemodel'
                    with open(path, 'wb') as f:
                        total_length = int(r.headers.get('content-length'))
                        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                            if chunk:
                                f.write(chunk)
                                f.flush()
            else:
                root.title(filename)
                warnings.filterwarnings("ignore")
                recognize(path=filename, model_path="data/ag/ag_models")
                warnings.filterwarnings("ignore")
        def ed():
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
            from fer import FER
            root.title(filename)
            warnings.filterwarnings("ignore")
            emotion_detector = FER()
            warnings.filterwarnings("ignore")
            test_img = cv2.imread(filename)
            warnings.filterwarnings("ignore")
            analysis = emotion_detector.detect_emotions(test_img)
            warnings.filterwarnings("ignore")
            dominant_emotion, emotion_score = emotion_detector.top_emotion(test_img)
            warnings.filterwarnings("ignore")
            if dominant_emotion != None:
                warnings.filterwarnings("ignore")
                print(f"{Fore.YELLOW}result: {dominant_emotion} \n more information: \n {analysis}")
                warnings.filterwarnings("ignore")
            else:
                warnings.filterwarnings("ignore")
                print("No Face Detected, Please Try Another with Picture Again")
                warnings.filterwarnings("ignore")
        def sf():
            mn = input("please write the name of the face(use underline instead of space) ==>> ")
            cmn = os.path.isfile("database/recognized_faces/train/{}".format(mn))
            if cmn == True:
                print(f"{Fore.RED}A face with this name({mn}) is already exist! please choose another name for the face")
            os.mkdir("database/recognized_faces/train/{}".format(mn))
            shutil.copy(filename, "database/recognized_faces/train/{}".format(mn))
            print(f"{Fore.GREEN}Face saved in database as {mn}(if you used space in the name, some functions for this face may not work)")
        # create the root window
        text1 = tk.Label(root, text = 'Choose your File', font = ('calibre',10,'bold'))
        def select_file():
            filetypes = (
                ('jpeg files', '*.jpg'),
                ('png files', '*.png')
            )
            global filename
            filename = fd.askopenfilename(
                title='Open a Image',
                initialdir='/',
                filetypes=filetypes)
        # open button
        open_button = ttk.Button(
            root,
            text='Open a Image',
            command=select_file
        )
        text1.pack()
        open_button.pack()
        text2 = ttk.Label(root, text = 'Choose your Operation', font = ('calibre',10,'bold'))
        name_entry = ttk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
        sub_btnn=ttk.Button(root,text = 'Recognize all Faces', command = submit2)
        sub_btn2=ttk.Button(root,text = 'find all biometrics on faces & save cropped faces', command = submit3)
        sub_btn3=ttk.Button(root,text = 'find all biometrics on faces', command = submit4)
        sub_btn4=ttk.Button(root,text = 'recognize age and gender', command = ag)
        sub_btn5=ttk.Button(root,text = 'recognize emotions', command = ed)
        sub_btn6=ttk.Button(root, text = 'save this face in database', command = sf)
        l=ttk.Label(root,text='------------------------------------------------------------------------------------------')
        ret=ttk.Button(root,text = 'retrain all faces', command = ret)
        text2.pack()
        sub_btnn.pack()
        sub_btn2.pack()
        sub_btn3.pack()
        sub_btn4.pack()
        sub_btn5.pack()
        sub_btn6.pack()
        l.pack()
        ret.pack()
        root.mainloop()
    def disable_event():
        root2.deiconify()
        root3.destroy()
    root3=tk.Tk()
    root3.protocol("WM_DELETE_WINDOW", disable_event)
    root3.title('Choose the Program')
    root3.geometry("240x110")
    root3.resizable(0, 0)
    name_var=tk.StringVar()
    text2 = tk.Label(root3, text = 'Choose the Program', font = ('calibre',10,'bold'))
    name_entry = tk.Entry(root3,textvariable = name_var, font=('calibre',10,'normal'))
    sub_btnn=ttk.Button(root3,text = 'face recognition from real time camera', command = submit)
    sub_btn2=ttk.Button(root3,text = 'face recognition from picture', command = submit2)
    text2.pack()
    sub_btnn.pack()
    sub_btn2.pack()
    root3.mainloop()
def fd():
    root2.withdraw()
    def submit():
        def main():
            import mediapipe as mp
            cap = cv.VideoCapture(camcapi)
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection= 1,
                min_detection_confidence= 0.5,
            )
            while 1:
                ret, image = cap.read()
                if not ret:
                    break
                
                global debug_image
                debug_image = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = face_detection.process(image)
                if results.detections is not None:
                    for detection in results.detections:
                        debug_image = draw_detection(debug_image, detection)
                cv.imshow("face detection", debug_image)
                key = cv.waitKey(1)
                if 27 == cv.waitKey(1):
                    break
            cap.release()
            cv.destroyAllWindows()
        def draw_detection(image, detection):
            cv.flip(image, 1)
            image_width, image_height = image.shape[1], image.shape[0]
            bbox = detection.location_data.relative_bounding_box
            bbox.xmin = int(bbox.xmin * image_width)
            bbox.ymin = int(bbox.ymin * image_height)
            bbox.width = int(bbox.width * image_width)
            bbox.height = int(bbox.height * image_height)
            cv.rectangle(image, (int(bbox.xmin), int(bbox.ymin)),
                         (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
                         (0, 0, 0), 2)
            xleft, ytop, xright, ybot  = int(bbox.xmin), int(bbox.ymin), int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)
            crop_img = image[ytop: ybot, xleft: xright]
            key = cv.waitKey(1)
            current_time = datetime.datetime.now()
            if key == ord('c'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                cv.imshow("cropped {}".format(random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            elif key == ord('s'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                print("[FACE SAVED] face saved in database/saved faces")
                cv.imwrite("cropped face {} {} {} {} {}.jpg".format(current_time.day, current_time.hour, current_time.minute, current_time.second, random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            return image
        if __name__ == '__main__':
            main()
    def submit2():
        def main():
            import mediapipe as mp
            cap = cv.VideoCapture(camcapi)
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection= 1,
                min_detection_confidence= 0.5,
            )
            while 1:
                ret, image = cap.read()
                if not ret:
                    break
                
                global debug_image
                debug_image = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = face_detection.process(image)
                if results.detections is not None:
                    for detection in results.detections:
                        debug_image = draw_detection(debug_image, detection)
                cv.imshow("face detection", debug_image)
                key = cv.waitKey(1)
                if 27 == cv.waitKey(1):
                    break
            cap.release()
            cv.destroyAllWindows()
        def draw_detection(image, detection):
            cv.flip(image, 1)
            image_width, image_height = image.shape[1], image.shape[0]
            bbox = detection.location_data.relative_bounding_box
            bbox.xmin = int(bbox.xmin * image_width)
            bbox.ymin = int(bbox.ymin * image_height)
            bbox.width = int(bbox.width * image_width)
            bbox.height = int(bbox.height * image_height)
            cv.rectangle(image, (int(bbox.xmin), int(bbox.ymin)),
                         (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
                         (0, 0, 0), -1)
            xleft, ytop, xright, ybot  = int(bbox.xmin), int(bbox.ymin), int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)
            crop_img = image[ytop: ybot, xleft: xright]
            key = cv.waitKey(1)
            current_time = datetime.datetime.now()
            if key == ord('c'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                cv.imshow("cropped {}".format(random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            elif key == ord('s'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                print("[FACE SAVED] face saved in database/saved faces")
                cv.imwrite("cropped face {} {} {} {} {}.jpg".format(current_time.day, current_time.hour, current_time.minute, current_time.second, random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            return image
        if __name__ == '__main__':
            main()
    def submit3():
        def main():
            import mediapipe as mp
            init()
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh
            # For webcam input:
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            cap = cv2.VideoCapture(camcapi)
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
              while cap.isOpened():
                success, image = cap.read()
                if not success:
                  print("Ignoring empty camera frame.")
                  # If loading a video, use 'break' instead of 'continue'.
                  continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_mesh.process(image)
                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                if results.multi_face_landmarks:
                  for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                cv2.imshow('Face Mesh', image)
                if cv2.waitKey(5) & 0xFF == 27:
                  break
            cap.release()
            cv2.destroyAllWindows()
        if __name__ == '__main__':
            main()
    root=tk.Tk()
    def disable_event():
        root2.deiconify()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", disable_event)
    root.title('Choose the Program')
    root.geometry("200x120")
    root.resizable(0, 0)
    name_var=tk.StringVar()
    text2 = tk.Label(root, text = 'Choose the Program', font = ('calibre',10,'bold'))
    name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
    sub_btnn=ttk.Button(root,text = 'face detection', command = submit)
    sub_btn2=ttk.Button(root,text = 'face hider', command = submit2)
    sub_btn3=ttk.Button(root,text = 'face mesh', command = submit3)
    text2.pack()
    sub_btnn.pack()
    sub_btn2.pack()
    sub_btn3.pack()
    root.mainloop()
def cs():
    os.system("cls")
def cc():
    for clean_up in glob.glob('data/ed/*.*'):
        if not clean_up.endswith('ed.prx'):    
            os.remove(clean_up)
    for clean_up in glob.glob('data/ag/*.*'):
        if not clean_up.endswith('ag.prx'):    
            os.remove(clean_up)
import tkinter as tk
root2=tk.Tk()
root2.title('Choose the Program')
root2.geometry("200x170")
root2.resizable(0, 0)
name_var=tk.StringVar()
text2 = tk.Label(root2, text = 'Choose the Program', font = ('calibre',10,'bold'))
name_entry = tk.Entry(root2,textvariable = name_var, font=('calibre',10,'normal'))
sub_btnn=ttk.Button(root2,text = 'Face Recognition', command = fr)
sub_btn2=ttk.Button(root2,text = 'Face Detection', command = fd)
recam=ttk.Button(root2,text = 'Choose Another Camera Input', command = ccam)
csb=ttk.Button(root2, text = "Clear Shell", command = cs)
ccb=ttk.Button(root2, text = "Clear Cache", command = cc)
text2.pack()
sub_btnn.pack()
sub_btn2.pack()
recam.pack()
csb.pack()
ccb.pack()
root2.mainloop()