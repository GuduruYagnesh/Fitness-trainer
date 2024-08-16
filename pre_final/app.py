from flask import Flask, Response, jsonify, render_template
import requests
import cv2
import mediapipe as mp
import threading
import math
import mysql.connector as conn 
import numpy as np
from utils import calculate_angle
from utils import provide_feedback
from utils import *
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io
import seaborn as sns
from matplotlib.dates import DateFormatter

app = Flask(__name__)
camera = None

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mydb = conn.connect(host = 'localhost',user = 'root' ,passwd = "yagnesh@123" )
cursor = mydb.cursor()
cursor.execute('use exercise_data')


laptop_resolution = (2560, 1440)
angle = 0

# Initialize correct and incorrect counters
correct_counter = 0
incorrect_counter = 0

def ArmCurlExercise():
    global camera, angle, correct_counter, incorrect_counter

    # Curl counter variables
    counter = 0
    stage = None

    # Set arm curl angle thresholds
    arm_curl_threshold = (20, 175)

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Offset for text position (to place the text slightly above the elbow)
                text_offset = (10, -10)

                # Calculate the position to display the angle
                text_position = (int(elbow[0] * image.shape[1]) + text_offset[0], int(elbow[1] * image.shape[0]) + text_offset[1])

                # Visualize angle
                cv2.putText(image, str(angle), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


                feedback = provide_feedback(angle, arm_curl_threshold)

                previous_stage = stage

                # Curl counter logic
                if angle > 150 and angle < 175:
                    stage = "down"
                    previous_stage = stage
                elif angle > 20 and stage == 'down' and angle < 50:
                    stage = "up"
                    prev_stage = stage
                    # Check if angle was within range during both up and down phases
                    if angle >= arm_curl_threshold[0] and angle <= arm_curl_threshold[1] and feedback=="Angle within range. Good job!" and stage=="up":
                        # previous_stage=stage
                        counter += 1
                        correct_counter += 1
                        print(f"Rep {counter}: Correct")
                        previous_stage=stage

                    elif prev_stage == "up" and feedback=="Angle too small. Extend your arm further.":
                        prev_stage="down"
                        incorrect_counter += 1
                        print(f"Rep {counter}: Incorrect (angle below threshold)")
                    # else:
                    #     if feedback!="Angle within range. Good job!":
                    #         incorrect_counter += 1
                    #         print(f"Rep {counter}: Incorrect (angle out of range)")
                if stage == "down"  and angle > arm_curl_threshold[1]:
                    stage = "up"
                    incorrect_counter += 1
                    print(f"Rep {counter}: Incorrect (angle above threshold)")
                    
                if prev_stage == "up"  and angle < arm_curl_threshold[0]:
                    prev_stage = "down"
                    incorrect_counter += 1
                    print(f"Rep {counter}: Incorrect (angle below threshold)")

                # Render curl counter, status box, rep data, stage data, and detections
                # (code remains largely the same as in the original responses)

            except:
                pass

           
            cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)

            cv2.putText(image, "Correct Exercises: " + str(correct_counter), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, "Incorrect Exercises: " + str(incorrect_counter), (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)
            
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            # Display feedback on frame
            cv2.putText(image, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        

            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)



            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break
        
    if camera is not None:
        camera.release()


# Variables for counting exercises
correct_counter_side_arm = 0
incorrect_counter_side_arm = 0

def SideArmExercise():
    global camera

    global correct_counter_side_arm
    global incorrect_counter_side_arm

    threshold_range = (10, 95)

    last_correct = False
    direction = None
    stage = None  # Start in down position for consistency
    counter = 0

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extracting landmark coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angles
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_body_angle = calculate_angle(left_ankle, left_hip, left_shoulder)
                right_body_angle = calculate_angle(right_ankle, right_hip, right_shoulder)

                prev_stage = stage

                if check_angles_for_posture_sidearm(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle) and check_for_upside_shoulder_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "down"

                elif stage == "down" and check_angles_for_posture_sidearm(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle) and check_for_downside_shoulder_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "up"

                    if check_for_threshold_sidearm(left_shoulder_angle, right_shoulder_angle):
                        counter += 1
                        correct_counter_side_arm += 1
                    else:
                        stage = "up"
                        incorrect_counter_side_arm += 1

                if stage == "down" and check_for_threshold_low_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "up"
                    incorrect_counter_side_arm += 1

                if stage == "up" and check_for_threshold_high_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "down"
                    incorrect_counter_side_arm += 1

                

                # ... (rest of the code for drawing landmarks, displaying angles, etc.)
                cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)
            
                cv2.putText(image, "Correct Exercises: " + str(correct_counter_side_arm), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, "Incorrect Exercises: " + str(incorrect_counter_side_arm), (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)

                # Visualize angles
                cv2.putText(image, "Left Shoulder Angle: " + str(round(left_shoulder_angle, 2)), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Shoulder Angle: " + str(round(right_shoulder_angle, 2)), (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Elbow Angle: " + str(round(left_elbow_angle, 2)), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Elbow Angle: " + str(round(right_elbow_angle, 2)), (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Body Angle: " + str(round(left_body_angle, 2)), (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Body Angle: " + str(round(right_body_angle, 2)), (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 0), 2, cv2.LINE_AA)
            

            except Exception as e:
                print(e)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
            
            
            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break

    if camera is not None:
        camera.release()


# Variables for counting exercises
correct_counter_up_arm = 0
incorrect_counter_up_arm = 0

def UpArmExercise():
    global camera,correct_counter_up_arm, incorrect_counter_up_arm

    last_correct = False
    direction = None
    stage = None  # Start in down position for consistency
    counter = 0

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calculate angle
                left_elbow_angle = calculate_angle_h_e(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle_h_e(right_shoulder,right_elbow,right_wrist)
                right_body_angle = calculate_angle_h_e(right_ankle,right_hip,right_shoulder)
                left_body_angle = calculate_angle_h_e(left_ankle,left_hip,left_shoulder)
                right_shoulder_angle = calculate_angle_shoulder(right_hip,right_shoulder,right_elbow)
                left_shoulder_angle = calculate_angle_shoulder(left_hip,left_shoulder,left_elbow)
                
                # Visualize angle
                #right_elbow_angle
                cv2.putText(image, str(right_elbow_angle), 
                            tuple(np.multiply(right_elbow, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #left_elbow_angle
                cv2.putText(image, str(left_elbow_angle), 
                            tuple(np.multiply(left_elbow, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #right_hip_angle
                cv2.putText(image, str(right_body_angle), 
                            tuple(np.multiply(right_hip, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #left_hip_angle
                cv2.putText(image, str(left_body_angle), 
                            tuple(np.multiply(left_hip, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
   
                prev_stage = stage

                if check_angles_for_posture_upraise(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle) and check_for_upside_shoulder_upraise(left_shoulder_angle, right_shoulder_angle):
                    stage = "down"

                elif stage == "down" and check_angles_for_posture_upraise(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle) and check_for_downside_shoulder_upraise(left_shoulder_angle, right_shoulder_angle):
                    stage = "up"

                    if check_for_threshold_upraise(left_shoulder_angle, right_shoulder_angle):
                        counter += 1
                        correct_counter_up_arm += 1
                    else:
                        stage = "up"
                        incorrect_counter_up_arm += 1

                if stage == "down" and check_for_threshold_low_upraise(left_shoulder_angle, right_shoulder_angle):
                    stage = "up"
                    incorrect_counter_up_arm += 1

                if stage == "up" and check_for_threshold_high_upraise(left_shoulder_angle, right_shoulder_angle):
                    stage = "down"
                    incorrect_counter_up_arm += 1

                # ... (rest of the code for drawing landmarks, displaying angles, etc.)
                cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)
            
                cv2.putText(image, "Correct Exercises: " + str(correct_counter_up_arm), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, "Incorrect Exercises: " + str(incorrect_counter_up_arm), (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)

                # Visualize angles
                cv2.putText(image, "Left Shoulder Angle: " + str(round(left_shoulder_angle, 2)), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Shoulder Angle: " + str(round(right_shoulder_angle, 2)), (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Elbow Angle: " + str(round(left_elbow_angle, 2)), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Elbow Angle: " + str(round(right_elbow_angle, 2)), (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Body Angle: " + str(round(left_body_angle, 2)), (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Body Angle: " + str(round(right_body_angle, 2)), (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
                
            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break

    if camera is not None:
        camera.release()


# Variables for counting exercises
elapsed_time_correct_wall_sit = 0
elapsed_time_incorrect_wall_sit = 0

def WallSitExercise():
    global camera, elapsed_time_correct_wall_sit, elapsed_time_incorrect_wall_sit

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    start_time = time.time()  # Start time for counting
    paused_time = 0  # Time when the timer was paused
    is_paused = False  # Flag to indicate if the timer is paused
    start_time_correct = time.time()  # Start time for correct pose
    start_time_incorrect = time.time()  # Start time for incorrect pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while camera.isOpened():

            ## read the camera frame
            success, frame = camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                # Calculate angle
                left_hip_angle = calc_angle(left_shoulder, left_hip, left_knee)
                right_hip_angle = calc_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calc_angle(left_hip, left_knee, left_heel)
                right_knee_angle = calc_angle(right_hip, right_knee, right_heel)

                # Check if angles are in the specified range for both hips and knees
                if not is_paused:
                    elapsed_time_correct_wall_sit = int(time.time() - start_time_correct)
                else:
                    elapsed_time_correct_wall_sit = int(paused_time - start_time_correct)

                if check_angles_in_range_for_wallsit(left_hip_angle, right_hip_angle, left_knee_angle,
                                                      right_knee_angle):
                    if is_paused:
                        # Adjust start time based on paused time
                        start_time_correct += (time.time() - paused_time)
                        is_paused = False
                else:
                    if not is_paused:
                        # Pause the timer
                        paused_time = time.time()
                        is_paused = True

                    # Calculate incorrect timing
                    elapsed_time_incorrect_wall_sit = int(time.time() - start_time_incorrect)

                # Display angle values
                cv2.putText(image, str(int(left_hip_angle)),
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(right_hip_angle)),
                            tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(left_knee_angle)),
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(right_knee_angle)),
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Check angles for knee health
                if check_angles_in_range_for_wallsit(left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle):
                    cv2.putText(image, "Good for knees", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 7)
                else:
                    cv2.putText(image, "Make angles to 90 degrees for knees", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                if check_angles_in_range_for_wallsit(left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle):
                    cv2.putText(image, "Good for knees", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 7)
                else:
                    cv2.putText(image, "Make angles to 90 degrees for knees", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            except:
                pass

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Display time spent correctly
            cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)
            cv2.putText(image, "Correct Time: " + str(elapsed_time_correct_wall_sit) + "s", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)

            # Display time spent incorrectly
            cv2.putText(image, "Incorrect Time: " + str(elapsed_time_incorrect_wall_sit) + "s", (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)

            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break

    if camera is not None:
        camera.release()



# Variables for counting exercises
elapsed_time_correct_plank = 0
elapsed_time_incorrect_plank = 0

def PlankExercise():
    global camera, elapsed_time_correct_plank, elapsed_time_incorrect_plank

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    start_time = time.time()  # Start time for counting
    paused_time = 0  # Time when the timer was paused
    is_paused = False  # Flag to indicate if the timer is paused
    start_time_correct = time.time()  # Start time for correct pose
    start_time_incorrect = time.time()  # Start time for incorrect pose

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.60, min_tracking_confidence=0.60) as pose:
        while camera.isOpened():
            success, frame = camera.read()
            
            if not success:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder,right_elbow,right_wrist)
                right_body_angle = calculate_angle(right_ankle,right_hip,right_shoulder)
                left_body_angle = calculate_angle(left_ankle,left_hip,left_shoulder)
                right_shoulder_angle = calculate_angle(right_hip,right_shoulder,right_elbow)
                left_shoulder_angle = calculate_angle(left_hip,left_shoulder,left_elbow)

                # Check if angles are in the specified range for both hips and knees
                if not is_paused:
                    elapsed_time_correct_plank = int(time.time() - start_time_correct)
                else:
                    elapsed_time_correct_plank = int(paused_time - start_time_correct)

                if check_angles_in_range_for_plank(left_elbow_angle, left_body_angle, left_shoulder_angle):
                    if is_paused:
                        # Adjust start time based on paused time
                        start_time_correct += (time.time() - paused_time)
                        is_paused = False
                else:
                    if not is_paused:
                        # Pause the timer
                        paused_time = time.time()
                        is_paused = True

                    # Calculate incorrect timing
                    elapsed_time_incorrect_plank = int(time.time() - start_time_incorrect)

                # Display angle values
                cv2.putText(image, str(int(left_body_angle)),
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(left_elbow_angle)),
                            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(left_shoulder_angle)),
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Body Corrections
                if 75 < left_elbow_angle < 95:
                    cv2.putText(image, "good for elbow", (40,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "make proper angle for elbow.", (40,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                    
                if 160 < left_body_angle < 180:
                    cv2.putText(image, "good for hip", (40,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Straighten your body.", (40,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

                if 75 < left_shoulder_angle < 95:
                    cv2.putText(image, "Good for shoulder.", (40,180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "make proper angle for shoulders.", (40,180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
 
            except Exception as e:
                print(e)
                
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )     

            # Display time spent correctly
            cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)
            cv2.putText(image, "Correct Time: " + str(elapsed_time_correct_plank) + "s", (100,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)

            # Display time spent incorrectly
            cv2.putText(image, "Incorrect Time: " + str(elapsed_time_incorrect_plank) + "s", (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)

            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break
        
    if camera is not None:
        camera.release()


# Variables for counting exercises
correct_counter_leg_raise = 0
incorrect_counter_leg_raise = 0

def LegRaiseExercise():
    global camera, correct_counter_leg_raise, incorrect_counter_leg_raise

    # Variables for counting exercises
    counter = 0 
    direction = 0

    last_correct = False
    stage = None

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder,right_elbow,right_wrist)
                right_body_angle = calculate_angle(right_ankle,right_hip,right_shoulder)
                left_body_angle = calculate_angle(left_ankle,left_hip,left_shoulder)
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)

                # Visualize angle
                #right_elbow_angle
                cv2.putText(image, str(right_elbow_angle), 
                           tuple(np.multiply(right_elbow, [640,480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                )
                #left_elbow_angle
                cv2.putText(image, str(left_elbow_angle), 
                            tuple(np.multiply(left_elbow, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #right_hip_angle
                cv2.putText(image, str(right_body_angle), 
                            tuple(np.multiply(right_hip, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #left_hip_angle
                cv2.putText(image, str(left_body_angle), 
                            tuple(np.multiply(left_hip, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #right_knee_angle
                cv2.putText(image, str(right_knee_angle), 
                            tuple(np.multiply(right_knee, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )
                #left_knee_angle
                cv2.putText(image, str(left_knee_angle), 
                            tuple(np.multiply(left_knee, [640,480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                    )

                prev_stage = stage

                if check_angles_for_posture_legraise(left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle) and check_for_upside_body_legraise(left_body_angle, right_body_angle):
                    stage = "down"
        
                elif stage=="down" and check_angles_for_posture_legraise(left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle) and check_for_downside_body_legraise(left_body_angle, right_body_angle):
                    stage = "up"
        
                    if check_for_threshold_leagraise(left_body_angle, right_body_angle) and check_angles_for_posture_legraise(left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle):
                        counter+=1
                        correct_counter_leg_raise+=1
                    else:
                        stage = "up"
                        incorrect_counter_leg_raise+=1
        
                if stage=="down" and check_for_threshold_high_legraise(left_body_angle, right_body_angle):
                    stage = "up"
                    incorrect_counter_leg_raise+=1
        
                if stage=="up" and check_for_threshold_low_legraise(left_body_angle, right_body_angle):
                    stage = "down"
                    incorrect_counter_leg_raise+=1

                # ... (rest of the code for drawing landmarks, displaying angles, etc.)
                cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)
            
                cv2.putText(image, "Correct Exercises: " + str(correct_counter_leg_raise), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, "Incorrect Exercises: " + str(incorrect_counter_leg_raise), (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)
        
                # Visualize angles
                cv2.putText(image, "Left knee Angle: " + str(round(left_knee_angle, 2)), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right knee Angle: " + str(round(right_knee_angle, 2)), (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Elbow Angle: " + str(round(left_elbow_angle, 2)), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Elbow Angle: " + str(round(right_elbow_angle, 2)), (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Body Angle: " + str(round(left_body_angle, 2)), (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Body Angle: " + str(round(right_body_angle, 2)), (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
            
            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break
        
    if camera is not None:
        camera.release()


# Variables for counting exercises
correct_counter_right_arm_curl = 0
incorrect_counter_right_arm_curl = 0

def RightArmCurlExercise():
    global camera, correct_counter_right_arm_curl, incorrect_counter_right_arm_curl

    # Curl counter variables
    counter = 0
    stage = None

    # Set arm curl angle thresholds
    arm_curl_threshold = (30, 160)

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                feedback = provide_feedback(angle, arm_curl_threshold)

                previous_stage = stage

                # Curl counter logic
                if angle > 150 and angle < 160:
                    stage = "down"
                    previous_stage = stage
                elif angle > 30 and stage == 'down' and angle < 50:
                    stage = "up"
                    prev_stage = stage
                    # Check if angle was within range during both up and down phases
                    if angle >= arm_curl_threshold[0] and angle <= arm_curl_threshold[1] and feedback=="Angle within range. Good job!" and stage=="up":
                        # previous_stage=stage
                        counter += 1
                        correct_counter_right_arm_curl += 1
                        print(f"Rep {counter}: Correct")
                        previous_stage=stage

                    elif prev_stage == "up" and feedback=="Angle too small. Extend your arm further.":
                        prev_stage="down"
                        incorrect_counter_right_arm_curl += 1
                        print(f"Rep {counter}: Incorrect (angle below threshold)")

                if stage == "down"  and angle > arm_curl_threshold[1]:
                    stage = "up"
                    incorrect_counter_right_arm_curl += 1
                    print(f"Rep {counter}: Incorrect (angle above threshold)")
                    
                if prev_stage == "up"  and angle < arm_curl_threshold[0]:
                    prev_stage = "down"
                    incorrect_counter_right_arm_curl += 1
                    print(f"Rep {counter}: Incorrect (angle below threshold)")

            except:
                pass

            cv2.rectangle(image, (0,0), (25 + 1400, 25 + 40), (189,246,254), -1)
            cv2.putText(image, "Correct Exercises: " + str(correct_counter_right_arm_curl), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, "Incorrect Exercises: " + str(incorrect_counter_right_arm_curl),(800,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (75,0,130), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            # Display feedback on frame
            cv2.putText(image, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('Camera Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break
        
    if camera is not None:
        camera.release()


# Variables for counting exercises
correct_counter_squat =  0
incorrect_counter_squat =  0

def SquatExercise():
    pass




# Webpages Routing.
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/arm_curl_ex')
def arm_curl_ex():
    return render_template('arm_curl.html')

@app.route('/side_arm_ex')
def side_arm_ex():
    return render_template('side_arm.html')

@app.route('/up_arm_ex')
def up_arm_ex():
    return render_template('up_arm.html')

@app.route('/wall_sit_ex')
def wall_sit_ex():
    return render_template('wall_sit.html')

@app.route('/plank_ex')
def plank_ex():
    return render_template('plank.html')

@app.route('/leg_raise_ex')
def leg_raise_ex():
    return render_template('leg_raise.html')

@app.route('/right_arm_curl_ex')
def right_arm_curl_ex():
    return render_template('right_arm_curl.html')

@app.route('/squat_ex')
def squat_ex():
    return render_template('squat.html')



# Route to get angle data
@app.route('/data')
def get_data():
    data = {
        'correct_counter' : correct_counter,
        'incorrect_counter' : incorrect_counter,
    }
    return jsonify(data)

@app.route('/data_side_arm')
def get_data_side_arm():
    data_side_arm= {
        'correct_counter_side_arm' : correct_counter_side_arm,
        'incorrect_counter_side_arm' : incorrect_counter_side_arm,
    }
    return jsonify(data_side_arm)

@app.route('/data_up_arm')
def get_data_up_arm():
    data_up_arm= {
        'correct_counter_up_arm' : correct_counter_up_arm,
        'incorrect_counter_up_arm' : incorrect_counter_up_arm,
    }
    return jsonify(data_up_arm)

@app.route('/data_wall_sit')
def get_data_wall_sit():
    data_wall_sit= {
        'elapsed_time_correct_wall_sit': elapsed_time_correct_wall_sit,
        'elapsed_time_incorrect_wall_sit':elapsed_time_incorrect_wall_sit,
    }
    return jsonify(data_wall_sit)

@app.route('/data_plank')
def get_data_plank():
    data_plank= {
        'elapsed_time_correct_plank': elapsed_time_correct_plank,
        'elapsed_time_incorrect_plank': elapsed_time_incorrect_plank,
    }
    return jsonify(data_plank)

@app.route('/data_leg_raise')
def get_data_leg_raise():
    data_leg_raise= {
        'correct_counter_leg_raise' : correct_counter_leg_raise,
        'incorrect_counter_leg_raise' : incorrect_counter_leg_raise,
    }
    return jsonify(data_leg_raise)

@app.route('/data_right_arm_curl')
def get_data_right_arm_curl():
    data_right_arm_curl = {
        'correct_counter_right_arm_curl' : correct_counter_right_arm_curl,
        'incorrect_counter_right_arm_curl' : incorrect_counter_right_arm_curl,
    }
    return jsonify(data_right_arm_curl)

@app.route('/data_squat')
def get_data_squat():
    data_squat = {
        'correct_counter_squat' : correct_counter_squat,
        'incorrect_counter_squat' : incorrect_counter_squat,
    }
    return jsonify(data_squat)



# Route to start camera.
@app.route('/start_camera_arm_curl')
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=ArmCurlExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_side_arm')
def start_camera_side_arm():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=SideArmExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_up_arm')
def start_camera_up_arm():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=UpArmExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_wall_sit')
def start_camera_wall_sit():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=WallSitExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_plank')
def start_camera_plank():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=PlankExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_leg_raise')
def start_camera_leg_raise():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=LegRaiseExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_right_arm_curl')
def start_camera_right_arm_curl():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=RightArmCurlExercise).start()  # Start processing frames in a separate thread
    return "Camera started"

@app.route('/start_camera_squat')
def start_camera_squat():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_resolution[1])
        threading.Thread(target=SquatExercise).start()  # Start processing frames in a separate thread
    return "Camera started"



# Route to stop camera.
@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None
    return "Camera stopped"

@app.route('/stop_camera_armcurl')
def stop_camera_armcurl():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data')
        data = response.json()
        correct_counter_exercise = data['correct_counter']
        incorrect_counter_exercise = data['incorrect_counter']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"arm_curl" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    

@app.route('/stop_camera_side_arm')
def stop_camera_side_arm():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data_side_arm')
        data = response.json()
        correct_counter_exercise = data['correct_counter_side_arm']
        incorrect_counter_exercise = data['incorrect_counter_side_arm']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"side_arm" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    

@app.route('/stop_camera_up_arm')
def stop_camera_up_arm():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data_up_arm')
        data = response.json()
        correct_counter_exercise = data['correct_counter_up_arm']
        incorrect_counter_exercise = data['incorrect_counter_up_arm']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"up_arm" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    

@app.route('/stop_camera_wall_sit')
def stop_camera_wall_sit():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data_wall_sit')
        data = response.json()
        correct_counter_exercise = data['elapsed_time_correct_wall_sit']
        incorrect_counter_exercise = data['elapsed_time_incorrect_wall_sit']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"wall_sit" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    

@app.route('/stop_camera_plank')
def stop_camera_plank():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data_plank')
        data = response.json()
        correct_counter_exercise = data['elapsed_time_correct_plank']
        incorrect_counter_exercise = data['elapsed_time_incorrect_plank']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"plank" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    

@app.route('/stop_camera_leg_raise')
def stop_camera_leg_raise():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data_leg_raise')
        data = response.json()
        correct_counter_exercise = data['correct_counter_leg_raise']
        incorrect_counter_exercise = data['incorrect_counter_leg_raise']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"leg_raise" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    
@app.route('/stop_camera_right_arm_curl')
def stop_camera_right_arm_curl():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None

        # Store exercise-specific counters in the database
        response = requests.get('http://localhost:5000/data_right_arm_curl')
        data = response.json()
        correct_counter_exercise = data['correct_counter_right_arm_curl']
        incorrect_counter_exercise = data['incorrect_counter_right_arm_curl']

        print(correct_counter_exercise)
        print(incorrect_counter_exercise)

        cursor.execute('insert into exercise_data.progress values(NOW(),"right_arm_curl" ,%s , %s )',(correct_counter_exercise,incorrect_counter_exercise))
        mydb.commit()

        return f"Camera stopped for {correct_counter_exercise}. and {incorrect_counter_exercise} Counters stored in the database."
    else:

        return "Camera stopped"
    


    
@app.route('/generate_graph_arm_curl')
def generate_graph_arm_curl():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('arm_curl',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'


@app.route('/generate_graph_side_arm')
def generate_graph_side_arm():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('side_arm',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'


@app.route('/generate_graph_up_arm')
def generate_graph_up_arm():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('up_arm',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'

@app.route('/generate_graph_wall_sit')
def generate_graph_wall_sit():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('wall_sit',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'


@app.route('/generate_graph_plank')
def generate_graph_plank():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('plank',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'


@app.route('/generate_graph_leg_raise')
def generate_graph_leg_raise():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('leg_raise',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'


@app.route('/generate_graph_right_arm_curl')
def generate_graph_right_arm_curl():

    # Get data from datbase
    db_data = pd.read_sql('select * from exercise_data.progress where exe_name=%s',params=('right_arm_curl',), con=mydb)

    grouped_data = db_data.groupby('pre_date').agg({'correct_ones':'sum','incorrect_ones':'sum'}).reset_index()

    # Generate data for the graph
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(8, 6))
    #for plotting correct ones
    plt.plot(grouped_data['pre_date'], grouped_data['correct_ones'], color='green', marker='o', label='Correct Ones')
    # Plot incorrect ones
    plt.plot(grouped_data['pre_date'], grouped_data['incorrect_ones'], color='red', marker='o', label='Incorrect Ones') 

    plt.legend()

    date_format = DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Date')
    plt.ylabel('Correct Ones')

    # Encode the plot image as a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{image_base64}" alt="Sine Wave">'


if __name__ == '__main__':
    app.run(debug=True)
