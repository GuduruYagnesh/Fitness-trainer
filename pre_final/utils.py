import numpy as np


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

## Arm Curl
# Arm Curl Feedback
def provide_feedback(angle, threshold):
    feedback = ""
    if angle < threshold[0]:
        feedback = "Angle too small. Extend your arm further."
    elif angle > threshold[1]:
        feedback = "Angle too large. Bend your arm more (but still count)."
    else:
        feedback = "Angle within range. Good job!"
    return feedback


##Side Arm
# Function to check if angles are within threshold range for side arm exercise
def check_angles_for_posture_sidearm(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle):
    return (165 < left_elbow_angle < 180) and (165 < right_elbow_angle < 180) and (165 < left_body_angle < 180) and (165 < right_body_angle < 180)

def check_for_upside_shoulder_sidearm(left_shoulder_angle, right_shoulder_angle):
    return (80 < left_shoulder_angle < 95) and (80 < right_shoulder_angle < 95)

def check_for_downside_shoulder_sidearm(left_shoulder_angle, right_shoulder_angle):
    return (0 < left_shoulder_angle < 15) and (0 < right_shoulder_angle < 15)

def check_for_threshold_sidearm(left_shoulder_angle, right_shoulder_angle):
    return (10 < left_shoulder_angle < 90) and (10 < right_shoulder_angle < 90)

def check_for_threshold_high_sidearm(left_shoulder_angle, right_shoulder_angle):
    return (left_shoulder_angle > 90) and (right_shoulder_angle > 90)

def check_for_threshold_low_sidearm(left_shoulder_angle, right_shoulder_angle):
    return (left_shoulder_angle < 10) and (right_shoulder_angle < 10)


## Up Arm
# Calculating the angles for the hips and elbows
def calculate_angle_h_e(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Calculating the angles for the shoulder
def calculate_angle_shoulder(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # if angle > 180.0:
    #     angle = 360 - angle

    return angle


# Function to check if angles are within threshold range for arm up raise
def check_angles_for_posture_upraise(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle):
    return (165 < left_elbow_angle < 180) and (165 < right_elbow_angle < 180) and (165 < left_body_angle < 180) and (165 < right_body_angle < 180)

def check_for_upside_shoulder_upraise(left_shoulder_angle, right_shoulder_angle):
    return (165 < left_shoulder_angle < 180) and (165 < right_shoulder_angle < 180)

def check_for_downside_shoulder_upraise(left_shoulder_angle, right_shoulder_angle):
    return (0 < left_shoulder_angle < 15) and (0 < right_shoulder_angle < 15)

def check_for_threshold_upraise(left_shoulder_angle, right_shoulder_angle):
    return (10 < left_shoulder_angle < 185) and (10 < right_shoulder_angle < 185)

def check_for_threshold_high_upraise(left_shoulder_angle, right_shoulder_angle):
    return (left_shoulder_angle > 185) and (right_shoulder_angle > 185)

def check_for_threshold_low_upraise(left_shoulder_angle, right_shoulder_angle):
    return (left_shoulder_angle < 10) and (right_shoulder_angle < 10)


## Wall Sit
# Function to calculate angle for wall sit
def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to check if angles are within the specified range for wall sit
def check_angles_in_range_for_wallsit(left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle):
    return (70 < left_hip_angle < 100) and (70 < right_hip_angle < 100) and (70 < left_knee_angle < 100) and (70 < right_knee_angle < 100)


## Leg Raise
# Function to check if angles are within threshold range for leg raise
def check_angles_for_posture_legraise(left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle):
    return (165 < left_elbow_angle < 180) and (165 < right_elbow_angle < 180) and (165 < left_knee_angle < 180) and (165 < right_knee_angle < 180)

def check_for_upside_body_legraise(left_body_angle, right_body_angle):
    return (70 < left_body_angle < 120) and (70 < right_body_angle < 120)

def check_for_downside_body_legraise(left_body_angle, right_body_angle):
    return (160 < left_body_angle < 180) and (160 < right_body_angle < 180)

def check_for_threshold_leagraise(left_body_angle, right_body_angle):
    return (70 < left_body_angle < 180) and (70 < right_body_angle < 180)

def check_for_threshold_high_legraise(left_body_angle, right_body_angle):
    return (left_body_angle > 180) and (right_body_angle > 180)

def check_for_threshold_low_legraise(left_body_angle, right_body_angle):
    return (left_body_angle < 70) and (right_body_angle < 70)


## Plank
# #to check for the given range for plank
def check_angles_in_range_for_plank(left_elbow_angle, left_body_angle,left_shoulder_angle):
    return (75 < left_elbow_angle < 95) and (75 < left_shoulder_angle < 95) and (160 < left_body_angle <180)
