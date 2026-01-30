# --- Libraries ---
import cv2                          # Computer Vision: The robot's eyes
import numpy as np                  # Math: Helps process the image grid
import RPi.GPIO as GPIO             # Hardware: Controls the Raspberry Pi pins
from picamera2 import Picamera2     # Camera: Grabs frames from the PiCam

# --- CONFIGURATION (Change these to tune your robot) ---
# Pins (BCM Numbering)
ENA, IN1, IN2 = 12, 23, 24          # Left Motor
ENB, IN3, IN4 = 13, 17, 27          # Right Motor

# Steering Settings
BASE_SPEED = 50                     # Speed on a straight line (0-100)
SPIN_POWER = 65                     # Power used for sharp pivot turns
KP = 1.2                            # How hard the robot steers (Proportional)
WIDTH, HEIGHT = 160, 120            # Camera resolution
CENTER = WIDTH // 2                 # The "Goal" (Pixel 80)

# Variables to remember
last_error = 0                      # Stores the last known side the line was on

# --- HARDWARE SETUP ---
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)

# Initialize PWM for speed control
pwmA = GPIO.PWM(ENA, 1000)          # Left Motor PWM (1kHz)
pwmB = GPIO.PWM(ENB, 1000)          # Right Motor PWM (1kHz)
pwmA.start(0)
pwmB.start(0)

# --- FUNCTIONS ---

def stop_motors():
    """Sets all motor pins to LOW and speed to 0."""
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

def search_for_line():
    """Spins in place based on the last known direction of the line."""
    global last_error
    search_speed = 90  # High power to overcome friction while spinning
    
    if last_error > 0:
        # PIVOT RIGHT
        GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    else:
        # PIVOT LEFT
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    
    pwmA.ChangeDutyCycle(search_speed)
    pwmB.ChangeDutyCycle(search_speed)

def move_robot(error):
    """Calculates motor speeds and handles sharp turn logic."""
    global last_error
    
    # Calculate how fast the error is changing
    diff = abs(error - last_error)
    last_error = error 

    # 1. SHARP TURN MODE (Pivot)
    # If the line is far to the side (>45px) or moving very fast (>30px change)
    if abs(error) > 45 or diff > 30:
        if error > 0: # Line is far Right -> Spin Right
            GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        else:        # Line is far Left -> Spin Left
            GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
            
        pwmA.ChangeDutyCycle(SPIN_POWER)
        pwmB.ChangeDutyCycle(SPIN_POWER)
        
    # 2. NORMAL DRIVING MODE
    else:
        # Standard forward motor signals
        GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH) 
        GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH) 
        
        # P-Control: steering is proportional to the error
        steering = error * KP
        
        l_speed = BASE_SPEED + steering
        r_speed = BASE_SPEED - steering

        # Clamp speed between 20 (min to move) and 100 (max)
        pwmA.ChangeDutyCycle(max(20, min(100, l_speed)))
        pwmB.ChangeDutyCycle(max(20, min(100, r_speed)))

# --- CAMERA INITIALIZATION ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

# --- MAIN LOOP ---
try:
    while True:
        # 1. Capture and Process Image
        frame = picam2.capture_array()
        color_view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert for OpenCV
        gray = cv2.cvtColor(color_view, cv2.COLOR_BGR2GRAY)  # Gray is faster
        
        # 2. Slice the image (ROI) - looking at a middle-low strip
        roi = gray[40:100, 0:160] 
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # 3. Create Binary Map (OTSU finds the best threshold automatically)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Calculate Center of Mass (Moments)
        M = cv2.moments(thresh)
        
        if M['m00'] > 500: # We see the line!
            cx = int(M['m10'] / M['m00'])
            error = cx - CENTER 
            
            # Save error to memory if it's significant
            if abs(error) > 30:
                last_error = error
                
            move_robot(error)
            
            # Draw a green dot where the robot "thinks" the line is
            cv2.circle(color_view, (cx, 70), 5, (0, 255, 0), -1)
        else:
            # We lost the line!
            search_for_line()

        # 5. Visual Debugging (Shows what the robot sees)
        thresh_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) # Convert to 3-channel for stacking
        combined = np.hstack((color_view, cv2.resize(thresh_3ch, (WIDTH, HEIGHT))))
        cv2.imshow("Robot Vision (Raw vs Processed)", cv2.resize(combined, (640, 240)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    print("\nRobot stopped by user.")
finally:
    # --- CLEANUP ---
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    picam2.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("Cleanup complete.")