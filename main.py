#-- Libraries --
import cv2                          # image processing
import numpy as np                  # process pixels
import RPi.GPIO as GPIO             # control GPIO pins
from picamera2 import Picamera2     # camera

# --- MOTOR SETUP ---
GPIO.setmode(GPIO.BCM)              

# Left Motor pins
ENA, IN1, IN2 = 12, 23, 24

# Right Motor pins
ENB, IN3, IN4 = 13, 17, 27

GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)

pwmA.start(0)
pwmB.start(0)

last_error = 0
Kp = 0.7

def move_robot(error):
    global last_error # Use global to remember last error
    
    # 1. Calculate how fast the error is changing (Derivative)
    # If this is a huge number, we are hitting a sharp turn
    diff = abs(error - last_error)
    last_error = error 

    # 2. Aggressive Sharp Turn Detection
    # If error is high OR changing very fast
    if abs(error) > 45 or diff > 30:
        spin_power = 55 # Lowered slightly so it doesn't "spin past" the line
        
        if error > 0: # Sharp Right
            GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        else: # Sharp Left
            GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
            
        pwmA.ChangeDutyCycle(spin_power)
        pwmB.ChangeDutyCycle(spin_power)
        
    else:
        # 3. Normal Steering - SLOWER base speed for better accuracy
        base_speed = 50 
        kp = 1.2
        steering = error * kp  #higher error -> more steering adjustment

        #move forward
        GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH) 
        GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH) 
        
        l_speed = base_speed + steering
        r_speed = base_speed - steering

        #allow speed between 20 and 100
        pwmA.ChangeDutyCycle(max(20, min(100, l_speed)))
        pwmB.ChangeDutyCycle(max(20, min(100, r_speed)))
        
# --- IN YOUR SEARCH FUNCTION ---
def search_for_line():
    global last_error
    # Use max power to break the static friction of the floor
    spin_speed = 95 
    
    if last_error > 0:
        # PIVOT RIGHT (Left wheel forward, Right wheel back)
        GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    else:
        # PIVOT LEFT (Left wheel back, Right wheel forward)
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    
    pwmA.ChangeDutyCycle(spin_speed)
    pwmB.ChangeDutyCycle(spin_speed)
  
    
def stop_motors():
    GPIO.output([IN1,IN2,IN3,IN4], GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# --- CAMERA SETUP ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (160, 120)}) #160 pixels wide, 120 pixels tall
picam2.configure(config)
picam2.start()

try:
    while True:
        frame = picam2.capture_array()                       #store image into grid of numbers
        color_view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #convert RGB to BGR for OpenCV
        gray = cv2.cvtColor(color_view, cv2.COLOR_BGR2GRAY) #convert BGR to Grayscale
        roi = gray[20:80, 0:160]                            #region of interest (crop to area around line) 20 to 80 vertical, 0 to 160 horizontal
        blur = cv2.GaussianBlur(roi, (5, 5), 0)             #blur out 
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #threshold to binary image, invert black and white

        M = cv2.moments(thresh)
        if M['m00'] > 500:
            cx = int(M['m10'] / M['m00'])
            # HERE IS WHERE WE GET THE ERROR
            error = cx - 80                     #balance point is 80, so error is how far from center the line is
                                                #pos -> robot too left -> turn right
                                                #neg -> robot too right -> turn left

            # If the line is far to the side, save that memory!
            if abs(error) > 40:
                last_error = error
                
            move_robot(error)
            cv2.circle(color_view, (cx, 90), 5, (0, 255, 0), -1)
        else:
            search_for_line()

        # Side-by-side display
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_resized = cv2.resize(thresh_color, (160, 120))
        combined = np.hstack((color_view, thresh_resized))
        cv2.imshow("Robot Vision", cv2.resize(combined, (640, 240)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    pass
finally:
    stop_motors() # Use our new robust stop
    pwmA.stop()
    pwmB.stop()
    picam2.stop()
    GPIO.cleanup() # This is the most important line to turn off pins
    cv2.destroyAllWindows()
    print("Cleanup complete.")
