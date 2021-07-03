import numpy as np
import cv2
cap = cv2.VideoCapture('video.mp4')
vehicle_count = 0
#https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
backSub = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold=100, detectShadows=True)
def detect_actions(dot_y, line_y, start_or_finish, frame):
    global vehicle_count
    if start_or_finish == 1: #Start
        if abs(dot_y - 150) < 2:
            print("Vehicle entered starting zone!")
            return True
    elif start_or_finish == -1:#End
         if abs(dot_y - 70) < 2 :
            print("Vehicle left ending zone!")
            return True

           
            


def circle_center(frame, center_x, center_y):
    center_ = (int(center_x), int(center_y))
    cv2.circle(frame, center_, 1, (0,0,255), 5)

def draw_interest_area_lines(frame, start, finish, width):
    cv2.line(frame, (0, int(start)), (int(width), int(start)), (0,255,0), 3)
    cv2.line(frame, (0, int(finish)), (int(width), int(finish)), (0,0,255), 3)


while(1):
    ret, frame = cap.read()

    start = False
    end = False
    cv2.putText(frame, "Number of cars: " + str(vehicle_count), (100,50), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,0),2 )
    interest_area = frame[600:1080, 720:1000]
    backup = frame[600:1080, 720:1000]
    
    interest_area = cv2.cvtColor(interest_area, cv2.COLOR_BGR2GRAY) #Convert it to grayscale before applsying a gaussian blur.
    interest_area = cv2.GaussianBlur(interest_area, (3,3), 5) #Apply Gaussian Blur.
    foreground = backSub.apply(interest_area)#Substract the current frame from the background to find the foreground objects.
    junk, foreground = cv2.threshold(foreground, 244, 255, cv2.THRESH_BINARY)#Apply a threshold.

    foreground = cv2.dilate(foreground, None, iterations = 2)#Apply a dilate to remove the noisy pixels.

    contours, junk = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = interest_area.shape
    #print("h,w," ,height, " ", width)
    start_line = 150 + 600
    end_line = 40 + 600
    draw_interest_area_lines(frame, start_line, end_line, 900)

    
    for single in contours:
        area = cv2.contourArea(single)
        if area > 800:
            #cv2.drawContours(interest_area, [single], -1, (128, 255, 128))
            x,y,w,h = cv2.boundingRect(single)
            
            
            center_x = (2 * x + w ) / 2
            center_y = (2 * y + h) / 2
            circle_center(backup,center_x, center_y)
            cv2.putText(backup, str(center_y), (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 2,(255,0,0),2 )
            cv2.rectangle(backup, (x,y), (x+w, y+h), (0, 255, 0), 3)
            #print("Start line, end line and dot: ", start_line,end_line, center_y)
            start = detect_actions(center_y, start_line, 1, frame)
            end = detect_actions(center_y, end_line, -1,frame)
            if start == True or end == True:
                 vehicle_count+=1

  
       
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground", foreground)
    cv2.imshow("interest_area", backup)
    listener = cv2.waitKey(10)
    if cv2.waitKey(27) == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()