import cv2 as cv
import mediapipe as mp
import time
import math
import HandTrackingModule as htm
from pynput.mouse import Button, Controller
import threading


cap = cv.VideoCapture(0)
curTime = 0
prevTime = 0

mouse = Controller()

while True:
    success, frame = cap.read()
    detector = htm.handDetector()

    status, frame = detector.findHands(frame, True, True)

    # Display FPS
    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime
    cv.putText(frame, str(int(fps)), (10, 100), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 0), 10)

    #print("Status : " + status)
    if (status == "Close"):
        threading.Thread(mouse.click(Button.left))

    cv.imshow("Frame",frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
