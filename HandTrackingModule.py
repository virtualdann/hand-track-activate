import cv2 as cv
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity= 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True, drawStatus=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        status = ""
        if results.multi_hand_landmarks:
            # cv.circle(frame, (mp.INDEX_FINGER_TIP.x, mp.INDEX_FINGER_TIP.y), 15, (255, 0, 0), 2)
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
                    status, frame = self.openStatus(frame, drawStatus, handLms)
        return status, frame

    def openStatus(self, frame, stat, handLms):
        img_height, img_width, _ = frame.shape
        x_ind = int(handLms.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].x * img_width)
        y_ind = int(handLms.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].y * img_height)
        x_thb = int(handLms.landmark[self.mpHands.HandLandmark.THUMB_TIP].x * img_width)
        y_thb = int(handLms.landmark[self.mpHands.HandLandmark.THUMB_TIP].y * img_height)

        length_line = math.sqrt(abs((x_ind - x_thb) ** 2 - (y_ind - y_thb) ** 2))
        # print("Length line : " + str(int(length_line)))
        w_h_ratio = img_height / img_width
        line_ratio = float(length_line / w_h_ratio)
        # print("Ratio line : " + str(line_ratio))
        line_color = (0, 255, 0)

        self.status = "Open"
        if (line_ratio < 100.00):
            line_color = (0, 0, 255)
            self.status = "Close"
        if stat:
            cv.line(frame, (x_ind, y_ind), (x_thb, y_thb), line_color, 3)
            cv.putText(frame, self.status, (x_ind + int(abs(x_ind - x_thb) / 2), y_ind + int(abs(y_ind - y_thb) / 2)),
                       cv.FONT_HERSHEY_PLAIN, 3, line_color, 3)

        return self.status, frame


def main():
    cap = cv.VideoCapture(0)
    curTime = 0
    prevTime = 0

    while True:
        success, frame = cap.read()
        detector = handDetector()

        status, frame = detector.findHands(frame, True, False)

        curTime = time.time()
        fps = 1/(curTime - prevTime)
        prevTime = curTime
        cv.putText(frame, str(int(fps)), (10, 100), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 0), 10)

        #cv.imshow("Frame Flipped", cv.flip(frame, 1))
        cv.imshow("Frame",frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()