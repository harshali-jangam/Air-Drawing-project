import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ------------------ Setup ------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Canvas (drawing layer)
imgCanvas = np.zeros((480, 640, 3), np.uint8)

# Previous position
xp, yp = 0, 0
path = []  # store finger points

# Default brush
color = (255, 0, 0)
brushThickness = 5
eraserThickness = 50
alpha = 0.5
eraserMode = False

print("Controls: E=Eraser, T=Toggle Brush, 1=Blue, 2=Red, 3=Green, R=Reset, S=Save, Q=Quit")

# ------------------ Loop ------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        x1, y1 = lmList[8][0], lmList[8][1]

        # -------- Drawing --------
        if fingers[1] == 1:  # index up
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            x_smooth = int(xp * (1 - alpha) + x1 * alpha)
            y_smooth = int(yp * (1 - alpha) + y1 * alpha)

            if eraserMode:
                # true erase (mask to black)
                cv2.circle(imgCanvas, (x_smooth, y_smooth), eraserThickness, (0, 0, 0), -1)
            else:
                cv2.line(imgCanvas, (xp, yp), (x_smooth, y_smooth), color, brushThickness)

            xp, yp = x_smooth, y_smooth
            path.append((x_smooth, y_smooth))

        else:
            # finger lifted → check shape
            if len(path) > 20:  # enough points
                pts = np.array(path, dtype=np.int32)

                # smooth + close contour
                hull = cv2.convexHull(pts)

                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)

                if len(approx) == 3:  # triangle
                    cv2.drawContours(imgCanvas, [approx], 0, color, -1)
                elif len(approx) == 4:  # square/rect
                    cv2.drawContours(imgCanvas, [approx], 0, color, -1)
                elif len(approx) > 6:  # circle
                    (x, y), radius = cv2.minEnclosingCircle(pts)
                    cv2.circle(imgCanvas, (int(x), int(y)), int(radius), color, -1)

            path = []
            xp, yp = 0, 0

    # -------- Merge --------
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    imgCombined = cv2.bitwise_and(img, imgInv)
    imgCombined = cv2.bitwise_or(imgCombined, imgCanvas)

    cv2.imshow("Air Drawing", imgCombined)
    cv2.imshow("Canvas", imgCanvas)

    # -------- Controls --------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("AirDrawing.png", imgCanvas)
        print("Saved!")
    elif key == ord('r'):
        imgCanvas = np.zeros((480, 640, 3), np.uint8)
        path = []
        print("Canvas Reset")
    elif key == ord('e'):
        eraserMode = not eraserMode
        print("Eraser ON" if eraserMode else "Eraser OFF")
    elif key == ord('1'):
        color = (255, 0, 0); eraserMode = False
    elif key == ord('2'):
        color = (0, 0, 255); eraserMode = False
    elif key == ord('3'):
        color = (0, 255, 0); eraserMode = False
    elif key == ord('t'):
        brushThickness = 15 if brushThickness == 5 else 5

cap.release()
cv2.destroyAllWindows()
