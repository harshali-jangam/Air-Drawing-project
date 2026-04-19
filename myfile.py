import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

# ------------------ Setup ------------------
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.85, maxHands=1)

# Canvas (drawing layer)
imgCanvas = np.zeros((480, 640, 3), np.uint8)

# Previous position
xp, yp = 0, 0
path = []  # store finger points

# Brush / Eraser
color = (255, 0, 0)
brushThickness = 5
eraserThickness = 50
eraserMode = False

# Smoothing + stability
alpha = 0.3
minMoveDist = 4

# Coordinate printing
lastPrint = time.time()
printDelay = 0.25  # seconds

# Robot action simulation
robotAction = "Idle"

print("Controls: E=Eraser, T=Toggle Brush, 1=Blue, 2=Red, 3=Green, R=Reset, S=Save, Q=Quit")

# ------------------ Robust Shape Detection Function ------------------
def detect_shape(path, canvas, color=(255,0,0)):
    shape = "Curve"
    robotAction = "Idle"

    if len(path) < 10:
        return shape, robotAction

    # Draw path on blank mask with thick lines
    mask = np.zeros(canvas.shape[:2], np.uint8)
    for i in range(1, len(path)):
        cv2.line(mask, tuple(path[i-1]), tuple(path[i]), 255, 12)

    # Dilate to fill gaps
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return shape, robotAction

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 300:  # ignore very small blobs
        return shape, robotAction

    # Approx polygon
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04*peri, True)

    # Bounding box & circularity
    x, y, w, h = cv2.boundingRect(approx)
    area = cv2.contourArea(cnt)
    circularity = 4*np.pi*area/(peri*peri) if peri !=0 else 0

    if len(approx) == 3:
        shape = "Triangle"
        robotAction = "Pick & Place"
    elif len(approx) == 4:
        ar = w/float(h)
        shape = "Square" if 0.8 < ar < 1.2 else "Rectangle"
        robotAction = "Move in Square Path"
    elif len(approx) <= 2:
        shape = "Line"
        robotAction = "Move Forward"
    elif circularity > 0.75:
        shape = "Circle"
        robotAction = "Rotate in Place"
    else:
        shape = "Curve"
        robotAction = "Idle"

    # Draw detected shape
    if shape == "Circle":
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, -1)
    else:
        cv2.drawContours(canvas, [approx], -1, color, -1)

    return shape, robotAction

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
        x1, y1 = lmList[8][0], lmList[8][1]  # index fingertip

        # -------- Drawing Mode --------
        if fingers[1] == 1:  # index up → drawing
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Smooth coordinates
            x_smooth = int(xp * (1 - alpha) + x1 * alpha)
            y_smooth = int(yp * (1 - alpha) + y1 * alpha)

            if abs(x_smooth - xp) > minMoveDist or abs(y_smooth - yp) > minMoveDist:
                if eraserMode:
                    cv2.circle(imgCanvas, (x_smooth, y_smooth), eraserThickness, (0, 0, 0), -1)
                else:
                    cv2.line(imgCanvas, (xp, yp), (x_smooth, y_smooth), color, brushThickness)

                path.append((x_smooth, y_smooth))
                xp, yp = x_smooth, y_smooth

                # Print coordinates slowly
                if time.time() - lastPrint > printDelay:
                    print(f"Coordinate: ({x_smooth}, {y_smooth})")
                    lastPrint = time.time()

        else:
            # -------- Finger lifted → Shape Recognition --------
            if len(path) > 5:
                shapeDetected, robotAction = detect_shape(path, imgCanvas, color)
                path = []
                xp, yp = 0, 0
                print(f"Detected Shape: {shapeDetected} → Robot Action: {robotAction}")

    # -------- Merge Canvas with Camera --------
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    imgCombined = cv2.bitwise_and(img, imgInv)
    imgCombined = cv2.bitwise_or(imgCombined, imgCanvas)

    # -------- Robot Action Display --------
    cv2.putText(imgCombined, f"Robot Action: {robotAction}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
        robotAction = "Idle"
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
