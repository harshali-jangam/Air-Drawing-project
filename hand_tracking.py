import cv2
from cvzone.HandTrackingModule import HandDetector

# Start webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for natural mirror view
    
    hands, img = detector.findHands(img)  # Detect hands
    
    if hands:
        hand = hands[0]  # Get the first detected hand
        lmList = hand["lmList"]  # List of 21 landmarks
        bbox = hand["bbox"]      # Bounding box around hand
        
        # Draw a circle on the tip of index finger (landmark 8)
        x, y = lmList[8][0], lmList[8][1]
        cv2.circle(img, (x, y), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Hand Tracking", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
