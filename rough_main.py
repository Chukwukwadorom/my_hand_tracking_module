import cv2
import mediapipe as mp
import time



cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

cur_time = 0
prev_time = 0

while True:
    success, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)
    # print(result.detections)

    cur_time = time.time()
    fps = int(1 / (cur_time - prev_time))
    prev_time = cur_time

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                x, y = lm.x, lm.y
                h, w, c = frame.shape
                cx, cy = int(w*x), int(h*y)
                #if id == 4:
                 #   cv2.circle(frame, (cx,cy), 15, (0,0, 255), -1)

            cv2.putText(frame,str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

    if not success:
        break
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()