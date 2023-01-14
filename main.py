import  cv2
import hand_tracking_module as htm
import time

hand_detection = htm.HandDetection()
prev_time = 0
cap = cv2.VideoCapture(0)


while True:
    success, frame = cap.read()
    cur_time = time.time()
    fps = int(1 / (cur_time - prev_time))
    prev_time = cur_time

    if not success:
        break
    hands_img = hand_detection.get_hands(frame)
    positions = hand_detection.get_positions(hands_img)
    if len(positions) > 0:
        print(positions[4])
    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # cv2.imshow("image", hands_img) this will not produce selfie mode, so flipped it
    cv2.imshow('image', cv2.flip(hands_img, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()