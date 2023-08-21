import cv2

cap = cv2.VideoCapture('../../../examples/media/test_video4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
