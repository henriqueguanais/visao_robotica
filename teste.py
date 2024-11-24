import cv2

cap = cv2.VideoCapture('videos/peixes.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('video', frame)

cap.release()