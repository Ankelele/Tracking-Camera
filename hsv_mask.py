import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def nothing(x):
    pass

cv2.namedWindow("mask")
cv2.createTrackbar("H min", "mask", 35, 179, nothing)
cv2.createTrackbar("H max", "mask", 85, 179, nothing)
cv2.createTrackbar("S min", "mask", 100, 255, nothing)
cv2.createTrackbar("V min", "mask", 80, 255, nothing)

while True:
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera is not returning frames")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR -> HSV
    h_min = cv2.getTrackbarPos("H min", "mask")
    h_max = cv2.getTrackbarPos("H max", "mask")
    s_min = cv2.getTrackbarPos("S min", "mask")
    v_min = cv2.getTrackbarPos("V min", "mask")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)  # white = matches, black = no match

    cv2.imshow("camera", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) == ord("q"):
        print(f"lower = ({h_min}, {s_min}, {v_min}), upper = ({h_max}, 255, 255)")
        break

cap.release()
cv2.destroyAllWindows()