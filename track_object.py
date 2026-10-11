import cv2
import numpy as np
import time
from ultralytics import YOLOWorld

# ---------- settings ----------
H_TOL = 10                # hue tolerance: H ± 10
SV_TOL = 70               # how much below the selected point we set S min and V min
PATCH = 5                 # radius of the square around the click (5 -> 11x11 pixels)
BTN = (460, 10, 620, 50)  # button on the image: x1, y1, x2, y2
# --------------------------------I

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pick_mode = False     # True = waiting for a click on the object
click = None          # mouse stores click coordinates here
mouse_pos = (0, 0)    # current cursor position
picked = None         # last selected HSV (for display on screen)

def nothing(x):
    pass

def on_mouse(event, x, y, flags, param):
    global pick_mode, click, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        x1, y1, x2, y2 = BTN
        if x1 <= x <= x2 and y1 <= y <= y2:
            pick_mode = not pick_mode      # click on the button: enable/disable
        elif pick_mode:
            click = (x, y)                 # click on the image: handle in the loop

cv2.namedWindow("camera")
cv2.setMouseCallback("camera", on_mouse)

cv2.namedWindow("mask")
cv2.createTrackbar("H min", "mask", 35, 179, nothing)
cv2.createTrackbar("H max", "mask", 85, 179, nothing)
cv2.createTrackbar("S min", "mask", 100, 255, nothing)
cv2.createTrackbar("V min", "mask", 80, 255, nothing)

kernel = np.ones((5, 5), np.uint8)         # 5x5 brush for cleaning the mask

while True:
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera is not returning frames")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # BGR -> HSV

    # --- object clicked: take its color and move the sliders ---
    if click is not None:
        height, width = hsv.shape[:2]
        x = min(max(click[0], 0), width - 1)
        y = min(max(click[1], 0), height - 1)
        click = None
        pick_mode = False

        # median over a small square instead of a single pixel - less noise
        patch = hsv[max(0, y - PATCH):y + PATCH + 1, max(0, x - PATCH):x + PATCH + 1]
        h, s, v = [int(c) for c in np.median(patch.reshape(-1, 3), axis=0)]
        picked = (h, s, v)

        cv2.setTrackbarPos("H min", "mask", (h - H_TOL) % 180)
        cv2.setTrackbarPos("H max", "mask", (h + H_TOL) % 180)
        cv2.setTrackbarPos("S min", "mask", max(0, s - SV_TOL))
        cv2.setTrackbarPos("V min", "mask", max(0, v - SV_TOL))

        print(f"Selected: H={h}, S={s}, V={v}")
        if s < 40:
            print("Warning: the color is almost gray/white/black, so H is unreliable here")

    h_min = cv2.getTrackbarPos("H min", "mask")
    h_max = cv2.getTrackbarPos("H max", "mask")
    s_min = cv2.getTrackbarPos("S min", "mask")
    v_min = cv2.getTrackbarPos("V min", "mask")

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2    # frame center

    # 1. Mask
    if h_min <= h_max:
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)      # white = match, black = no match
    else:
        # H min > H max: hue "wraps" through 0 (common with red),
        # so we combine two ranges: [H min .. 179] and [0 .. H max]
        m1 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

    # --- the button and hints are drawn on a copy of the frame; hsv and mask stay clean ---
    view = frame.copy()
    x1, y1, x2, y2 = BTN
    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 200, 0) if pick_mode else (200, 200, 200), -1)
    cv2.putText(view, "PICK COLOR", (x1 + 12, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if pick_mode:
        mx, my = mouse_pos
        cv2.rectangle(view, (mx - PATCH, my - PATCH), (mx + PATCH, my + PATCH), (255, 255, 255), 1)
        cv2.putText(view, "Click on the object", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if picked is not None:
        cv2.putText(view, f"picked HSV = {picked}", (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 2. Remove small noise
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 3. Find separate white blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    status = "no object"
    if contours:
        biggest = max(contours, key=cv2.contourArea)   # largest blob
        if cv2.contourArea(biggest) > 300:             # ignore very small ones
            # 4. Center of mass of the blob
            m = cv2.moments(biggest)
            obj_x = int(m["m10"] / m["m00"])
            obj_y = int(m["m01"] / m["m00"])

            # 5. Error: how far the object is shifted relative to the frame center
            error_x = obj_x - center_x     # > 0: object is to the right of center
            error_y = obj_y - center_y     # > 0: object is lower than center
            status = f"dx={error_x} dy={error_y}"

            cv2.drawContours(view, [biggest], -1, (0, 255, 0), 2)
            cv2.circle(view, (obj_x, obj_y), 6, (0, 0, 255), -1)
            cv2.line(view, (center_x, center_y), (obj_x, obj_y), (0, 255, 255), 2)

    cv2.drawMarker(view, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    #cv2.imshow("camera", frame)
    cv2.imshow("camera", view)
    cv2.imshow("mask", mask)               # now this is the cleaned mask
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
