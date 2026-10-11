import cv2
import numpy as np

# ---------- настройки ----------
H_TOL = 10                # допуск по оттенку: H ± 10
SV_TOL = 70               # на сколько ниже выбранной точки ставим S min и V min
PATCH = 5                 # радиус квадрата вокруг клика (5 -> 11x11 пикселей)
BTN = (460, 10, 620, 50)   # кнопка на картинке: x1, y1, x2, y2
# --------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # свой номер камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pick_mode = False     # True = ждём клик по объекту
click = None          # сюда мышь кладёт координаты клика
mouse_pos = (0, 0)    # где сейчас курсор
picked = None         # последний выбранный HSV (для подписи на экране)

def nothing(x):
    pass

def on_mouse(event, x, y, flags, param):
    global pick_mode, click, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        x1, y1, x2, y2 = BTN
        if x1 <= x <= x2 and y1 <= y <= y2:
            pick_mode = not pick_mode      # клик по кнопке: вкл/выкл
        elif pick_mode:
            click = (x, y)                 # клик по картинке: обработаем в цикле

cv2.namedWindow("camera")
cv2.setMouseCallback("camera", on_mouse)

cv2.namedWindow("mask")
cv2.createTrackbar("H min", "mask", 35, 179, nothing)
cv2.createTrackbar("H max", "mask", 85, 179, nothing)
cv2.createTrackbar("S min", "mask", 100, 255, nothing)
cv2.createTrackbar("V min", "mask", 80, 255, nothing)

kernel = np.ones((5, 5), np.uint8)         # «кисточка» 5x5 для чистки маски

while True:
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Камера не отдаёт кадры")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # BGR -> HSV
    
        # --- был клик по объекту: берём его цвет и двигаем ползунки ---
    if click is not None:
        height, width = hsv.shape[:2]
        x = min(max(click[0], 0), width - 1)
        y = min(max(click[1], 0), height - 1)
        click = None
        pick_mode = False
 
        # медиана по небольшому квадрату, а не один пиксель - так меньше шума
        patch = hsv[max(0, y - PATCH):y + PATCH + 1, max(0, x - PATCH):x + PATCH + 1]
        h, s, v = [int(c) for c in np.median(patch.reshape(-1, 3), axis=0)]
        picked = (h, s, v)
        
        cv2.setTrackbarPos("H min", "mask", (h - H_TOL) % 180)
        cv2.setTrackbarPos("H max", "mask", (h + H_TOL) % 180)
        cv2.setTrackbarPos("S min", "mask", max(0, s - SV_TOL))
        cv2.setTrackbarPos("V min", "mask", max(0, v - SV_TOL))
 
        print(f"Выбрано: H={h}, S={s}, V={v}")
        if s < 40:
            print("Внимание: цвет почти серый/белый/чёрный, оттенок H тут ненадёжен")
 
    h_min = cv2.getTrackbarPos("H min", "mask")
    h_max = cv2.getTrackbarPos("H max", "mask")
    s_min = cv2.getTrackbarPos("S min", "mask")
    v_min = cv2.getTrackbarPos("V min", "mask")

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2    # центр кадра

    # 1. Маска
    if h_min <= h_max:
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)      # белое = подходит, чёрное = нет
    else:
        # H min > H max: оттенок "переходит" через 0 (так бывает у красного),
        # поэтому склеиваем два диапазона: [H min .. 179] и [0 .. H max]
        m1 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        
    # --- кнопка и подсказки рисуются на копии кадра, hsv и маска остаются чистыми ---
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
    #cv2.imshow("camera", view)

    # 2. Чистим мелкий мусор
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 3. Ищем отдельные белые пятна
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    status = "no object"
    if contours:
        biggest = max(contours, key=cv2.contourArea)   # самое большое пятно
        if cv2.contourArea(biggest) > 300:             # совсем мелкие игнорируем
            # 4. Центр масс пятна
            m = cv2.moments(biggest)
            obj_x = int(m["m10"] / m["m00"])
            obj_y = int(m["m01"] / m["m00"])

            # 5. Ошибка: насколько объект смещён от центра кадра
            error_x = obj_x - center_x     # > 0: объект правее центра
            error_y = obj_y - center_y     # > 0: объект НИЖЕ центра
            status = f"dx={error_x} dy={error_y}"

            cv2.drawContours(view, [biggest], -1, (0, 255, 0), 2)
            cv2.circle(view, (obj_x, obj_y), 6, (0, 0, 255), -1)
            cv2.line(view, (center_x, center_y), (obj_x, obj_y), (0, 255, 255), 2)

    cv2.drawMarker(view, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    #cv2.imshow("camera", frame)
    cv2.imshow("camera", view)
    cv2.imshow("mask", mask)               # теперь это уже очищенная маска
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
