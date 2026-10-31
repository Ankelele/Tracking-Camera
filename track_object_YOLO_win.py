# Трекинг объекта по цвету (HSV) + поиск объекта по тексту нейросетью YOLO-World.
#
#   T                 - напечатать, что искать (по-английски, например: red cup)
#   Enter / Esc       - искать / отмена (пока печатаете)
#   кнопка PICK COLOR - выбрать цвет кликом по объекту, как раньше
#   Q                 - выход
#
# Для поиска по тексту нужна библиотека: pip install -U ultralytics
# Без неё скрипт работает как раньше, только поиск по тексту выдаст ошибку в консоль.

import sys
import cv2
import numpy as np

# ---------- settings ----------
H_TOL = 10  # hue tolerance: H ± 10
SV_TOL = 70  # how much below the selected point we set S min and V min
PATCH = 5  # radius of the square around the click (5 -> 11x11 pixels)
BTN = (460, 10, 620, 50)  # button on the image: x1, y1, x2, y2
MODEL_NAME = "yolov8s-worldv2.pt"  # нейросеть, которая ищет объект по тексту (скачается при первом поиске)
CONF = 0.15  # порог уверенности нейросети: ниже -> находит больше, но чаще ошибается
IMGSZ = 640  # размер кадра для нейросети; на Raspberry Pi можно 416 или 320: быстрее, но хуже видит мелкие предметы
CAMERA_INDEX = 0
# --------------------------------


def open_camera(index):
    """Открывает камеру. Бэкенд (способ доступа к камере) зависит от операционной системы, поэтому пробуем по очереди."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Windows
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]  # macOS
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]  # Linux
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, _ = cap.read()  # проверяем, что кадры реально идут
        if ok:
            return cap
        cap.release()
    raise SystemExit(
        f" не удалось открыть камеру {index}: проверьте номер и не занята ли она другой программой"
    )


cap = open_camera(CAMERA_INDEX)

# ------------------------------- color pick variables -------------------------------
pick_mode = False  # True = waiting for a click on the object
click = None  # mouse stores click coordinates here
mouse_pos = (0, 0)  # current cursor position
picked = None  # last selected HSV (for display on screen)

# -------------------------------- YOLO search variables -------------------------------
typing = False  # what the user is typing (for YOLO search)
text = ""  # what the user typed (for YOLO search)
note = ""  # message to display on the screen (for YOLO search)
model = None  # YOLO model (loaded on first search)
current_search = None  # current search result (for YOLO search)
det_box = None  # bounding box of the detected object (for YOLO search)
det_ttl = 0  # time of the last detection (for YOLO search)
# ------------------------------- functions -------------------------------


def nothing(x):
    pass


def on_mouse(event, x, y, flags, param):
    global pick_mode, click, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        x1, y1, x2, y2 = BTN
        if x1 <= x <= x2 and y1 <= y <= y2:
            pick_mode = not pick_mode  # click on the button: enable/disable
        elif pick_mode:
            click = (x, y)  # click on the image: handle in the loop


def set_color(h, s, v):
    """Двигает ползунки под выбранный цвет (одинаково для клика и для нейросети)."""
    global picked
    picked = (h, s, v)
    cv2.setTrackbarPos("H min", "mask", (h - H_TOL) % 180)
    cv2.setTrackbarPos("H max", "mask", (h + H_TOL) % 180)
    cv2.setTrackbarPos("S min", "mask", max(0, s - SV_TOL))
    cv2.setTrackbarPos("V min", "mask", max(0, v - SV_TOL))
    print(f"Selected: H={h}, S={s}, V={v}")
    if s < 40:
        print("Warning: the color is almost gray/white/black, so H is unreliable here")


def color_from_box(hsv, box):
    """Берёт средний цвет в прямоугольнике (для нейросети). Eсли обьект не найден, возвращает None."""
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    # берем центральную часть прямоугольника, чтобы не попасть на фон
    roi = hsv[by1 + bh // 4 : by2 - bh // 4, bx1 + bw // 4 : bx2 - bw // 4]
    if roi.size == 0:
        return None
    hh = roi[:, :, 0].astype(int).ravel()  # H values
    ss = roi[:, :, 1].ravel()  # S values
    vv = roi[:, :, 2].ravel()  # V values
    # # у серых/белых/чёрных пикселей оттенок H случайный, такие не считаем
    colored = (ss > 40) & (vv > 40)
    if colored.mean() < 0.15:  # меньше 15% цветных пикселей
        return None

    # Ищем оттенок, вокруг которого (±H_TOL) набралось больше всего пикселей.
    # Шкала H закольцована (0 и 179 - соседи), поэтому красный не разваливается на две половины.
    hist = np.bincount(hh[colored], minlength=180)
    window = np.convolve(
        np.pad(hist, H_TOL, mode="wrap"), np.ones(2 * H_TOL + 1), mode="valid"
    )
    h = int(np.argmax(window))

    # Окно могло встать с краю (например, если все пиксели одного оттенка),
    # поэтому двигаем его центр в середину набравшихся пикселей (тоже по кругу).
    for _ in range(2):
        off = (hh[colored] - h + 90) % 180 - 90  # смещение пикселей относительно центра окна
        h = int(round(float(h + off[np.abs(off) <= H_TOL].mean()))) % 180

    # S и V - медиана по пикселям этого оттенка
    off = (hh - h + 90) % 180 - 90
    same = colored & (np.abs(off) <= H_TOL)
    return h, int(np.median(ss[same])), int(np.median(vv[same]))

def find_by_text(prompt, frame, hsv):
    """Нейросеть ищет объект по тексту, мы берём  цвет и двигаем ползунки.
    Возвращает короткое сообщение для экрана."""
    global model, current_search, det_box, det_ttl
    try:
        if model is None:
            from ultralytics import YOLOWorld
            print(f"Loading YOLO model {MODEL_NAME}...")
            model = YOLOWorld(MODEL_NAME)
        if prompt != current_search:
            model.set_classes([prompt])
            current_search = prompt
        res = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    except ImportError as e:
        print("Не хватает библиотеки:", e)
        print("Если это ultralytics: pip install -U ultralytics")
        print('Если это clip: pip install "git+https://github.com/ultralytics/CLIP.git"')
        return "missing library, see console"
    except Exception as e:
        print("Ошибка нейросети:", e)
        return "YOLO error, see console"

    if len(res.boxes) == 0:
        return f'"{prompt}" not found'

    best = int(res.boxes.conf.argmax()) # рамка с наибольшей уверенностью
    conf = float(res.boxes.conf[best]) # уверенность
    fh, fw = frame.shape[:2] # размеры кадра
    x1, y1, x2, y2 = [int(c) for c in res.boxes.xyxy[best].tolist()] # координаты рамки
    box = (max(0, x1), max(0, y1), min(fw - 1, x2), min(fh - 1, y2)) # ограничиваем рамку размерами кадра
    det_box, det_ttl = box, 30 # рамка будет отображаться 30 кадров

    color = color_from_box(hsv, box)
    if color is None:
        return "found, but it has no color (gray/white/black)"
    set_color(*color)
    return f'found "{prompt}" (conf={conf:.2f})'


cv2.namedWindow("camera")
cv2.setMouseCallback("camera", on_mouse)

cv2.namedWindow("mask")
cv2.createTrackbar("H min", "mask", 35, 179, nothing)
cv2.createTrackbar("H max", "mask", 85, 179, nothing)
cv2.createTrackbar("S min", "mask", 100, 255, nothing)
cv2.createTrackbar("V min", "mask", 80, 255, nothing)

kernel = np.ones((5, 5), np.uint8)  # 5x5 brush for cleaning the mask

while True:
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera is not returning frames")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR -> HSV

    # --- object clicked: take its color and move the sliders ---
    if click is not None:
        height, width = hsv.shape[:2]
        x = min(max(click[0], 0), width - 1)
        y = min(max(click[1], 0), height - 1)
        click = None
        pick_mode = False

        # median over a small square instead of a single pixel - less noise
        patch = hsv[
            max(0, y - PATCH) : y + PATCH + 1, max(0, x - PATCH) : x + PATCH + 1
        ]
        h, s, v = [int(c) for c in np.median(patch.reshape(-1, 3), axis=0)]
        set_color(h, s, v)

    h_min = cv2.getTrackbarPos("H min", "mask")
    h_max = cv2.getTrackbarPos("H max", "mask")
    s_min = cv2.getTrackbarPos("S min", "mask")
    v_min = cv2.getTrackbarPos("V min", "mask")

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2  # frame center

    # 1. Mask
    if h_min <= h_max:
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)  # white = match, black = no match
    else:
        # H min > H max: hue "wraps" through 0 (common with red),
        # so we combine two ranges: [H min .. 179] and [0 .. H max]
        m1 = cv2.inRange(
            hsv, np.array([h_min, s_min, v_min]), np.array([179, 255, 255])
        )
        m2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

    # --- the button and hints are drawn on a copy of the frame; hsv and mask stay clean ---
    view = frame.copy()
    x1, y1, x2, y2 = BTN
    cv2.rectangle(
        view, (x1, y1), (x2, y2), (0, 200, 0) if pick_mode else (200, 200, 200), -1
    )
    cv2.putText(
        view,
        "PICK COLOR",
        (x1 + 12, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    if pick_mode:
        mx, my = mouse_pos
        cv2.rectangle(
            view, (mx - PATCH, my - PATCH), (mx + PATCH, my + PATCH), (255, 255, 255), 1
        )
        cv2.putText(
            view,
            "Click on the object",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    if picked is not None:
        cv2.putText(
            view,
            f"picked HSV = {picked}",
            (10, view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # строка внизу: что сейчас печатаем или чем закончился последний поиск
    line = f"search: {text}" if typing else note
    if line:
        cv2.putText(
            view,
            line,
            (10, view.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # рамка, которую нашла нейросеть (держится примерно секунду)
    if det_ttl>0:
        bx1, by1, bx2, by2 = det_box
        cv2.rectangle(view, (bx1, by1), (bx2, by2), (255, 128, 0), 2)
        det_ttl -= 1

    # 2. Remove small noise
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 3. Find separate white blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    status = "no object"
    if contours:
        biggest = max(contours, key=cv2.contourArea)  # largest blob
        if cv2.contourArea(biggest) > 300:  # ignore very small ones
            # 4. Center of mass of the blob
            m = cv2.moments(biggest)
            obj_x = int(m["m10"] / m["m00"])
            obj_y = int(m["m01"] / m["m00"])

            # 5. Error: how far the object is shifted relative to the frame center
            error_x = obj_x - center_x  # > 0: object is to the right of center
            error_y = obj_y - center_y  # > 0: object is lower than center
            status = f"dx={error_x} dy={error_y}"

            cv2.drawContours(view, [biggest], -1, (0, 255, 0), 2)
            cv2.circle(view, (obj_x, obj_y), 6, (0, 0, 255), -1)
            cv2.line(view, (center_x, center_y), (obj_x, obj_y), (0, 255, 255), 2)

    cv2.drawMarker(view, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    
    cv2.imshow("camera", view)
    cv2.imshow("mask", mask)  # now this is the cleaned mask

        # --- клавиши ---
    raw = cv2.waitKey(1) # сырой код: в разных ОС и сборках OpenCV он разный
    key = raw & 0xFF  # младший байт: для обычных клавиш это ASCII
    # Настоящий печатный символ: младший байт совпадает с кодом без служебных битов (Shift, NumLock...).
    # Так стрелки (на Linux они дают коды вроде 0xFF51, то есть «Q») и кириллица не попадут в текст.
    printable =32<=key<=127 and (raw & 0xFFFF) == key  # печатный символ
    if typing: 
        if key == 13:
            typing = False
            if text.strip():
                cv2.putText(view, "searching...", (center_x - 70, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("camera", view)
                cv2.waitKey(1)
                note = find_by_text(text.strip(), frame, hsv)
        elif key == 27:
            typing = False
            note = "search canceled"
        elif key in (8, 127):  # backspace
            text = text[:-1]
        elif printable:
            text += chr(key) # добавить печатный символ в строку
    elif key == ord("t"):
        typing = True
        text = ""
    elif key == ord("q"):
        break  # exit the loop and close the program

cap.release()
cv2.destroyAllWindows()
