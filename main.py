import cv2

# Инициализация переменных
ix, iy, w, h = -1, -1, -1, -1
drawing = False
tracker = None


# Callback-функция для обработки нажатий кнопок мыши
def mouse_callback(event, x, y, flags, param):
    """
        Callback-функция для обработки нажатий кнопок мыши.

        При нажатии левой кнопки мыши инициализирует начальные координаты прямоугольника.
        При движении мыши с зажатой левой кнопкой обновляет размеры прямоугольника.
        При отпускании кнопки завершает рисование прямоугольника и инициализирует трекер.

        Аргументы:
        event: Тип события мыши.
        x, y: Координаты курсора мыши.
        flags: Дополнительные флаги, передаваемые OpenCV.
        param: Дополнительные параметры (не используются).
        """
    global ix, iy, w, h, drawing, tracker, selected_region

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        w, h = 0, 0
        drawing = True
        tracker = None

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if w > 0 and h > 0:

            tracker = cv2.TrackerMIL_create()
            bbox = (ix, iy, w, h)
            ok = tracker.init(frame, bbox)
            selected_region = frame[iy:iy + h, ix:ix + w].copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x - ix, y - iy


# Включение видеозахвата
cap = cv2.VideoCapture(0)

# Установка callback-функции
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    """
        Основной цикл захвата видео.

        В этом цикле происходит чтение кадров с камеры и обработка событий мыши.
        В случае активации трекера, он обновляет положение прямоугольника.
        При нажатии клавиши 'q' цикл завершается и закрываются все окна.
        """
    # Захват кадра с видеопотока
    ret, frame = cap.read()

    if tracker:
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Рисование квадрата на кадре
    if drawing and w > 0 and h > 0:

        cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 0), 2)

    # Отображение кадра с выделенной областью
    cv2.imshow("Frame", frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Сохранение выбранной области в файл

# if selected_region is not None:
#     cv2.imwrite('selected_region.png', selected_region)

cap.release()
cv2.destroyAllWindows()
