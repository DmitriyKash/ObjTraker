import cv2
import numpy as np

# Шляхи до файлів моделі YOLO
modelConfiguration = "yolov3-spp.cfg"
modelWeights = "yolov3-spp.weights"
classesFile = "coco.names"

# Завантаження назв класів
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Налаштування мережі YOLO
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Функція для отримання висновків від моделі YOLO
def get_outputs(image):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    return outputs


# Ініціалізація змінних
ix, iy, w, h = -1, -1, -1, -1
drawing = False
tracker = None
selected_region = None
tracking = False


# Callback-функція для обробки нажатий кнопок мыши
def mouse_callback(event, x, y, flags, param):
    global ix, iy, w, h, drawing, tracker, selected_region, tracking

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        w, h = 0, 0
        drawing = True
        tracking = False
        tracker = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x - ix, y - iy

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if w > 0 and h > 0:
            tracker = cv2.TrackerMIL_create()
            bbox = (ix, iy, w, h)
            ok = tracker.init(frame, bbox)
            selected_region = frame[iy:iy + h, ix:ix + w].copy()
            tracking = True
            process_selected_region(selected_region)


def process_selected_region(region):
    class_label, confidence = "", 0.0
    if region is not None and region.shape[0] > 0 and region.shape[1] > 0:
        outputs = get_outputs(region)

        # Перебір кожного виявлення
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    class_label = classes[class_id]
                    break  # припиняємо цикл після першого виявлення

    return class_label, confidence


# Включення відеозахвату
cap = cv2.VideoCapture(0)

# Налаштування callback-функції
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    # Захват кадру з відеопотоку
    ret, frame = cap.read()
    if not ret:
        break

    # Відстеження об'єкта
    if tracking:
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Отримання інформації про об'єкт та її відображення
            class_label, confidence = process_selected_region(frame[y:y + h, x:x + w])
            if class_label:
                label = '%s: %.2f' % (class_label, confidence)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Рисування квадрату на кадрі
    if drawing and w > 0 and h > 0:
        cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 0), 2)

    # Відображення кадру з виділеною областю
    cv2.imshow("Frame", frame)

    # Вихід з циклу при натисканні клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
