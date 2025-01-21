import cv2
import numpy as np
import onnxruntime as ort

input_video_path = "test_video.MP4"
output_video_path = "output_video.mp4"
model_path = "yolov8n.onnx"
window_size = (384, 640)
step_size = (192, 320)
confidence_threshold = 0.5
iou_threshold = 0.4

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def preprocess(image, input_size=(640, 640)):
    image_resized = cv2.resize(image, input_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    return np.expand_dims(image_transposed, axis=0).astype(np.float32)

def nms(boxes, scores, iou_threshold):
    indices = []
    if len(boxes) == 0:
        return indices

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        indices.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        if areas[i] == 0 or np.any(areas[order[1:]] == 0):
            continue
        iou = np.where(areas[i] + areas[order[1:]] - inter > 0, inter / (areas[i] + areas[order[1:]] - inter), 0)

        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return indices

def process_frame(frame):
    h, w, _ = frame.shape
    win_h, win_w = window_size
    step_h, step_w = step_size

    output_frame = frame.copy()
    all_boxes = []
    all_scores = []

    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            x1, y1 = x, y
            x2, y2 = min(x + win_w, w), min(y + win_h, h)
            window = frame[y1:y2, x1:x2]

            input_tensor = preprocess(window)
            outputs = session.run([output_name], {input_name: input_tensor})[0]
            for detection in outputs[0]:
                x_min, y_min, x_max, y_max, conf, cls = detection[:6]
                if conf < confidence_threshold:
                    continue

                x_min = int(x_min * (x2 - x1) + x1)
                y_min = int(y_min * (y2 - y1) + y1)
                x_max = int(x_max * (x2 - x1) + x1)
                y_max = int(y_max * (y2 - y1) + y1)

                all_boxes.append([x_min, y_min, x_max, y_max])
                all_scores.append(conf)

    selected_indices = nms(all_boxes, all_scores, iou_threshold)

    for i in selected_indices:
        x_min, y_min, x_max, y_max = all_boxes[i]
        conf = all_scores[i]

        cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 7)
        cv2.putText(
            output_frame,
            f"Conf: {conf:.2f}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return output_frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = current_frame / fps
    if current_time > 30:
        break

    processed_frame = process_frame(frame)
    out.write(processed_frame)

cap.release()
out.release()
print(f"saved in {output_video_path}")
