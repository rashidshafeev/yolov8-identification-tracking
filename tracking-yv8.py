import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import easyocr
import requests
from datetime import datetime
import json

reader = easyocr.Reader(['en'], gpu=False)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)
model = YOLO("yolov8l.pt")
det = 0

def send_to_server(frame_result, object_info):
    url = 'https://bot-adcanimal.host2bot.ru/api/data/post'
    headers = {'Content-Type': 'application/json'}
    current_time = datetime.now().strftime("%H:%M:%S")
    data = dict(token='DJ97KS', time = current_time)
    data['count'] = len(frame_result)
    for i, item in enumerate(frame_result):
        data[f'text{i}'] = f'{item[1]}'
        data[f'text{i}-confidence'] = f'{item[2]}'
        data[f'text{i}-top-left'] = f'{item[0][0][0]},{item[0][0][1]}'
        data[f'text{i}-bottom-right'] = f'{item[0][2][0]},{item[0][2][1]}'
    final_data = {**data, **object_info}
    r = requests.post(url, data = final_data)
    print(r.request.url)
    print(r.request.body)
    print(r.request.headers)
    print(r.text)

def process_frame(frame):
    result = reader.readtext(frame, allowlist = '0123456789')
    print(result)
    for res in result:
        top_left = tuple([round(res[0][0][0]), round(res[0][0][1])])
        bottom_right = tuple([round(res[0][2][0]), round(res[0][2][1])]) 
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2) 
        cv2.putText(frame, res[1], (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 1)
    processed_frame = frame
    return processed_frame, result


for result in model.track(source="rtsp://192.168.123.2:8554/proxied1", show=True, stream=True, agnostic_nms=True):
    
    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[detections.class_id == 0]

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]


    for i, val in enumerate(detections.xyxy):
        obj = frame[round(detections.xyxy[i][1]):round(detections.xyxy[i][3]), round(detections.xyxy[i][0]):round(detections.xyxy[i][2])]
        processed_frame, ocr_result = process_frame(obj)
        cv2.imwrite(f'object{det}-{i}.jpg', processed_frame)
        results_dict = {
            'tracker_id' : detections.tracker_id[i],
            'class_id' : detections.class_id[i],
            'confidence' : detections.confidence[i],
            'object-top-left' : f'{detections.xyxy[i][0]},{detections.xyxy[i][1]}',
            'object-bottom-right' : f'{detections.xyxy[i][2]},{detections.xyxy[i][3]}'
        }
        send_to_server(ocr_result, results_dict)

    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections,
        labels=labels
    )
    

    cv2.imwrite(f'detected{det}.jpg', frame)
    det += 1
