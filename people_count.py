import torch
import cv2
import math

model = torch.hub.load('./yolov5/', 'custom', path='./yolov5/runs/train/head_custom_v2_50it/weights/best.pt', source='local')  # custom model

# Initialize video capture
#cap = cv2.VideoCapture('./people_joining_bus.mp4')
cap = cv2.VideoCapture('./people_leaving_bus.mp4')

# Counting line position
count_value = 100

# Tracking variables
head_id = 0
tracked_heads = {}  # Dictionary to store tracked person
crossed_ids = set()   # Set to store IDs of people who crossed the line

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw counting line
    cv2.line(frame, (0, count_value), (frame.shape[1], count_value), (255, 255, 0), 2)

    # Extract detections
    head_detections = []
    for head in results.xyxy[0]:
        xA, yA, xB, yB = map(int, head[:4])  # Bounding box coordinates
        centroid_x = int((xA + xB) / 2)
        centroid_y = int((yA + yB) / 2)
        head_detections.append((centroid_x, centroid_y, xA, yA, xB, yB))

    # Update tracked 
    new_tracked_heads = {}
    for detection in head_detections:
        centroid_x, centroid_y, xA, yA, xB, yB = detection
        # Find if this detection matches a previously tracked person
        matched_id = None
        for hid, (last_x, last_y) in tracked_heads.items():
            # Compute Euclidean distance
            distance = math.sqrt((last_x - centroid_x) ** 2 + (last_y - centroid_y) ** 2)
            if distance < 100:  # 100 is the distance threshold
                matched_id = hid
                min_distance = distance

        # New head detected
        if matched_id is None:  
            head_id += 1
            matched_id = head_id

        new_tracked_heads[matched_id] = (centroid_x, centroid_y)

        # If this head has been tracked before, use their last known position
        if matched_id in tracked_heads:
            last_x, last_y = tracked_heads[matched_id]
            # Check if the head crosses the counting line
            print('id:'+str(matched_id)+':'+str(last_y))
            if matched_id not in crossed_ids and count_value - 5 < last_y < count_value + 5:
                crossed_ids.add(matched_id)

        # Draw bounding box and ID
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.circle(frame, (centroid_x, centroid_y), 2, (0, 255, 255), -1)
        cv2.putText(frame, f"ID: {matched_id}", (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update tracked heads
    tracked_heads = new_tracked_heads

    # Display count
    person_count = len(crossed_ids)
    cv2.putText(frame, f"Count: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 123, 10), 2)

    # Show frame
    cv2.imshow("People Counting", frame)

    key = cv2.waitKey(20)
    if key & 0xFF == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
