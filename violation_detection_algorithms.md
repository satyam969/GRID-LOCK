# TrafficGuard AI: Violation Detection Algorithms

This document outlines the precise algorithms, models, and heuristics used by the TrafficGuard AI engine to detect traffic violations.

## 1. Illegal Parking 🅿️
**Detection Type:** Heuristic (Geometry + Proximity)
**Models Used:** YOLOv8n (General Object Detector)
**Approach:**
We intentionally avoid training a dedicated "parked vs moving" model, as single-frame images lack temporal context (we don't know if a car is stopped at a light or parked). Instead, we use two highly efficient geometric strategies targeting **Cars, Buses, and Trucks** (Motorcycles are excluded to prevent mass false positives):
1. **No-Parking Zones (Strategy 1):** We calculate the geometric center `(x, y)` of the vehicle's bounding box. If this center falls within an admin-configured rectangular "No-Parking Zone", it is flagged immediately.
2. **Road-Edge Proximity Heuristic (Strategy 2):** If no zones are configured, the system checks if the vehicle is sitting at the extreme edges of the camera frame (outer 15% on the left or right). If it is at the edge, the system calculates the Euclidean distance between the vehicle and all detected `person` classes. If **no person** is found within 120 pixels, the vehicle is assumed to be unoccupied and illegally parked.

## 2. Helmet Non-Compliance ⛑️
**Detection Type:** Cascaded Deep Learning (Detection + Pose + Classification)
**Models Used:** YOLOv8n (General), YOLOv8n-Pose (Skeletal), Custom INT8 Helmet Classifier
**Approach:**
1. **Rider Association:** The general detector finds motorcycles and persons. The Pose model extracts 17 skeletal keypoints for each person. We check if the person's **hip keypoints** fall inside the motorcycle's bounding box to confirm they are *riding*, not just standing nearby.
2. **Crop & Classify:** Once a rider is confirmed, we crop the top 40% of the rider's bounding box (where the head is located).
3. **Inference:** This cropped image is passed to a custom quantized INT8 classifier which outputs a binary prediction: `helmet` or `no_helmet`.

## 3. Seatbelt Non-Compliance 🚗
**Detection Type:** Cascaded Deep Learning (Detection + Classification)
**Models Used:** YOLOv8n (General), Custom INT8 Seatbelt Classifier
**Approach:**
1. **Vehicle Detection:** The system detects enclosed vehicles (Cars, Trucks, Buses).
2. **Windshield Cropping:** The engine dynamically crops the top 45% of the vehicle's bounding box. This heuristic reliably captures the windshield and front-seat area regardless of the vehicle type.
3. **Inference:** The cropped windshield image is fed into the custom quantized INT8 classifier to detect the presence of a diagonal seatbelt strap.

## 4. Triple Riding 🏍️
**Detection Type:** Deep Learning + Geometric Mapping
**Models Used:** YOLOv8n (General), YOLOv8n-Pose (Skeletal)
**Approach:**
1. **Person-to-Bike Mapping:** Similar to helmet detection, the system identifies all motorcycles and all persons in the frame.
2. **Hip Validation:** Using the Pose model, we map the coordinates of every person's hips. If a person's hips are physically inside the bounding box of a specific motorcycle, they are assigned to that motorcycle.
3. **Thresholding:** We simply count the number of validated riders assigned to a single motorcycle. If `rider_count >= 3`, a Triple Riding violation is triggered.

## 5. Red-Light Violation 🔴
**Detection Type:** Deep Learning + Computer Vision (HSV) + Geometry
**Models Used:** YOLOv8n (General)
**Approach:**
1. **Traffic Light Detection:** YOLOv8n detects traffic lights (COCO class 9) in the frame.
2. **Color State Extraction (HSV):** The traffic light bounding box is cropped and converted to the HSV color space. We isolate the top 1/3rd of the traffic light box and apply an HSV threshold mask for Red colors (wrapping around the 0-10 and 160-180 Hue ranges). If more than 5% of the pixels in that zone are red, the light state is locked as `RED`.
3. **Stop-Line Check:** If the light is `RED`, we check the bottom `y2` coordinate (the front bumper) of all approaching vehicles. If a vehicle's bumper exceeds the configured `stop_line_y` threshold, a Red Light Violation is recorded.

## 6. Stop-Line Violation 🛑
**Detection Type:** Geometric Thresholding
**Models Used:** YOLOv8n (General)
**Approach:**
This uses the exact same geometric logic as the Red-Light violation, but operates independently of the traffic light state. A horizontal line is virtually drawn across the frame (e.g., at 65% of the image height). If any vehicle's bottom bounding box edge `y2` is strictly greater than this line's Y-coordinate, it has encroached on the pedestrian crossing / stop line.

## 7. Wrong-Side Driving 🔄
**Detection Type:** Directional Flow Heuristic
**Models Used:** YOLOv8n (General)
**Approach:**
*(Note: A true production implementation requires Optical Flow across video frames, but for single-image processing we use a spatial heuristic).*
If the configured traffic flow for a specific camera is "keep right", the system bisects the image vertically. If a vehicle is detected traveling or positioned predominantly in the left half of the frame (facing oncoming traffic angles), it is flagged. 
