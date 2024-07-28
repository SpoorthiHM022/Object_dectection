import cv2
from ultralytics import YOLO
import numpy as np
import os
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import google.auth.transport.requests
import time

# Function to check if a point is inside a polygon using the ray casting algorithm
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# Function to send an email using the Gmail API
def send_email(subject, message_text, attachment_path=None):
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    creds = None

    # Load credentials from file
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)

    message = MIMEMultipart()
    message['to'] = 'khushangowdagh@gmail.com'
    message['from'] = 'swayammm04@gmail.com'
    message['subject'] = subject

    message.attach(MIMEText(message_text, 'plain'))

    if attachment_path:
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
            message.attach(part)

    encoded_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    try:
        send_message = (service.users().messages().send(userId="me", body=encoded_message).execute())
        print(f'Message Id: {send_message["id"]}')
    except Exception as e:
        print(f'An error occurred: {e}')

# Function to detect persons with a hexagon area
def detect_persons_with_hexagon(video_path, output_path, scale=0.5):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Set up video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Set up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Define the coordinates of the hexagon
    hexagon_points = np.array([[250, 260], [600, 280], [810, 320], [250, 430], [100, 410], [160, 290]], np.int32)
    hexagon_points = hexagon_points.reshape((-1, 2))

    # Initialize the last email sent time
    last_email_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Run YOLOv8 inference on the resized frame for person detection
            results = model(resized_frame)

            persons_in_hexagon = 0

            # Extract bounding boxes and labels for persons
            for result in results[0].boxes:
                x1, y1, x2, y2 = result.xyxy[0]  # Bounding box coordinates
                conf = result.conf[0]  # Confidence score
                cls = result.cls[0]  # Class label
                if cls == 0:  # class 0 corresponds to 'person' in COCO
                    # Draw bounding box and label
                    cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(resized_frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Check if the center of the bounding box is inside the hexagon
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    if is_point_in_polygon((center_x, center_y), hexagon_points):
                        persons_in_hexagon += 1

            # Draw the hexagon on the frame
            cv2.polylines(resized_frame, [hexagon_points], isClosed=True, color=(0, 0, 255), thickness=2)

            # Display the count of persons in the hexagon
            cv2.putText(resized_frame, f'Number of persons: {persons_in_hexagon}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display alert message if any person is in the hexagon and send email every 15 seconds
            if persons_in_hexagon > 0:
                current_time = time.time()
                if current_time - last_email_time >= 5:
                    cv2.putText(resized_frame, 'Alert: Person in pond!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # Save the frame as an image
                    alert_image_path = 'alert_image.jpg'
                    cv2.imwrite(alert_image_path, resized_frame)
                    send_email('Alert: Person Detected in Pond', f'{persons_in_hexagon} person(s) detected in the pond area.', alert_image_path)
                    last_email_time = current_time

            # Write the processed frame to the output video
            out.write(resized_frame)

            # Display the frame
            cv2.imshow('Person Detection with Hexagon', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define paths
video_path = 'dataset.mp4'
output_path = 'output.mp4'

# Run person detection with resizing and hexagon drawing
detect_persons_with_hexagon(video_path, output_path, scale=0.5)
