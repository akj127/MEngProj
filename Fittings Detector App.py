import os
import cv2
import torch
import csv
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyperclip
import json

def create_capture_folder():
    folder_name = "Captured_Images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Created folder: {folder_name}')

class ImageCaptureApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Detect Fittings")
        self.root.geometry("1280x720")

        # Frame to hold the buttons horizontally
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.capture_button = ttk.Button(self.button_frame, text="Click Image", command=self.capture_image)
        self.capture_button.pack(side="left", padx=5)

        self.flag_button = ttk.Button(self.button_frame, text="Flag Image", command=self.flag_image)
        self.flag_button.pack(side="left", padx=5)
        self.flag_button.config(state=tk.DISABLED)  # Disabled by default

        self.calibrate_button = ttk.Button(self.button_frame, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack(side="left", padx=5)

        self.exit_button = ttk.Button(self.button_frame, text="Exit App", command=self.exit_app)
        self.exit_button.pack(side="left", padx=5)

        # Frame to hold the camera feed and detected image side by side
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)

        # Initialize camera
        self.cap = cv2.VideoCapture(1)  # 0 for built-in camera, 1 or 2 if camera is connected via USB
        self.camera_feed_label = tk.Label(self.image_frame)
        self.camera_feed_label.pack(side="left", padx=10)

        # Label to display the last detected image
        self.detected_image_label = tk.Label(self.image_frame)
        self.detected_image_label.pack(side="left", padx=10)

        # Entry to allow editing detected items
        self.detected_items_entry = tk.Entry(root, width=100)
        self.detected_items_entry.pack(pady=5)

        # Placeholder for calibration points
        self.calibration_file = 'calibration.json'
        self.calibration_points = self.load_calibration_points()

        self.update_camera_feed()

        # Placeholder to store the last captured image path and filename
        self.last_captured_image_path = None
        self.last_captured_image_filename = None

    def create_detections_csv(self):
        csv_file_path = os.path.join("Captured_Images", "Detections.csv")
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Image File Name", "Detected Items", "Edited Detected Items", "Flagged"])
        return csv_file_path

    def append_to_detections_csv(self, csv_file, image_filename, detected_items, edited_detected_items="", flagged="No"):
        with open(csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([image_filename, detected_items, edited_detected_items, flagged])

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))  # Resize for display

            # Draw rectangle on live feed if calibration points exist
            if self.calibration_points:
                frame_with_rect = self.draw_rectangle(frame.copy())
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_with_rect))
            else:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

            self.camera_feed_label.configure(image=self.photo)
            self.camera_feed_label.image = self.photo

            self.root.after(10, self.update_camera_feed)

    def draw_rectangle(self, frame):
        rect = np.array(self.calibration_points, dtype="float32")
        # cv2.polylines(frame, [rect.astype(np.int32)], True, (0, 255, 0), 2)
        return frame

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_filename = f'capture_img_{timestamp}.jpg'
            image_path = os.path.join("Captured_Images", image_filename)
            cv2.imwrite(image_path, frame)
            print(f'Image saved as: {image_path}')

            # Transform the image using the saved calibration points
            if self.calibration_points:
                transformed_image = self.transform_image(image_path)
                transformed_image_with_rect = self.draw_rectangle(transformed_image)
            else:
                transformed_image = frame
                transformed_image_with_rect = frame

            # Detect objects in the transformed image
            detected_image, detected_items = self.detect_objects(transformed_image)

            # Display the detected image in the GUI
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            detected_image_pil = Image.fromarray(detected_image_rgb)
            self.detected_image_photo = ImageTk.PhotoImage(image=detected_image_pil)
            self.detected_image_label.configure(image=self.detected_image_photo)
            self.detected_image_label.image = self.detected_image_photo

            # Populate the entry box with detected items
            self.detected_items_entry.delete(0, tk.END)
            self.detected_items_entry.insert(0, detected_items)

            # Append detected items to the CSV file
            csv_file = self.create_detections_csv()
            self.append_to_detections_csv(csv_file, image_filename, detected_items)

            self.last_captured_image_path = image_path  # Store the last captured image path
            self.last_captured_image_filename = image_filename  # Store the last captured image filename
            self.flag_button.config(state=tk.NORMAL)  # Enable flag button

    def transform_image(self, image_path):
        image = cv2.imread(image_path)
        rect = np.array(self.calibration_points, dtype="float32")
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def detect_objects(self, image, confidence_threshold=0.5):
        # Model Location, Weights Location
        model = torch.hub.load('D:\MEngProj\yolov5', 'custom', path='D:\MEngProj\Latest_Weights.pt', source='local')
        model.eval()

        results = model(image)

        detected_image = results.render()[0]  # Use the detected image from the model
        detected_items = []

        for detection in results.pred[0]:
            *box, confidence, class_index = detection.tolist()
            class_index = int(class_index)
            confidence = float(confidence)

            if confidence >= confidence_threshold:
                label = model.names[class_index]
                detected_items.append(f"{label}")

        detected_items = ', '.join(detected_items)
        pyperclip.copy(detected_items)  # Copy to clipboard

        return detected_image, detected_items

    def calibrate(self):
        ret, frame = self.cap.read()
        if ret:
            calibration_window = tk.Toplevel(self.root)
            calibration_window.title("Calibration")
            calibration_window.geometry("800x600")

            calibration_canvas = tk.Canvas(calibration_window, width=800, height=600)
            calibration_canvas.pack()

            self.calibration_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            calibration_canvas.create_image(0, 0, anchor=tk.NW, image=self.calibration_image)

            self.calibration_points = []

            def click_event(event):
                if len(self.calibration_points) < 4:
                    self.calibration_points.append([event.x, event.y])
                    calibration_canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="red")
                    if len(self.calibration_points) == 4:
                        with open(self.calibration_file, 'w') as f:
                            json.dump(self.calibration_points, f)
                        print("Calibration points saved.")

            calibration_canvas.bind("<Button-1>", click_event)

    def load_calibration_points(self):
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                points = json.load(f)
            print(f"Calibration points loaded: {points}")
            return points
        return None

    def flag_image(self):
        if self.last_captured_image_filename:
            csv_file = self.create_detections_csv()
            # Read the existing CSV file and update the flagged status
            rows = []
            with open(csv_file, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                headers = next(csv_reader)
                for row in csv_reader:
                    if row[0] == self.last_captured_image_filename:
                        row[2] = self.detected_items_entry.get()  # Update edited detected items with user input
                        row[3] = "Yes"
                    rows.append(row)

            # Write back the updated rows to the CSV file
            with open(csv_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(headers)
                csv_writer.writerows(rows)
            
            print(f'Image {self.last_captured_image_filename} flagged in CSV.')
            self.flag_button.config(state=tk.DISABLED)  # Disable flag button after flagging

    def exit_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    create_capture_folder()
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()
