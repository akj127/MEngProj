import os
import cv2
import torch
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import pyperclip
from collections import Counter

def create_capture_folder():
        folder_name = "Captured Images"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f'Created folder: {folder_name}')

class ImageCaptureApp:


    def __init__(self, root):
        self.root = root
        self.root.title("Detect Fittings")
        self.root.geometry("700x700")

        self.capture_button = ttk.Button(root, text="Click Image", command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.exit_button = ttk.Button(root, text="Exit App", command=self.exit_app)
        self.exit_button.pack()

        # Initialize camera
        self.cap = cv2.VideoCapture(0) # 0 for built-in camera, 1 or 2 if camera is connected via USB
        self.camera_feed_label = tk.Label(root)
        self.camera_feed_label.pack()

        self.update_camera_feed()



    def create_detections_csv(self):
        csv_file_path = os.path.join("Captured Images", "Detections.csv")
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Image File Name", "Detected Items"])
        return csv_file_path

    def append_to_detections_csv(self, csv_file, image_filename, detected_items):
        with open(csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([image_filename, detected_items])

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))  # Resize for display

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.camera_feed_label.configure(image=self.photo)
            self.camera_feed_label.image = self.photo

            self.root.after(10, self.update_camera_feed)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_filename = f'capture_img_{timestamp}.jpg'
            image_path = os.path.join("Captured Images", image_filename)
            cv2.imwrite(image_path, frame)
            print(f'Image saved as: {image_path}')

            # Detect objects in the captured image
            detected_image, detected_items = self.detect_objects(image_path)

            # Display the detected image
            cv2.imshow('Objects Detected', detected_image)
            cv2.waitKey(0)

            # Save the detected image with the captured image timestamp
            detected_image_filename = f'detected_img_{timestamp}.jpg'
            detected_image_path = os.path.join("Captured Images", detected_image_filename)
            cv2.imwrite(detected_image_path, detected_image)
            print(f'Detected image saved as: {detected_image_path}')

            # Append detected items to the CSV file
            csv_file = self.create_detections_csv()
            self.append_to_detections_csv(csv_file, image_filename, detected_items)

    def detect_objects(self, image_path, confidence_threshold=0.5):

        #Model Location, Weights Location 
        #Format: model = torch.hub.load('D:\Woleseley\yolov5', 'custom', path="D:\Woleseley\Latest_Weights.pt", source='local') 
        model = torch.hub.load('D:\MEngProj\yolov5', 'custom', path='D:\MEngProj\Latest_Weights.pt', source='local')
        model.eval()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model(image)

        detected_image = results.render()[0]
        detected_items = []

        for detection in results.pred[0]:
            *box, confidence, class_index = detection.tolist()
            class_index = int(class_index)
            confidence = float(confidence)

            if confidence >= confidence_threshold:
                xmin, ymin, xmax, ymax = [int(coord) for coord in box]
                label = model.names[class_index]
                detected_items.append(f"{label}")


        detected_items = ', '.join(detected_items)
        pyperclip.copy(detected_items) #Copy to clipboard


        return detected_image, detected_items

    def exit_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    create_capture_folder()
    import PIL.Image, PIL.ImageTk
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()
