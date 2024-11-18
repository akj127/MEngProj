import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox

# Global variables
start_point = None
end_point = None
cropping = False
image = None
labels = []
label_names = []

LABELS_FILE = "labels.txt"

# Function to load predefined labels from file
def load_labels_from_file():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

# Function to save labels to file
def save_labels_to_file():
    with open(LABELS_FILE, "w") as f:
        for label in label_names:
            f.write(f"{label}\n")

# Function to draw bounding boxes
def click_and_crop(event, x, y, flags, param):
    global start_point, end_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False

        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Image", image)

        if selected_label.get():
            labels.append((selected_label.get(), start_point, end_point))

# Function to save annotations in YOLOv5 format
def save_annotations(image_path, labels, image_shape):
    h, w = image_shape[:2]
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    with open(f"{file_name}.txt", "w") as f:
        for label, (x1, y1), (x2, y2) in labels:
            x_center = (x1 + x2) / 2.0 / w
            y_center = (y1 + y2) / 2.0 / h
            bbox_width = (x2 - x1) / w
            bbox_height = (y2 - y1) / h
            f.write(f"{label_names.index(label)} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Function to start capturing images
def start_capture(image_path = None):
    global image, labels

    if image_path:
        image = cv2.imread(image_path)
        process_image(image, image_path)
    else:
        cap = cv2.VideoCapture(0)

        if not os.path.exists("images"):
            os.makedirs("images")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Press 'c' to capture image, 'q' to quit", frame)
            key = cv2.waitKey(1)

            if key == ord('c'):
                image_name = f"images/image_{len(os.listdir('images')) + 1}.jpg"
                cv2.imwrite(image_name, frame)
                image = frame.copy()

                cv2.namedWindow("Image")
                cv2.setMouseCallback("Image", click_and_crop)

                while True:
                    cv2.imshow("Image", image)
                    key = cv2.waitKey(1)

                    if key == ord('s'):
                        save_annotations(image_name, labels, image.shape)
                        messagebox.showinfo("Info", f"Image and labels saved as {image_name} and {image_name}.txt")
                        labels.clear()
                        break

                    elif key == ord('r'):
                        image = frame.copy()
                        labels.clear()

                    elif key == ord('q'):
                        break

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def process_image(image, image_path):
    global labels
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_and_crop)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)

        if key == ord('s'):
            save_annotations(image_path, labels, image.shape)
            messagebox.showinfo("Info", f"Image and labels saved as {image_path} and {image_path}.txt")
            labels.clear()
            break

        elif key == ord('r'):
            image = cv2.imread(image_path)
            labels.clear()

        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

# Function to add a new label
def add_label():
    global label_entry, label_dropdown,selected_label
    new_label = label_entry.get()
    if new_label and new_label not in label_names:
        label_names.append(new_label)
        label_dropdown['values'] = label_names
        selected_label.set(new_label)
        save_labels_to_file()
        label_entry.delete(0, tk.END)

def create_win(parent=None, image_path=None):

    global label_entry, label_dropdown,selected_label, image, label_names
    # Create the main window
    # root = tk.Tk()
    # root.title("Image Labeling Tool")
    if parent:
        top = tk.Toplevel(parent)
    else:
        top = tk.Tk()
    top.title("Image Labeling Tool")

    # Create the Tkinter StringVar after the root window is created
    selected_label = tk.StringVar()

    # Load labels at startup
    label_names = load_labels_from_file()

    # Dropdown menu for label selection
    label_frame = tk.Frame(top)
    label_frame.pack(pady=10)

    label_label = tk.Label(label_frame, text="Select or Add Label:")
    label_label.pack(side=tk.LEFT)

    label_dropdown = ttk.Combobox(label_frame, textvariable=selected_label)
    label_dropdown['values'] = label_names
    label_dropdown.pack(side=tk.LEFT)

    # Entry field to add new labels
    label_entry = tk.Entry(label_frame)
    label_entry.pack(side=tk.LEFT)

    # Button to add a new label
    add_button = tk.Button(label_frame, text="Add Label", command=add_label)
    add_button.pack(side=tk.LEFT)

    # Start capture button
    start_button = tk.Button(top, text="Start Capture", command=lambda: start_capture(image_path))
    start_button.pack(pady=20)

    # if image_path:
    #     image = cv2.imread(image_path)
    #     if image is not None:
    #         cv2.imshow("Image", image)
    #         cv2.setMouseCallback("Image", click_and_crop)
    # If `parent` is not provided, start the main loop
    # if not parent:
    top.mainloop()

if __name__== '__main__' :
    create_win(None, None)