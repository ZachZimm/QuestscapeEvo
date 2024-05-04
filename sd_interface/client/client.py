import asyncio
import sd_front
import numpy as np
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import Button
import webcam

async def main():
    id = 1
    denoise = 0.5
    cn_end_percent = 0.75
    cn_strength = 0.75
    landmark_type = ""
    theme_prompt = ""
    custom_prompt = ""
    checkpoint = "crossroads_v10AlphaNoVae.safetensors"
    mode = "very_fast_canny"
    keep_background = True

    use_ssl = False
    # api_url = "host.zzimm.com"
    api_url = "localhost"
    # api_url = "lab"
    # port = ":443"
    port = ":8080"
    endpoint = "/generate-images/"
    frame = np.zeros((512, 512, 3), np.uint8)  # RGB
    frame = Image.fromarray(frame)
    frames = 1
    generation_config = {
                'id': id,
                'sub_id': frames,
                'denoise': denoise,
                'cn_end_percent': cn_end_percent,
                'cn_strength': cn_strength,
                'landmark_type': landmark_type,
                'theme_prompt': theme_prompt,
                'custom_prompt': custom_prompt,
                'checkpoint': checkpoint,
                'mode': mode,
                'keep_background': keep_background
            }

    if use_ssl:
        protocol = "https://"
    else:
        protocol = "http://"

    # frame_dir = "../imports/webcam_image.png"
    # frame = Image.open(frame_dir)
    cam = webcam.Webcam()
    print("Created webcam object.")
    frame = cam.get_frame()
    # frame = Image.fromarray(frame)

    await sd_front.request_generated_frame(frame=frame, 
                            api_url=protocol+api_url+port+endpoint, 
                            generation_config=generation_config)


import cv2
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

cam = webcam.Webcam()
# Function to handle the webcam and update the GUI
def update_frame():
    ret, frame = cam.get_frame()

    if ret:
        # Resize the frame to ensure it's at least 512x512
        if frame.shape[0] < 512 or frame.shape[1] < 512:
            frame = cv2.resize(frame, (512, 512))
        
        # Crop the center 512x512
        height, width = frame.shape[:2]
        startx = width//2 - 256
        starty = height//2 - 256
        cropped_frame = frame[starty:starty+512, startx:startx+512]

        # Convert to a format tkinter can use
        # cv_img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    lmain.after(10, update_frame)  # Refresh the frame every 10 milliseconds

# Function to capture and save the image
def take_snapshot():
    ret, frame = cap.read()
    if ret:
        # Crop and save as before
        height, width = frame.shape[:2]
        startx = width//2 - 256
        starty = height//2 - 256
        cropped_frame = frame[starty:starty+512, startx:startx+512]
        # cv2.imwrite("snapshot.png", cropped_frame)
        cv2.imwrite("snapshot.png", frame)
        print("Image saved as snapshot.png")

# Setup the main window
root = tk.Tk()
root.title("Webcam")

# Create a label in the window to hold the video frames
lmain = tk.Label(root)
lmain.pack()

# Button to take a snapshot
btn_snapshot = Button(root, text="Take Snapshot", width=50, command=take_snapshot)
btn_snapshot.pack(anchor=tk.CENTER, expand=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the ID for the default webcam
if not cap.isOpened():
    raise ValueError("Unable to open webcam")

# Start the video process
update_frame()

# Start the GUI
root.mainloop()

# Release the camera when closing the app
cap.release()
cv2.destroyAllWindows()