import cv2
import numpy as np
from roboflow import Roboflow
from tqdm import tqdm
import os
from dotenv import load_dotenv

    

def process_frame(frame, model, liquid_level_heights, bottle_heights):
    # Convert frame to RGB (Roboflow expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference on the frame
    result = model.predict(rgb_frame, confidence=40, overlap=77).json()
    
    bottle_height = None
    liquid_level_height = None
    
    # Parse predictions
    for pred in result['predictions']:
        if pred['class'] == 'bottle' or pred['class'] == 'iv_bottle':
            bottle_height = pred['height']
            bottle_heights.append(bottle_height)
        elif pred['class'] == 'liquid_level':
            liquid_level_height = pred['height']
            liquid_level_heights.append(liquid_level_height)
    
    # Calculate percentage if both heights are found
    if bottle_height and liquid_level_height:
        return (liquid_level_height / bottle_height) * 100
    return None

def main():
<<<<<<< HEAD
    # Initialize Roboflow
    rf = Roboflow(api_key="haha")
    project = rf.workspace().project("glucose")
    model = project.version("3").model

    # Open video file
    video_path = "/content/drive/MyDrive/Colab Notebooks/video.mov"
    cap = cv2.VideoCapture(video_path)
=======
    # Load environment variables
    load_dotenv()
>>>>>>> c12ea1a (backend logic for object detection completed)
    
    # Get API key from environment variable
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in environment variables.")
        print("Please set your Roboflow API key as an environment variable.")
        return

    # Initialize Roboflow
    rf = Roboflow(api_key)
    project = rf.workspace().project("glucose")
    model = project.version("2").model

    # Open video file
    video_path = "samples/video.mov"
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please check the file path and ensure the video file exists.")
        return

    cap = cv2.VideoCapture(video_path)
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}.")
        print("This might be due to a codec issue or corrupted file.")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    output_path = 'output_video.mov'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Common codec for .mov files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize lists to store results
    percentages = []
    liquid_level_heights = []
    bottle_heights = []

    # Process video frames
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame
        percentage = process_frame(frame, model, liquid_level_heights, bottle_heights)
        if percentage:
            percentages.append(percentage)

        # Draw bounding boxes and text on frame
        result = model.predict(frame, confidence=40, overlap=77).json()
        for pred in result['predictions']:
            x1, y1, x2, y2 = pred['x'] - pred['width']/2, pred['y'] - pred['height']/2, pred['x'] + pred['width']/2, pred['y'] + pred['height']/2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, pred['class'].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    
    # Calculate and print results
    if len(bottle_heights) < 1 or len(liquid_level_heights) < 1:
        print("Either we are not able to find the bottle or liquid level, or there is some error in detection.")
    elif len(bottle_heights) >= 1 and len(liquid_level_heights) < 1:
        print("Bottle detected, but the liquid level is either empty, not detected, or has insufficient data.")
    else:
        if percentages:
            avg_percentage = np.mean(percentages)
            print(f"Average liquid level percentage: {avg_percentage:.2f}%")
        else:
            print("No valid liquid level percentages were detected in the video.")

if __name__ == "__main__":
    main()
<<<<<<< HEAD
=======

>>>>>>> c12ea1a (backend logic for object detection completed)
