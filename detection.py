import cv2
import numpy as np
from roboflow import Roboflow
from tqdm import tqdm

def process_frame(frame, model,liquid_level_heights, bottle_heights):
    # Convert frame to RGB (Roboflow expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference on the frame
    result = model.predict(rgb_frame, confidence=40, overlap=77).json()
    
    bottle_height = None
    liquid_level_height = None
    
    # Parse predictions
    for pred in result['predictions']:
        if pred['class'] =='bottle' or pred['class'] == 'iv_bottle':
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
    # Initialize Roboflow
    rf = Roboflow(api_key="haha")
    project = rf.workspace().project("glucose")
    model = project.version("3").model

    # Open video file
    video_path = "/content/drive/MyDrive/Colab Notebooks/video.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    percentages = []
    liquid_level_heights = []
    bottle_heights=[]
    # Process video frames
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        percentage = process_frame(frame, model,liquid_level_heights,bottle_heights)
        if percentage:
            percentages.append(percentage)
        
    # Calculate total sums
    # Check if both lists are empty or have fewer than 3-4 values
    
    cap.release()
    
    # Calculate and print results
    if len(bottle_heights) < 1 or len(liquid_level_heights) < 1:
        print("Either we are not able to find the bottle or liquid level, or there is some error in detection.")
        
    # Check if only bottle heights are detected and liquid levels are missing or have fewer values
    elif len(bottle_heights) >= 1 and len(liquid_level_heights) < 1:
        print("Bottle detected, but the liquid level is either empty, not detected, or has insufficient data.")
    else : 
      if percentages:
          avg_percentage = np.mean(percentages)
          print(f"Average liquid level percentage: {avg_percentage:.2f}%")
      else:
          print("No valid liquid level percentages were detected in the video.")

if __name__ == "__main__":
    main()
