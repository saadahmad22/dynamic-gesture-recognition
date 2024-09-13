'''Helper functions for classes

get_activation_layer: Create activation layer from string/function.
extract_key_frames: Extract key frames from a video and save them to the output directory.
compute_optical_flow: Compute optical flow between two frames.
process_video: Process a video by extracting key frames and computing optical flow.
process_dataset: Process a dataset by extracting key frames and computing optical flow for all videos.

Credits: osmr on github for "get_activation_layer"
'''

import cv2
import os
import numpy as np
import concurrent.futures

from config import isfunction, nn
from activations import Swish, HSwish


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.

    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video and save them to the output directory.
    
    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the extracted frames.
    - frame_interval: Interval between frames to be saved (i.e., fps).
    """

    current = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = current.read()
    
    while success:
        # writes every frame_interval'th frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        success, frame = current.read()
        frame_count += 1
    
    current.release()

def compute_optical_flow(video_path, output_dir):
    """
    Compute optical flow between consecutive frames in a video and save the results.
    
    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the optical flow images.
    """

    current = cv2.VideoCapture(video_path)
    # just get first, not use it since nothing before it
    ret, frame1 = current.read()
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    while True:
        ret, frame2 = current.read()
        # break on the i+1 frame
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        flow_filename = os.path.join(output_dir, f"flow_{frame_count}.jpg")
        cv2.imwrite(flow_filename, bgr)
        
        prev = next
        frame_count += 1
    
    current.release()

def process_video(video_path, class_output_dir, frame_interval):
    video_name = os.path.basename(video_path)
    video_output_dir = os.path.join(class_output_dir, os.path.splitext(video_name)[0])
    os.makedirs(video_output_dir, exist_ok=True)
    
    extract_frames(video_path, video_output_dir, frame_interval)
    compute_optical_flow(video_path, video_output_dir)

def process_dataset(dataset_dir, output_dir, frame_interval=30, max_workers=None, classes_=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for class_name in (os.listdir(dataset_dir) if not classes_ else classes_):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                futures.append(executor.submit(process_video, video_path, class_output_dir, frame_interval))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video: {e}")