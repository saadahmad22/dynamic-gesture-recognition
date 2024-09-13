import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from config import class_labels, torch

class VideoDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, keyframe_threshold=0.5, classes_=None):
        '''Initialize the dataset
        
        Parameters:
        - dataset_dir: Path to the dataset directory.
        - transform: Optional transform to be applied to the samples.
        - keyframe_threshold: Threshold for key frame detection.
        - classes_: List of classes to include in the dataset.
        '''

        self.samples = []
        self.transform = transform
        self.keyframe_threshold = keyframe_threshold
        
        for class_name in (os.listdir(dataset_dir) if not classes_ else classes_):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                key_frames, optical_flows = self.extract_key_frames_and_flows(video_path)
                for frame_path, flow_path in zip(key_frames, optical_flows):
                    self.samples.append((frame_path, flow_path, class_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, flow_path, class_name = self.samples[idx]
                
        key_frame = Image.open(frame_path).convert('RGB')
        optical_flow = Image.open(flow_path).convert('RGB')
        
        if self.transform:
            key_frame = self.transform(key_frame)
            optical_flow = self.transform(optical_flow)
        
        label = class_labels[class_name]

        return key_frame, optical_flow, label
    
    def extract_key_frames_and_flows(self, video_path):
        '''Extract key frames, and thus and optical flows from a video'''

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        key_frames = []
        optical_flows = []
        # 0th frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return key_frames, optical_flows
        
        # Convert the frame to HSV and compute its histogram
        prev_frame_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        prev_hist = self.compute_hsv_histogram(prev_frame_hsv)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            curr_hist = self.compute_hsv_histogram(frame_hsv)
            similarity = self.calculate_histogram_similarity(prev_hist, curr_hist)
            
            # if not similar enough, should be key frame
            if similarity < self.keyframe_threshold:
                # set file path
                frame_path = f"{video_path}_frame_{frame_idx}.jpg"
                # write out 
                cv2.imwrite(frame_path, frame)
                key_frames.append(frame_path)
                
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(prev_frame)
                hsv[..., 1] = 255
                hsv[..., 0] = flow_angle * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                flow_path = f"{video_path}_flow_{frame_idx}.jpg"
                cv2.imwrite(flow_path, flow_rgb)
                optical_flows.append(flow_path)
            
            prev_hist = curr_hist
            prev_frame = frame
            frame_idx += 1
        
        cap.release()
        return key_frames, optical_flows
    
    def compute_hsv_histogram(self, hsv_frame):
        '''Compute a histogram from the HSV, as seen in the paper'''

        h_bins = 12
        s_bins = 5
        v_bins = 5
        hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    
    def calculate_histogram_similarity(self, hist1, hist2):
        '''Calculate the similarity between two histograms'''

        # Accumulate the minimum values at the corresponding index of the two histograms
        min_hist = np.minimum(hist1, hist2)
        similarity = np.sum(min_hist)

        normalized_similarity = similarity / hist1.size
        
        return normalized_similarity
    
    def rgb_to_optical_flow(self, image):
        image_np = np.array(image)
        
        # Split the image into its three channels
        r, g = image_np[:, :, 0], image_np[:, :, 1]
        
        # Use the red and green channels as the horizontal and vertical components of the flow
        flow = np.stack((r, g), axis=0)
        
        # Convert the flow to a PyTorch tensor
        flow = torch.tensor(flow, dtype=torch.float32)
        
        return flow