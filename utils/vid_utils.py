import cv2

class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
    
    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        # Handle Slice (e.g., frames[0:10]) for batch processing
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.frame_count)
            if step is None: step = 1
            
            results = []
            if step == 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(start, stop):
                    ret, frame = self.cap.read()
                    if not ret: break
                    results.append(frame)
            return results

        # Handle Single Index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret: raise IndexError(f"Could not read frame {idx}")
        return frame

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            yield frame

def read_video(video_path):
    return VideoReader(video_path)

def save_video(output_video_frames, output_video_path):
    # --- THE FIX IS HERE ---
    # We cannot use output_video_frames[0] because it is a generator.
    # We use next() to grab the first frame, getting its size, 
    # and then write it.
    try:
        # Consume the first frame to get dimensions
        first_frame = next(iter(output_video_frames))
    except StopIteration:
        print("Error: No frames to save.")
        return

    height, width = first_frame.shape[:2]
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    # Write the first frame (which we already consumed)
    out.write(first_frame)
    
    # Write the remaining frames in the loop
    for frame in output_video_frames:
        out.write(frame)
        
    out.release()
