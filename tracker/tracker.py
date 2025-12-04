from ultralytics import YOLO
import supervision as sv
import pickle
import os 
import cv2
import numpy as np
import pandas as pd 
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=30, 
            minimum_matching_threshold=0.5
        )
    
    def detect_frames_generator(self, frames, batch_size=30):
        """
        Yields batches of detections to avoid storing all YOLO results in memory.
        """
        # Iterate through frames in batches using the length of the frames object (VideoReader)
        for i in range(0, len(frames), batch_size):
            # Batch prediction is much faster than single-frame prediction
            # frames[i:i+batch_size] triggers the slice reading in VideoReader
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.2, agnostic_nms=True, verbose=False, iou=0.7)
            yield detections_batch

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        print("Detecting objects (Batch Processing)...")
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        print("Tracking objects...")
        
        frame_num = 0
        # iterate over batches
        for detections_batch in self.detect_frames_generator(frames):
            for detection in detections_batch:
                cls_names = detection.names
                cls_names_inv = {v:k for k,v in cls_names.items()}

                # Convert to supervision Detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)

                # ============================================================
                # --- INSERT THIS BLOCK HERE ---
                # 1. Get the actual image of the current frame
                current_frame = frames[frame_num]

                # 2. Filter out detections that are NOT on the grass (Crowd/Stands)
                detection_supervision = self.filter_detections_by_grass(current_frame, detection_supervision)
                # ============================================================

                # Handle Goalkeeper as Player for tracking continuity
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]

                # Update Tracker
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                # Prepare empty dicts for this frame
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})

                # Store Tracked Objects (Players/Refs)
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                    elif cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
                # Store Untracked Objects (Ball)
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
                
                frame_num += 1

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # OPTIMIZATION: Removed list append. Using yield for memory efficiency.
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Yield frame one by one to prevent RAM overflow
            yield frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Safety check for OpenCV color format (numpy int vs python int)
        if isinstance(color, (np.ndarray, list, tuple)):
            color = tuple(int(c) for c in color)

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Ensure color is standard int tuple
        if isinstance(color, (np.ndarray, list, tuple)):
            color = tuple(int(c) for c in color)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        total = team_1_num_frames + team_2_num_frames
        
        if total == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_1 = team_1_num_frames / total
            team_2 = team_2_num_frames / total

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill().ffill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def interpolate_object_tracks(self, object_tracks, dist_tolerance=100):
        """
        Fixed interpolation: Only fills short gaps (occlusion).
        Does NOT extend tracks to the start/end of video.
        """
        # 1. Identify all unique track IDs
        unique_track_ids = set()
        for frame_tracks in object_tracks:
            unique_track_ids.update(frame_tracks.keys())
        
        # 2. Iterate over each track ID
        for track_id in unique_track_ids:
            frame_bboxes = []
            
            # Collect data for this specific ID
            for frame_tracks in object_tracks:
                if track_id in frame_tracks:
                    frame_bboxes.append(frame_tracks[track_id]['bbox'])
                else:
                    frame_bboxes.append([np.nan, np.nan, np.nan, np.nan]) 

            df = pd.DataFrame(frame_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
            
            # --- THE FIX IS HERE ---
            # limit=20: Only fill gaps smaller than 20 frames (approx 0.6 seconds)
            # If the gap is larger, it means the player likely left the view.
            df = df.interpolate(method='linear', limit=20, limit_direction='forward')
            
            # REMOVED: df.ffill().bfill() -> This was causing the "ghosts"
            # -----------------------

            # 3. Write interpolated values back
            for frame_num, row in df.iterrows():
                # check if row still has NaNs (meaning we shouldn't draw here)
                if row.isnull().values.any():
                    continue

                bbox = row.to_list()
                
                # Only add the track if it wasn't there (filling the gap)
                # OR overwrite it if you trust interpolation more than raw detection
                if track_id not in object_tracks[frame_num]:
                    object_tracks[frame_num][track_id] = {"bbox": bbox}

        return object_tracks

    def filter_short_tracks(self, tracks, min_track_length=60):
        """
        Removes any object ID that appears in fewer than 'min_track_length' frames.
        This eliminates flickering 'ghosts' and noise.
        """
        # 1. Count how many frames each track_id appears in
        track_lengths = {}
        for frame_tracks in tracks:
            for track_id in frame_tracks.keys():
                track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

        # 2. Identify short tracks to remove
        short_tracks = {k for k, v in track_lengths.items() if v < min_track_length}

        # 3. Remove them from the main tracks list
        for frame_tracks in tracks:
            for track_id in short_tracks:
                if track_id in frame_tracks:
                    del frame_tracks[track_id]

        print(f"Removed {len(short_tracks)} ghost tracks that were too short.")
        return tracks
    
    def normalize_track_ids(self, tracks):
        """
        Renames track IDs to standard numbers (1, 2, 3...) based on 
        which objects appear the longest.
        """
        # 1. Calculate duration of every track ID
        track_durations = {}
        for frame_tracks in tracks:
            for track_id in frame_tracks.keys():
                track_durations[track_id] = track_durations.get(track_id, 0) + 1

        # 2. Sort IDs by duration (Longest tracked object gets ID 1)
        # We assume real players are on screen longest.
        sorted_ids = sorted(track_durations.keys(), key=lambda x: track_durations[x], reverse=True)
        
        # 3. Create a map: {Old_ID: New_ID}
        id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted_ids)}

        # 4. Apply the new IDs
        new_tracks = []
        for frame_tracks in tracks:
            new_frame_tracks = {}
            for track_id, track_info in frame_tracks.items():
                new_id = id_map[track_id]
                new_frame_tracks[new_id] = track_info
            new_tracks.append(new_frame_tracks)
        
        return new_tracks
    def clean_duplicate_tracks(self, tracks, iou_threshold=0.1):
        """
        Removes overlapping tracks for Referees.
        If two referees overlap, it keeps the one with the longest history.
        """
        # Calculate track lengths to know which one is the "main" track
        track_lengths = {}
        for frame_tracks in tracks:
            for track_id in frame_tracks.keys():
                track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

        for frame_num, frame_tracks in enumerate(tracks):
            track_ids = list(frame_tracks.keys())
            
            # Check every pair of tracks in this frame
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    id1 = track_ids[i]
                    id2 = track_ids[j]
                    
                    if id1 not in frame_tracks or id2 not in frame_tracks:
                        continue
                    
                    bbox1 = frame_tracks[id1]['bbox']
                    bbox2 = frame_tracks[id2]['bbox']
                    
                    # Calculate Intersection over Union (IoU) manually
                    x1 = max(bbox1[0], bbox2[0])
                    y1 = max(bbox1[1], bbox2[1])
                    x2 = min(bbox1[2], bbox2[2])
                    y2 = min(bbox1[3], bbox2[3])
                    
                    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
                    
                    if intersection_area > 0: # If they overlap AT ALL
                        # Keep the longer track, delete the shorter one
                        if track_lengths[id1] >= track_lengths[id2]:
                            del frame_tracks[id2]
                        else:
                            del frame_tracks[id1]
                            
        return tracks
    # In tracker.py

    def filter_detections_by_grass(self, frame, detection_supervision):
        """
        Removes detections where the 'feet' (bottom center) are not on the grass.
        """
        # 1. Convert frame to HSV (Hue, Saturation, Value) for color checking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Define the "Green" range for a football pitch
        # OpenCV Hue range is [0-179]. Green is usually around 35-85.
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # 3. Create a mask (White = Grass, Black = Crowd/Stands)
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        valid_indices = []
        
        for i, bbox in enumerate(detection_supervision.xyxy):
            x1, y1, x2, y2 = bbox
            
            # Check the "Feet" position (Bottom Center)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            
            # Clip coordinates to be inside the frame
            h, w = grass_mask.shape
            foot_x = max(0, min(w - 1, foot_x))
            foot_y = max(0, min(h - 1, foot_y))
            
            # Check a small area around the foot (3x3 pixels) to be safe
            # If the area is mostly green (255), keep the player.
            region = grass_mask[max(0, foot_y-5):foot_y, max(0, foot_x-2):foot_x+2]
            
            if np.mean(region) > 127: # If > 50% of the foot area is green
                valid_indices.append(i)
        
        # 4. Filter the detections object
        # sv.Detections allows indexing to keep only valid ones
        return detection_supervision[valid_indices]
