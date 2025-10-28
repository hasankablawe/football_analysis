from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.team_colors_lab = {}  # Store LAB values for better color comparison
        self.player_team_dict = {}
        self.player_positions = {}  # Store last known player positions
        self.team_positions = {1: [], 2: []}  # Store initial position clusters
        self.confidence_threshold = 0.7  # Confidence threshold for color assignment

    def get_player_color(self, frame, bbox):
        """
        Extract dominant jersey color by filtering out grass, gray, and dark pixels.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.array([128, 128, 128])  # Default gray

        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.array([128, 128, 128])
        
        # Focus on upper body (jersey area) - top 60%
        jersey_height = int(player_region.shape[0] * 0.6)
        jersey_region = player_region[:jersey_height, :]
        
        if jersey_region.size < 20: # Fallback if jersey region is too small
            jersey_region = player_region
        
        if jersey_region.size < 20:
            return np.array([128, 128, 128])

        # --- 1. Filter out unwanted colors ---
        # Convert to HSV for filtering
        jersey_hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Filter 1: Green (grass)
        # Hue: 30-90, Saturation > 40, Value > 30
        lower_green = np.array([30, 40, 30])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(jersey_hsv, lower_green, upper_green)
        
        # Filter 2: White/Gray/Black (low saturation)
        # Saturation < 35
        lower_gray = np.array([0, 0, 0])
        upper_gray = np.array([180, 35, 255])
        mask_gray = cv2.inRange(jersey_hsv, lower_gray, upper_gray)

        # Filter 3: Very dark pixels
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 30])
        mask_dark = cv2.inRange(jersey_hsv, lower_dark, upper_dark)

        # Combine masks: We want pixels that are NOT green AND NOT gray AND NOT dark
        mask_unwanted = cv2.bitwise_or(mask_green, mask_gray)
        mask_unwanted = cv2.bitwise_or(mask_unwanted, mask_dark)
        mask_jersey = cv2.bitwise_not(mask_unwanted)
        
        # Get BGR pixels that are part of the jersey
        jersey_pixels_bgr = jersey_region[mask_jersey > 0]
        
        if jersey_pixels_bgr.shape[0] < 20:  # Not enough pixels after filtering
            # Fallback: use mean of the LAB color space (more robust than BGR mean)
            jersey_lab_full = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
            mean_lab = np.mean(jersey_lab_full.reshape(-1, 3), axis=0)
            mean_bgr = cv2.cvtColor(np.uint8([[mean_lab]]), cv2.COLOR_LAB2BGR)[0][0]
            return mean_bgr
        
        # --- 2. Cluster the filtered pixels in LAB space ---
        try:
            # Convert filtered pixels to LAB
            jersey_pixels_lab = cv2.cvtColor(jersey_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            
            # K-means with 2 clusters (jersey color, maybe secondary color/skin)
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=0)
            kmeans.fit(jersey_pixels_lab)
            
            # Find the most prominent color (largest cluster)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster_idx = labels[np.argmax(counts)]
            dominant_color_lab = kmeans.cluster_centers_[dominant_cluster_idx]
            
            # Convert back to BGR
            player_color_bgr = cv2.cvtColor(np.uint8([[dominant_color_lab]]), cv2.COLOR_LAB2BGR)[0][0]
            
        except Exception as e:
            # Fallback if clustering fails
            player_color_bgr = np.mean(jersey_pixels_bgr, axis=0)
        
        return player_color_bgr


    def assign_team_color(self, frames, player_tracks):
        """
        Assign team colors by clustering filtered player colors in LAB space.
        """
        print("Initializing team colors with improved filtering...")
        
        all_player_colors = []
        all_player_positions = []
        
        # Sample frames (e.g., every 10th frame up to 50 frames)
        sample_frames = list(range(0, min(len(frames), 50), 10))
        
        for frame_idx in sample_frames:
            if frame_idx < len(player_tracks):
                frame = frames[frame_idx]
                frame_tracks = player_tracks[frame_idx]
                
                for player_id, track in frame_tracks.items():
                    bbox = track["bbox"]
                    # Use the new, improved color extraction
                    player_color = self.get_player_color(frame, bbox)
                    
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    all_player_colors.append(player_color)
                    all_player_positions.append([center_x, center_y])
        
        if len(all_player_colors) < 4:
            print("Warning: Not enough players detected for team clustering. Using defaults.")
            self.team_colors[1] = np.array([255, 0, 0])  # Red
            self.team_colors[2] = np.array([0, 0, 255])  # Blue
            self.team_colors_lab[1] = cv2.cvtColor(np.uint8([[self.team_colors[1]]]), cv2.COLOR_BGR2LAB)[0][0]
            self.team_colors_lab[2] = cv2.cvtColor(np.uint8([[self.team_colors[2]]]), cv2.COLOR_BGR2LAB)[0][0]
            return
        
        # --- Cluster in LAB space ---
        player_colors_bgr = np.array(all_player_colors, dtype=np.uint8)
        player_colors_lab = cv2.cvtColor(player_colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        
        print(f"Collected {len(player_colors_lab)} color samples. Clustering...")
        
        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=20, random_state=42)
            kmeans.fit(player_colors_lab)
            
            # Get the two cluster centers (in LAB)
            center_lab_1 = kmeans.cluster_centers_[0]
            center_lab_2 = kmeans.cluster_centers_[1]
            
            # --- Check if colors are too similar ---
            color_distance = np.linalg.norm(center_lab_1 - center_lab_2)
            # A distance < 30 in LAB space is perceptually very close
            if color_distance < 30:
                print(f"**WARNING**: Clustered team colors are too similar (Dist: {color_distance:.2f}).")
                print("Assignment may be unreliable. This can happen if only one team is visible.")
            
            # Store LAB centers
            self.team_colors_lab[1] = center_lab_1
            self.team_colors_lab[2] = center_lab_2
            
            # Store BGR centers (for drawing)
            self.team_colors[1] = cv2.cvtColor(np.uint8([[center_lab_1]]), cv2.COLOR_LAB2BGR)[0][0]
            self.team_colors[2] = cv2.cvtColor(np.uint8([[center_lab_2]]), cv2.COLOR_LAB2BGR)[0][0]
            
            print(f"Team 1 color (BGR): {self.team_colors[1]}")
            print(f"Team 2 color (BGR): {self.team_colors[2]}")
            
            # Store team position clusters for spatial analysis
            labels = kmeans.labels_
            player_positions = np.array(all_player_positions)
            for i, label in enumerate(labels):
                team_id = label + 1
                self.team_positions[team_id].append(player_positions[i])
            
            print(f"Team assignment initialized successfully")

        except Exception as e:
            print(f"Team color clustering failed: {e}. Using default colors.")
            self.team_colors[1] = np.array([255, 0, 0])  # Red
            self.team_colors[2] = np.array([0, 0, 255])  # Blue
            self.team_colors_lab[1] = cv2.cvtColor(np.uint8([[self.team_colors[1]]]), cv2.COLOR_BGR2LAB)[0][0]
            self.team_colors_lab[2] = cv2.cvtColor(np.uint8([[self.team_colors[2]]]), cv2.COLOR_BGR2LAB)[0][0]


    def get_player_team(self, frame, player_bbox, player_id, is_goalkeeper=False):
        """
        Get player team using distance in LAB color space and spatial fallbacks.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color_bgr = self.get_player_color(frame, player_bbox)
        
        # --- Compare in LAB space for perceptual accuracy ---
        try:
            player_color_lab = cv2.cvtColor(np.uint8([[player_color_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        except Exception:
            # Fallback if conversion fails (e.g., color is invalid)
            player_color_lab = np.array([0, 0, 0]) 
        
        # Method 1: Distance-based assignment in LAB space
        dist_team1 = np.linalg.norm(player_color_lab - self.team_colors_lab[1])
        dist_team2 = np.linalg.norm(player_color_lab - self.team_colors_lab[2])
        
        total_dist = dist_team1 + dist_team2 + 1e-6 # Avoid division by zero
        
        if dist_team1 < dist_team2:
            team_id = 1
            confidence = 1.0 - (dist_team1 / total_dist)
        else:
            team_id = 2
            confidence = 1.0 - (dist_team2 / total_dist)
        
        # Calculate player position for spatial analysis
        center_x = (player_bbox[0] + player_bbox[2]) / 2
        center_y = (player_bbox[1] + player_bbox[3]) / 2
        player_position = np.array([center_x, center_y])
        
        # Method 2: Spatial analysis for goalkeepers or low confidence
        if is_goalkeeper or confidence < self.confidence_threshold:
            spatial_team = self._get_spatial_team_assignment(player_position)
            if spatial_team is not None:
                team_id = spatial_team
                confidence = max(confidence, 0.6)  # Boost confidence for spatial match
        
        # Method 3: Fallback to most common team if still uncertain
        if confidence < 0.3:
            # Assign to team with fewer players
            team_counts = {1: 0, 2: 0}
            for assigned_team in self.player_team_dict.values():
                if assigned_team in team_counts:
                    team_counts[assigned_team] += 1
            
            team_id = 1 if team_counts[1] <= team_counts[2] else 2
        
        # Store assignment
        self.player_team_dict[player_id] = team_id
        self.player_positions[player_id] = player_position
        
        return team_id

    
    def _get_spatial_team_assignment(self, player_position):
        """Use spatial analysis to assign team based on field position"""
        try:
            if not self.team_positions[1] or not self.team_positions[2]:
                return None
            
            # Calculate average positions for each team from initialization
            team1_pos = np.mean(self.team_positions[1], axis=0)
            team2_pos = np.mean(self.team_positions[2], axis=0)
            
            # Calculate distances
            dist_team1 = np.linalg.norm(player_position - team1_pos)
            dist_team2 = np.linalg.norm(player_position - team2_pos)
            
            return 1 if dist_team1 < dist_team2 else 2
                
        except Exception:
            return None