from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape to 2D array
        image_2d = image.reshape(-1, 3)

        # K-means with 2 clusters (Background vs Shirt)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Take only top half
        top_half_image = image[0:int(image.shape[0]/2), :]

        # --- SPEED OPTIMIZATION 1: Resize ---
        if top_half_image.size == 0:
            return np.array([0,0,0]) 
            
        top_half_image = cv2.resize(top_half_image, (30, 30)) 
        
        # --- NEW FIX: CENTER CROP ---
        height, width, _ = top_half_image.shape
        start_x = int(width * 0.2)
        end_x = int(width * 0.8)
        top_half_image = top_half_image[:, start_x:end_x]
        # ----------------------------

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape to matches image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # --- LOGIC FIX: USE CENTER PIXEL ---
        # Since we cropped to the center, the middle pixel is definitely the player.
        # The corners might now be the shirt (bottom) or background (top), which is risky.
        center_y = int(clustered_image.shape[0] / 2)
        center_x = int(clustered_image.shape[1] / 2)
        
        player_cluster = clustered_image[center_y, center_x]
        
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # --- SPEED OPTIMIZATION 2: The Cache ---
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1 

        self.player_team_dict[player_id] = team_id

        return team_id
