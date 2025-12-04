from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import os

def main():
    # 0. Setup
    video_path = 'input_videos/08fd33_4.mp4'
    model_path = 'models/best.pt'
    
    if not os.path.exists('stubs'):
        os.makedirs('stubs')

    # 1. Read Video
    print("Reading video...")
    video_frames = read_video(video_path)

    # 2. Tracking
    tracker = Tracker(model_path)
    print("Tracking objects...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False
    )
    
    print("Interpolating missing positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracks["players"] = tracker.interpolate_object_tracks(tracks["players"])
    tracks["referees"] = tracker.interpolate_object_tracks(tracks["referees"])
    
    print("Filtering ghost tracks...")
    tracks["players"] = tracker.filter_short_tracks(tracks["players"], min_track_length=60)
    tracks["referees"] = tracker.filter_short_tracks(tracks["referees"], min_track_length=30)

    print("Cleaning duplicate referees...")
    tracks["referees"] = tracker.clean_duplicate_tracks(tracks["referees"])

    print("Normalizing Player IDs...")
    tracks["players"] = tracker.normalize_track_ids(tracks["players"])

    # 6. Team Assignment
    print("Assigning teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, (frame, player_track) in enumerate(zip(video_frames, tracks['players'])):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame, track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # 7. Ball Possession
    print("Assigning ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        # Safe access to ball
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        assigned_player = -1

        if ball_bbox is not None:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(-1) 

    team_ball_control = np.array(team_ball_control)

    # 8. Draw & Save (Chained Generators)
    print("Drawing and Saving Video...")
    
    # Chain 1: Tracker Annotations
    output_gen = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    # Consumes the chain frame-by-frame
    save_video(output_gen, 'output_videos/output_video.mp4')
    print("Done!")

if __name__ == '__main__':
    main()
