from utils import read_vid,save_vid
from tracker import Tracker
import cv2 
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import time
import os
import gc

def get_center_of_bbox(bbox):
    """Get center point of bounding box"""
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

def calculate_player_speeds(tracks, fps=30):
    """Calculate speed for each player across frames"""
    player_speeds = {}
    player_positions = {}
    
    # Collect positions for each player
    for frame_num, frame_tracks in enumerate(tracks['players']):
        for player_id, track in frame_tracks.items():
            if player_id not in player_positions:
                player_positions[player_id] = []
            
            center = get_center_of_bbox(track['bbox'])
            player_positions[player_id].append({
                'frame': frame_num,
                'position': center,
                'bbox': track['bbox']
            })
    
    # Calculate speeds
    for player_id, positions in player_positions.items():
        speeds = []
        distances = []
        
        # Need at least 2 positions to calculate speed
        if len(positions) < 2:
            player_speeds[player_id] = {
                'speeds': [], 'distances': [], 'avg_speed': 0, 
                'max_speed': 0, 'total_distance': 0, 'positions': positions
            }
            continue

        for i in range(1, len(positions)):
            prev_pos = positions[i-1]['position']
            curr_pos = positions[i]['position']
            
            # Calculate distance in pixels
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            distances.append(distance)
            
            # Convert to speed (pixels per second)
            speed = distance * fps
            speeds.append(speed)
        
        player_speeds[player_id] = {
            'speeds': speeds,
            'distances': distances,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'total_distance': np.sum(distances) if distances else 0,
            'positions': positions
        }
    
    return player_speeds

def analyze_player_movement(player_speeds):
    """Analyze player movement patterns"""
    analysis = {}
    
    for player_id, data in player_speeds.items():
        speeds = data['speeds']
        if not speeds:
            analysis[player_id] = {
                'avg_speed': 0, 'max_speed': 0, 'total_distance': data['total_distance'],
                'stationary_pct': 100, 'walking_pct': 0, 'running_pct': 0,
                'activity_level': 'Low'
            }
            continue
            
        # Movement categories
        stationary_frames = sum(1 for s in speeds if s < 10)  # Less than 10 pixels/sec
        walking_frames = sum(1 for s in speeds if 10 <= s < 50)  # 10-50 pixels/sec
        running_frames = sum(1 for s in speeds if s >= 50)  # 50+ pixels/sec
        
        total_frames = len(speeds)
        
        analysis[player_id] = {
            'avg_speed': data['avg_speed'],
            'max_speed': data['max_speed'],
            'total_distance': data['total_distance'],
            'stationary_pct': (stationary_frames / total_frames) * 100 if total_frames > 0 else 0,
            'walking_pct': (walking_frames / total_frames) * 100 if total_frames > 0 else 0,
            'running_pct': (running_frames / total_frames) * 100 if total_frames > 0 else 0,
            'activity_level': 'High' if data['avg_speed'] > 30 else 'Medium' if data['avg_speed'] > 15 else 'Low'
        }
    
    return analysis

def main():
    start_time = time.time()
    print("Starting football analysis...")
    
    #Reading video frames
    print("Loading video...")
    frames = read_vid('/home/linux/Downloads/08fd33_4.mp4')
    print(f"Loaded {len(frames)} frames")
    
    # Memory optimization: limit frames if needed
    max_frames = 1000  # Adjust based on your system
    if len(frames) > max_frames:
        print(f"Limiting to {max_frames} frames for memory optimization")
        frames = frames[:max_frames]
    
    tracker = Tracker('/home/linux/my_new_project/ml_projects/match_analsys/Models/best.pt')
    tracks = tracker.get_object_tracks(frames,read_from_stub=False,stub_path='stubs/track_stubs.pkl')
    
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames,
                                    tracks['players'])

    # Track team assignment statistics
    team_assignment_stats = {1: 0, 2: 0, 'unassigned': 0}
    total_assignments = 0
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            is_goalkeeper = track.get('is_goalkeeper', False)
            team = team_assigner.get_player_team(frames[frame_num],
                                                track['bbox'],
                                                player_id,
                                                is_goalkeeper=is_goalkeeper)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            tracks['players'][frame_num][player_id]['is_goalkeeper'] = is_goalkeeper
            
            # Track assignment statistics
            if team in [1, 2]:
                team_assignment_stats[team] += 1
            else:
                team_assignment_stats['unassigned'] += 1
            total_assignments += 1
            
            # if frame_num == 0:  # Only print for first frame to avoid spam
            #     print(f"Player {player_id}: Team {team} (Goalkeeper: {is_goalkeeper})")
    
    print(f"\nTeam Assignment Statistics:")
    print(f"Team 1: {team_assignment_stats[1]} assignments")
    print(f"Team 2: {team_assignment_stats[2]} assignments")
    print(f"Unassigned: {team_assignment_stats['unassigned']} assignments")
    print(f"Total assignments: {total_assignments}")


    #Assign ball To player 
    Player_Assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num , player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = Player_Assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1 :
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no player has the ball, carry over the last known team
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0) # Start with no team
                
    team_ball_control = np.array(team_ball_control)
    
    # Calculate speed analysis
    print("Calculating player speeds...")
    player_speeds = calculate_player_speeds(tracks, fps=30)
    movement_analysis = analyze_player_movement(player_speeds)
    
    # Calculate statistics
    print("\n=== FOOTBALL ANALYSIS STATISTICS ===")
    total_frames = len(frames)
    fps = 30  # Assuming 30 FPS
    
    # Ball possession statistics
    team_1_possession_frames = np.sum(team_ball_control == 1)
    team_2_possession_frames = np.sum(team_ball_control == 2)
    total_possession_frames = team_1_possession_frames + team_2_possession_frames
    
    if total_possession_frames == 0:
        team_1_possession = 0
        team_2_possession = 0
    else:
        team_1_possession = (team_1_possession_frames / total_possession_frames) * 100
        team_2_possession = (team_2_possession_frames / total_possession_frames) * 100
    
    print(f"Total frames analyzed: {total_frames}")
    print(f"Video duration: {total_frames/fps:.1f} seconds")
    print(f"Team 1 ball possession: {team_1_possession:.1f}%")
    print(f"Team 2 ball possession: {team_2_possession:.1f}%")
    
    # Player statistics with detailed analysis
    all_player_ids = set()
    player_appearances = {}
    
    for frame_num, frame_tracks in enumerate(tracks['players']):
        for player_id in frame_tracks.keys():
            all_player_ids.add(player_id)
            if player_id not in player_appearances:
                player_appearances[player_id] = []
            player_appearances[player_id].append(frame_num)
    
    total_players = len(all_player_ids)
    print(f"Total unique players detected: {total_players}")
    
    # Show player ID range
    min_id, max_id = 0, 0
    if all_player_ids:
        min_id = min(all_player_ids)
        max_id = max(all_player_ids)
        print(f"Player ID range: {min_id} to {max_id}")
        
        # Show most active players (appeared in most frames)
        player_activity = {pid: len(frames) for pid, frames in player_appearances.items()}
        most_active = sorted(player_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Most active players (by frame appearances):")
        for pid, frames_count in most_active:
            print(f"  Player {pid}: {frames_count} frames")
    
    # Ball touches (approximated by frames with ball)
    ball_touches = np.sum(team_ball_control != 0)
    print(f"Frames with ball control detected: {ball_touches}")
    
    # Speed analysis summary
    if movement_analysis:
        avg_speeds = [data['avg_speed'] for data in movement_analysis.values()]
        max_speeds = [data['max_speed'] for data in movement_analysis.values()]
        total_distances = [data['total_distance'] for data in movement_analysis.values()]
        
        # --- PROACTIVE FIX: Check for empty lists ---
        avg_speed_stat = np.mean(avg_speeds) if avg_speeds else 0
        max_speed_stat = np.max(max_speeds) if max_speeds else 0
        total_dist_stat = np.sum(total_distances) if total_distances else 0
        # --- END FIX ---

        print(f"\n=== SPEED ANALYSIS ===")
        print(f"Average player speed: {avg_speed_stat:.1f} pixels/sec")
        print(f"Fastest player speed: {max_speed_stat:.1f} pixels/sec")
        print(f"Total distance covered: {total_dist_stat:.1f} pixels")
        
        # Most active players
        most_active = sorted(movement_analysis.items(), key=lambda x: x[1]['avg_speed'], reverse=True)[:5]
        print("Most active players (by average speed):")
        for player_id, data in most_active:
            print(f"  Player {player_id}: {data['avg_speed']:.1f} px/sec ({data['activity_level']} activity)")
    
    #Draw object track 
    print("Drawing annotations...")
    output_video = tracker.draw_annotations(frames,tracks,team_ball_control)

    #Save video
    os.makedirs('output_video', exist_ok=True)
    print("Saving video...")
    save_vid(output_video , 'output_video/video.avi')
    
    # Save statistics to file
    stats_file = 'output_video/analysis_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("FOOTBALL ANALYSIS STATISTICS\n")
        f.write("="*40 + "\n")
        f.write(f"Total frames analyzed: {total_frames}\n")
        f.write(f"Video duration: {total_frames/fps:.1f} seconds\n")
        f.write(f"Team 1 ball possession: {team_1_possession:.1f}%\n")
        f.write(f"Team 2 ball possession: {team_2_possession:.1f}%\n")
        f.write(f"Total unique players detected: {total_players}\n")
        f.write(f"Player ID range: {min_id} to {max_id}\n")
        f.write(f"Frames with ball control: {ball_touches}\n")
        f.write(f"Processing time: {time.time() - start_time:.1f} seconds\n")
        
        # Add speed analysis
        if movement_analysis:
            # Re-using variables from above
            avg_speeds = [data['avg_speed'] for data in movement_analysis.values()]
            max_speeds = [data['max_speed'] for data in movement_analysis.values()]
            total_distances = [data['total_distance'] for data in movement_analysis.values()]
            
            avg_speed_stat = np.mean(avg_speeds) if avg_speeds else 0
            max_speed_stat = np.max(max_speeds) if max_speeds else 0
            total_dist_stat = np.sum(total_distances) if total_distances else 0

            f.write(f"\nSPEED ANALYSIS\n")
            f.write("-"*20 + "\n")
            f.write(f"Average player speed: {avg_speed_stat:.1f} pixels/sec\n")
            f.write(f"Fastest player speed: {max_speed_stat:.1f} pixels/sec\n")
            f.write(f"Total distance covered: {total_dist_stat:.1f} pixels\n")
            
            # Most active players
            most_active = sorted(movement_analysis.items(), key=lambda x: x[1]['avg_speed'], reverse=True)[:10]
            f.write(f"\nMost Active Players (by speed):\n")
            for player_id, data in most_active:
                f.write(f"Player {player_id}: {data['avg_speed']:.1f} px/sec ({data['activity_level']})\n")
        
        # Add detailed player analysis
        if player_appearances:
            f.write("\nPLAYER ACTIVITY ANALYSIS\n")
            f.write("-"*30 + "\n")
            player_activity = {pid: len(frames) for pid, frames in player_appearances.items()}
            most_active = sorted(player_activity.items(), key=lambda x: x[1], reverse=True)
            for pid, frames_count in most_active:
                f.write(f"Player {pid}: {frames_count} frames ({frames_count/total_frames*100:.1f}%)\n")
    
    print(f"\nAnalysis complete!")
    print(f"Processing time: {time.time() - start_time:.1f} seconds")
    print(f"Statistics saved to: {stats_file}")
    print(f"Video saved to: output_video/video.avi")
    
    # Clean up memory
    del frames
    del tracks
    del output_video
    gc.collect()

if __name__ == '__main__'  :
    main()