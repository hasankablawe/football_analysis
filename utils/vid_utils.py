import cv2 
def read_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret :
            break 
        frames.append(frame)
    return frames

def save_vid(output_vid_frames ,output_vid_path ):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(output_vid_path,fourcc , 20.0 , (output_vid_frames[0].shape[1],output_vid_frames[0].shape[0]))
    for frame in output_vid_frames:
        out.write(frame)
    out.release()



