import cv2
import moviepy.editor as mp


def process_video(video_clip, output_filename):
    # Load the cascade classifier for face detection
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Get the video properties
    width = int(video_clip.size[0])
    height = int(video_clip.size[1])
    fps = video_clip.fps
    frame_count = int(video_clip.duration * fps)

    # Variables for tracking the position of the face
    total_faces = 0
    total_x = 0
    total_y = 0

    # Number of frames to evenly sample
    num_sample_frames = 10

    # Calculate the interval between sample frames
    frame_interval = max(1, frame_count // num_sample_frames)

    # Iterate through the frames of the video to detect the face position
    for i in range(0, frame_count, frame_interval):
        # Get the frame at the given time
        frame = video_clip.get_frame(i / fps)

        # Detect faces in the frame
        faces = cascade.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30)
        )

        # If a face is detected, update the position variables
        if len(faces) > 0:
            x, y, w, h = faces[0]
            total_faces += 1
            total_x += x + w // 2
            total_y += y + h // 2

    # Calculate the average position of the face
    avg_x = total_x // total_faces if total_faces > 0 else 0
    avg_y = total_y // total_faces if total_faces > 0 else 0

    # Calculate the dimensions of the output video
    out_width = int(height * 9 / 16)
    out_height = height

    # Create the VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (out_width, out_height))

    # Reset the video clip to process frames again
    video_clip.reader.initialize()

    # Iterate through the frames of the video
    for i in range(frame_count):
        # Get the frame at the given time
        frame = video_clip.get_frame(i / fps)

        # Crop the frame around the perfect center of the face
        left = max(avg_x - out_width // 2, 0)
        right = min(left + out_width, width)
        top = max(avg_y - out_height // 2, 0)
        bottom = min(top + out_height, height)
        cropped_frame = frame[top:bottom, left:right]

        # Resize the cropped frame to the output size
        resized_frame = cv2.resize(cropped_frame, (out_width, out_height))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Write the resized frame to the output video
        out.write(resized_frame)

    # Release the resources
    out.release()

    input_audio = video_clip.audio
    output_clip = mp.VideoFileClip(output_filename)
    output_clip = output_clip.set_audio(input_audio)

    return output_clip
