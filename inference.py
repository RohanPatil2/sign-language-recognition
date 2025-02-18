import cv2
import os
import time
from datetime import datetime

def save_webcam(out_dir="recordings", fps=24.0, mirror=False, resolution=(1280, 720)):
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outPath = os.path.join(out_dir, f"recording_{timestamp}.avi")

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Get actual frame width, height, and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps  # Use webcam FPS if available
    
    print(f"Recording at {width}x{height}, {actual_fps} FPS")

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outPath, fourcc, actual_fps, (width, height))

    recording = True  # Flag to toggle recording
    start_time = time.time()  # Track recording duration

    print("Press 's' to pause/resume, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Frame capture failed.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        if recording:
            out.write(frame)

            # Display elapsed recording time
            elapsed_time = time.time() - start_time
            time_text = f"Recording: {elapsed_time:.1f}s"
            cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Webcam Recording', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit recording
            print("Stopping recording...")
            break
        elif key == ord('s'):  # Toggle pause/resume
            recording = not recording
            state = "Resumed" if recording else "Paused"
            print(f"Recording {state}")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    save_webcam(fps=30.0, mirror=True, resolution=(1280, 720))

if __name__ == '__main__':
    main()
