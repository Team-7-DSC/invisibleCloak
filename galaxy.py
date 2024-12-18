import cv2
import numpy as np
import time

def create_background(cap, num_frames=30):
    print("Capturing background. Please move out of frame.")
    backgrounds = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            backgrounds.append(frame)
        else:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
        time.sleep(0.1)
    if backgrounds:
        return np.median(backgrounds, axis=0).astype(np.uint8)
    else:
        raise ValueError("Could not capture any frames for background")

def create_mask(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return mask

def apply_cloak_effect_with_video(frame, mask, galaxy_video):
    # Read a frame from the galaxy video
    ret, galaxy_frame = galaxy_video.read()
    if not ret:
        # Restart video if it ends
        galaxy_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, galaxy_frame = galaxy_video.read()

    # Resize galaxy frame to match the size of the input frame
    galaxy_resized = cv2.resize(galaxy_frame, (frame.shape[1], frame.shape[0]))

    # Create the inverse of the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to separate the foreground and background
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(galaxy_resized, galaxy_resized, mask=mask)

    # Combine the foreground and the new background
    return cv2.add(fg, bg)

def main():
    print("OpenCV version:", cv2.__version__)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Load the galaxy video
    galaxy_video = cv2.VideoCapture('galaxy.mp4')
    if not galaxy_video.isOpened():
        print("Error: Could not open galaxy video.")
        cap.release()
        return

    try:
        background = create_background(cap)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        galaxy_video.release()
        return

    # Define the color range for the cloak (e.g., blue)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    print("Starting main loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(1)
            continue

        # Create a mask for the cloak
        mask = create_mask(frame, lower_blue, upper_blue)

        # Apply the augmented background overlay effect
        result = apply_cloak_effect_with_video(frame, mask, galaxy_video)

        # Show the result
        cv2.imshow('Augmented Invisibility Cloak', result)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    galaxy_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
