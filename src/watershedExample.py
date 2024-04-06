import cv2
import numpy as np

# Read the video
cap = cv2.VideoCapture('./src/TestVideoClipped.mp4')

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Loop over the video frames
while True:
    # Read the next frame
    ret, frame = cap.read()

    # If the frame is empty, break
    if not ret:
        break

    # Apply the background subtractor
    fg_mask = bg_subtractor.apply(frame)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(fg_mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 2)

    # opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = 2)

    # # Apply the watershed algorithm
    # markers = cv2.watershed(fg_mask, frame)

    # Display the segmented frame
    cv2.imshow('Segmented Frame', closing)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()