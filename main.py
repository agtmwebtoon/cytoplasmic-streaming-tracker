import cv2

# Path to the TIF image file
tif_file_path = 'raw_data/IT283501l_25_50fps_binning2_sample2(1).tif'

# Open the TIF image as a 'video'
cap = cv2.VideoCapture(tif_file_path)

# Check if the TIF image was opened successfully
if not cap.isOpened():
    print("Error opening TIF file")
    exit()

# Loop through each frame in the TIF image
while True:
    ret, frame = cap.read()
    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Display or process the frame
    cv2.imshow('TIF Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(10):
        break

# Release the TIF file capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
