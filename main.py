# Importing Opencv
import cv2

# Path to the tiff file
path = "dataset/IT229148/IT229148_25_50fps_binning2_sample3(1).tif"

# List to store the loaded image
images = []

ret, images = cv2.imreadmulti(mats=images,
                              filename=path,
                              flags=cv2.IMREAD_ANYCOLOR)

idx = 0
# Show the images
while True:
    # Displaying the image
    cv2.imshow("TF frame", images[idx])

    idx += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Waiting for user to press any key to stop displaying
cv2.waitKey()

# Destroying all windows
cv2.destroyAllWindows()
