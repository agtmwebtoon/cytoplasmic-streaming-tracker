import cv2
import numpy as np
import matplotlib.pyplot as plt

first_flag = True


def drawFlow(img, flow, step=2):
    h, w = img.shape[:2]

    idx_y, idx_x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    indicies = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

    min_dist = find_optical_distance(flow, 90, 512)

    print(min_dist)
    for x, y in indicies:
        dx, dy = flow[y, x].astype(int)
        if np.sqrt(dx ** 2 + dy ** 2) > min_dist:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)

def find_optical_distance(pixel_dist, percent, pixel_size):
    global first_flag
    dists = []
    for i in range(pixel_size):
        for j in range(pixel_size):
              dists.append(np.sqrt(pixel_dist[i][j][0] ** 2 + pixel_dist[i][j][1] ** 2))

    if first_flag:
        plt.plot(dists)
        plt.show()
        first_flag = False

    return np.percentile(dists, percent)



prev = None

cap = cv2.VideoCapture('../dataset/50fps_CM334_18_sample2 (5)_low_activity.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev is None:
        prev = gray
    else:

        '''
        prev – first 8-bit single-channel input image.
        next – second input image of the same size and the same type as prev.
        flow – computed flow image that has the same size as prev and type CV_32FC2.
        pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
        levels – number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
        winsize – averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
        iterations – number of iterations the algorithm does at each pyramid level.
        poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
        poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
        '''
        flow = cv2.calcOpticalFlowFarneback(prev, gray, flow=0.5,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=10,
                                            iterations=7,
                                            poly_n=13,
                                            poly_sigma=1.1,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        drawFlow(frame, flow)
        prev = gray

    cv2.imshow('OpticalFlow-Farneback', frame)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()