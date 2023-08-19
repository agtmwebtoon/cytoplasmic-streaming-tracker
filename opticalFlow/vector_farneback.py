import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scienceplots
plt.style.use(['science','ieee'])

first_flag = True
roi_flag = True
length = 0
prev = None

cap = cv2.VideoCapture("../dataset/IT283501l_25_50fps_binning2_sample4(2).avi")
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
roi = [0, 0, 0, 0]
center_of_roi = [0, 0]
transposed_pos = [0, 0]
mask = np.nan


def drawFlow(img, flow, step=1, filtering=False):
    global min_dist
    h, w = flow.shape[:2]

    idx_y, idx_x = np.mgrid[step / 2:h:step, step / 2:w:step].astype(int)
    indicies = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

    if filtering:
        min_dist = find_optical_distance(flow, 99.5, h)
        print(min_dist)
        pass
    for x, y in indicies:
        dy, dx = flow[y, x].astype(int)
        if filtering and np.sqrt(dx ** 2 + dy ** 2) > min_dist:
            # print(min_dist)
            # x += transposed_pos[0]
            # y += transposed_pos[1]
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 255, 0), 1, cv2.LINE_AA)

        elif filtering is False:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 255, 0), 1, cv2.LINE_AA)


def masking_circle(unmask_frame):
    result_image = cv2.bitwise_and(unmask_frame, unmask_frame, mask=mask)

    return result_image


def find_optical_distance(pixel_dist, percent, pixel_size):
    global first_flag
    dists = []
    for i in range(pixel_size):
        for j in range(pixel_size):
            dists.append(np.sqrt(pixel_dist[i][j][0] ** 2 + pixel_dist[i][j][1] ** 2))

    if first_flag:
        plt.plot(dists)
        dists = np.array(dists).reshape(-1, 1)
        plt.show()

        first_flag = False
        kde = KernelDensity(bandwidth=1, kernel='gaussian')
        kde.fit(dists)

        # x 범위 생성
        x = np.linspace(min(dists), max(dists), 1000).reshape(-1, 1)

        # KDE 계산
        log_density = kde.score_samples(x)

        # 시각화
        plt.plot(x, np.exp(log_density), label='KDE')
        plt.scatter(dists, np.zeros_like(dists), color='red', label='Data Points')
        plt.title('Kernel Density Estimation')
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    return np.percentile(dists, percent)


while cap.isOpened():

    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if roi_flag:

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 300, param1=150, param2=50, minRadius=0, maxRadius=0)

        if circles is not None and circles.all():
            for i in circles[0]:
                # roi => (x_1, y_1), (x_2, y_2)
                roi = [int(i[0] - i[2]), int(i[1] - i[2]), int(i[0] + i[2]), int(i[1] + i[2])]
                center_of_roi = [(roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2]

                # transposed_pos => (x_1, y_1)
                transposed_pos = [roi[0], roi[1]]
                mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                mask = cv2.circle(mask, (int(i[0]), int(i[1])), int(i[2]) - 20, (255, 255, 255), -1)

                masked_image = masking_circle(gray)
            roi_flag = False

    else:
        # gray = gray[roi[0]:roi[2], roi[1]:roi[3]]
        if prev is None:
            prev = masking_circle(gray)
        else:
            gray = masking_circle(gray)
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
            flow = cv2.calcOpticalFlowFarneback(prev, gray, flow=1,
                                                pyr_scale=0.5,
                                                levels=5,
                                                winsize=5,
                                                iterations=7,
                                                poly_n=13,
                                                poly_sigma=1.1,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            drawFlow(frame, flow, step=2, filtering=True)
            prev = gray

    cv2.imshow('OpticalFlow-Farneback', frame)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
