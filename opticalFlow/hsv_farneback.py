import numpy as np
import cv2 as cv

cap = cv.VideoCapture('raw_data/IT248053l_25_50fps_binning2_sample5(6).avi')
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while (1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff

    if k == 27:
        break

    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
        prvs = next
cv.destroyAllWindows()

'''
import cv2
import numpy as np

def drawFlow(img, flow, step=4):
    h, w = img.shape[:2]

    idx_y, idx_x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    indicies = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

    for x, y in indicies:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        dx, dy = flow[y, x].astype(int)
        cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)

prev = None

cap = cv2.VideoCapture('raw_data/IT261516_25_50fps_binning2_sample2(1).avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev is None:
        prev = gray
    else:
        flow = cv2.calcOpticalFlowFarneback(prev, gray, 0.5, 0.5, 3, 30, 7, 13, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        drawFlow(frame, flow)
        prev = gray

    cv2.imshow('OpticalFlow-Farneback', frame)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()


'''
