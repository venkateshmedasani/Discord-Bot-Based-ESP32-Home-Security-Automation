import cv2 as cv
import numpy as np

img = cv.imread('./Images/face.jpg')
# cv.imshow('Face', img)


video = cv.VideoCapture(0)

while True:
    isTrue, frame = video.read()
    cv.imshow('Video', frame)

    if cv.waitKey(30) & 0xFF == ord('d'):
        break

video.release()

# img = cv.imread('./Images/face.jpg')
# cv.imshow('Face-original', img)


def rescale(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    new_dimensions = (width, height)
    return cv.resize(frame, new_dimensions, interpolation=cv.INTER_AREA)


# cv.imshow('Face-resized', rescale(img, 0.5))

# cv.waitKey(0)

# while True:
#     isTrue, frame = video.read()
#     cv.imshow('Original Video', frame)
#     cv.imshow('Resized Video', rescale(frame, 0.5))
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# video.release()


# blank = np.zeros((500, 500, 3), dtype='uint8')

# cv.rectangle(blank, (0, 0), (250, 250), (255, 0, 0), thickness=1)
# cv.circle(blank, (0, 0), 50, (255, 0, 0), 40)

# cv.imshow('Blank', blank)
# cv.imshow('Face', img)
# cv.imshow('Face-grayscale', cv.cvtColor(img, cv.COLOR_BGR2GRAY))


# cv.waitKey(0)
