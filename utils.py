import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def show_images(images):
    plt.figure(figsize=(16, 8))
    count = len(images)
    for idx in range(count):
        subplot = plt.subplot(1, count, idx + 1)
        img = images[idx]
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap=cmap)
    plt.show()


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def get_points(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def draw_lines(img, p1, p2, p3, color=(255, 255, 0)):
    cv.line(img, p1, p2, color, 1, 0)
    cv.line(img, p2, p3, color, 1, 0)
    cv.line(img, p3, p1, color, 1, 0)
    return img


def correct_colours(img1, img2, landmarks):
    blur = 0.75
    left_eye = list(range(42, 48))
    right_eye = list(range(36, 42))

    blur_amount = blur * np.linalg.norm(
        np.mean(landmarks[left_eye], axis=0) -
        np.mean(landmarks[right_eye], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    img1_blur = cv.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur = img2_blur.astype(int)
    img2_blur += 128 * (img2_blur <= 1)

    result = img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def affine_matrix(points, face_landmarks, body_landmarks):
    for indices in points:
        face = np.vstack((face_landmarks[indices, :].T, [1, 1, 1]))
        body = np.vstack((body_landmarks[indices, :].T, [1, 1, 1]))
        mat = np.dot(face, np.linalg.inv(body))[:2, :]
        yield mat


def grid_coordinates(points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def bilinear_interpolate(img, coords):
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def get_masks(body, body_convexhull):
    body_mask = np.zeros_like(cv.cvtColor(body, cv.COLOR_BGR2GRAY))
    head_mask = cv.fillConvexPoly(body_mask, body_convexhull, 255)
    body_mask = cv.bitwise_not(head_mask)

    return head_mask, body_mask


def apply_blending(result, body, head_mask, body_convexhull):
    (x, y, w, h) = cv.boundingRect(body_convexhull)
    center = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone = cv.seamlessClone(result, body, head_mask, center, cv.NORMAL_CLONE)

    return seamlessclone