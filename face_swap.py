import dlib
import scipy.spatial as spatial

from utils import *


def detect_bboxes(img_gray):
    # predict face bounding boxes with dlib
    detector = dlib.get_frontal_face_detector()
    faces = detector(img_gray)
    return faces


def detect_landmarks(img_gray, faces):
    # predict face landmarks with dlib shape predictor
    predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
    points_list = []
    for bbox in faces:
        points = predictor(img_gray, bbox)
        points = get_points(points)
        points_list.append(points)

    return points_list


def construct_triangulation(landmarks):
    # build Delaunay triangulation
    convexhull = cv.convexHull(landmarks)
    delaunay = spatial.Delaunay(landmarks)

    return delaunay, convexhull


def new_face_generation(face, body, face_landmarks, body_landmarks, delaunay):
    # calculate affine transformation matrix for each
    # triangle vertex from face_landmarks to body_landmarks
    transformed_triangles = np.asarray(list(affine_matrix(delaunay.simplices, face_landmarks, body_landmarks)))

    h, w = body.shape[:2]
    new_face = np.zeros((h, w, 3), dtype=np.uint8)

    # warp each triangle from source image (face)
    # construct a new_face - warped source face (link warped triangles together)

    triangle_coords = grid_coordinates(body_landmarks)
    triangle_indices = delaunay.find_simplex(triangle_coords)

    for index in range(len(delaunay.simplices)):
        coords = triangle_coords[triangle_indices == index]
        num_coords = len(coords)
        out_coords = np.dot(transformed_triangles[index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        new_face[y, x] = bilinear_interpolate(face, out_coords)

    return new_face


def swap_and_blend(face, body, face_landmarks, body_landmarks, delaunay, body_convexhull):
    """
    :param face: the image from which we take the face
    :param body: the image to which we apply the face swapping
    :param face_landmarks: face landmarks for source image
    :param body_landmarks: face landmarks for destination image
    :param delaunay: triangulation of the destination image (body)
    :param body_convexhull: convexhull of the destination image (body)
    :return: image with the replaced face
    """
    new_face = new_face_generation(face, body, face_landmarks, body_landmarks, delaunay)

    # body and head masks for the destination image
    head_mask, body_mask = get_masks(body, body_convexhull)

    # apply color correction:
    # prev_face - original face from destination image
    # new_face - warped source face

    prev_face = cv.bitwise_and(body, body, mask=head_mask)
    new_face = correct_colours(prev_face, new_face, body_landmarks)

    # add a warped source face to a body image
    body_maskless = cv.bitwise_and(body, body, mask=body_mask)
    result = cv.add(body_maskless, new_face)

    # apply Poisson blending
    result = apply_blending(result, body, head_mask, body_convexhull)

    return result


def compute_all_coordinates(img_gray):
    """
    Compute landmark coordinates, triangulation and convexhull for each face
    :return: points_list (landmark coordinates), triangles_list (delaunay triangulation list), convexhull_list
    """
    faces = detect_bboxes(img_gray)
    points_list = detect_landmarks(img_gray, faces)
    triangles_list = []
    convexhull_list = []

    for face_landmarks in points_list:
        delaunay, convexhull = construct_triangulation(face_landmarks)
        triangles_list.append(delaunay)
        convexhull_list.append(convexhull)

    return points_list, triangles_list, convexhull_list


def face_swap(img_1, img_2):
    """
    Method for face swapping between two images: img_1 and img_2
    """
    img_gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    img_gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    # Compute landmark coordinates, triangulation and convexhull for each face
    points_list_1, triangles_list_1, convexhull_list_1 = compute_all_coordinates(img_gray_1)
    points_list_2, triangles_list_2, convexhull_list_2 = compute_all_coordinates(img_gray_2)

    # in case more than one face is detected in the image
    faces_count = min(len(points_list_1), len(points_list_2))

    for i in range(faces_count):
        landmarks_1 = points_list_1[i]
        landmarks_2 = points_list_2[i]

        delaunay_1 = triangles_list_1[i]
        delaunay_2 = triangles_list_2[i]

        convexhull_1 = convexhull_list_1[i]
        convexhull_2 = convexhull_list_2[i]

        # swap faces, apply color correction and blending
        face_1_body_2 = swap_and_blend(img_1, img_2, landmarks_1, landmarks_2, delaunay_2, convexhull_2)
        face_2_body_1 = swap_and_blend(img_2, img_1, landmarks_2, landmarks_1, delaunay_1, convexhull_1)

        return face_1_body_2, face_2_body_1
