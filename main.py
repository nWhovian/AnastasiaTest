import argparse
import os

import cv2 as cv

from face_swap import face_swap

def parse_args():
    parser = argparse.ArgumentParser(description='Face swap application')
    parser.add_argument('--img1', required=True, help='Path to the first image')
    parser.add_argument('--img2', required=True, help='Path to the second image')
    parser.add_argument('--out', default='results', help='Path for storing output images')
    parser.add_argument('--predictor_path', type=str, default='assets/shape_predictor_68_face_landmarks.dat',
                        help='Path to the trained shape predictor model')

    return parser.parse_args()

def main():
    args = parse_args()

    img_1 = cv.imread(args.img1)
    img_2 = cv.imread(args.img2)

    face_1_body_2, face_2_body_1 = face_swap(img_1, img_2)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    name_1 = os.path.join(args.out, 'img2_face1.jpg')
    name_2 = os.path.join(args.out, 'img1_face2.jpg')

    cv.imwrite(name_1, face_1_body_2)
    cv.imwrite(name_2, face_2_body_1)


if __name__ == '__main__':
    main()
