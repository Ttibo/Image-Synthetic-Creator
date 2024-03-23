import os
import cv2
import argparse
import random
import json
from src.processors import *
import matplotlib.pyplot as plt
from skimage import transform


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--background', type=str , nargs='?' , help='Path background' )
    parser.add_argument('--image', type=str, nargs='?', help='Path images' )
    parser.add_argument('--output', type=str, nargs='?', help='Path output' )
    parser.add_argument('--numbers', type=str, nargs='?', help='numbers synthetic views' , default=20 )
    parser.add_argument('--dimension' , type=int , nargs='?' , help='dimension max image' , default=640)

    args = parser.parse_args()

    print(args)

    images_ =  get_files(args.image)
    background_ =  get_files(args.background)
    create_folder(args.output)

    print("images : {} , background {}".format(len(images_) , len(background_)))

    for i in range(len(images_)):
        img_ = resizer(cv2.imread(images_[i]) , max_dimension = args.dimension)

        create_folder(args.output + "image_" + str(i))
        create_folder(args.output + "image_" + str(i) + "/images/")
        cv2.imwrite(args.output + "image_" + str(i) + "/image.jpeg" , img_)

        mesh_rows , mesh_cols = 10 , 10

        for j in range(args.numbers):
            create_folder(args.output + "image_" + str(i) + "/images/image_" + str(j))

            img = copy.deepcopy(img_)
            back = resize_to_fit(cv2.imread(background_[random.randint(0 , len(background_))]) , img)
            mask = np.ones((img.shape[0] , img.shape[1] , 1))*255

            mesh_points = create_mesh(img.copy(), mesh_rows, mesh_cols)
            transform_matrix = perspective_transform(img)

            img = apply_perspective_transform(img, transform_matrix)
            mask = apply_perspective_transform(mask, transform_matrix)
            perturbed_mesh_points = apply_homography_to_points(mesh_points , transform_matrix)

            img , perturbed_mesh_points , mask = distort_mesh_image(img , perturbed_mesh_points, mask)

            cv2.imwrite(args.output + "image_" + str(i) + "/images/image_" + str(j) + "/image.jpeg" , img)
            cv2.imwrite(args.output + "image_" + str(i) + "/images/image_" + str(j) + "/mask.jpeg" , mask)

            final_ = add_background(img, mask, back)

            cv2.imwrite(args.output + "image_" + str(i) + "/images/image_" + str(j) + "/final.jpeg" , final_)


            with open(args.output + "image_" + str(i) + "/images/image_" + str(j) + "/data.json", 'w') as json_file:
                json.dump({
                    'h' : transform_matrix.tolist(),
                    'mesh' : mesh_points.tolist(),
                    'perturbed_mesh' : perturbed_mesh_points.tolist(),

                }, json_file)
