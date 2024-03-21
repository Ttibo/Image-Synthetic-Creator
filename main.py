import os
import cv2
import argparse
import random
from src.processors import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--background', type=str , nargs='?' , help='Path background' , default="/Users/thibaultlelong/Documents/Dataset/IMAGES/")
    parser.add_argument('--image', type=str, nargs='?', help='Path images' , default="/Users/thibaultlelong/Documents/Dataset/DOCUMENTS/")
    parser.add_argument('--output', type=str, nargs='?', help='Path output' , default="/Users/thibaultlelong/Documents/Dataset/SyntheticDataset/")
    parser.add_argument('--numbers', type=str, nargs='?', help='numbers synthetic views' , default=4 )
    parser.add_argument('--dimension' , type=int , nargs='?' , help='dimension max image' , default=640)

    args = parser.parse_args()

    print(args)


    # get images and background files
    images_ =  get_files(args.image)
    background_ =  get_files(args.background)

    print("images : {} , background {}".format(len(images_) , len(background_)))


    for i in range(4):
        # open image and resize and creat mask
        img = resizer(cv2.imread(images_[i]) , max_dimension = args.dimension)
        back = resize_to_fit(cv2.imread(background_[i]) , img)
        mask = np.ones((img.shape[0] , img.shape[1]))





        # Paramètres de la maille
        mesh_rows = 10  # Nombre de lignes dans la maille
        mesh_cols = 5  # Nombre de colonnes dans la maille

        # Créer la maille sur l'image
        mesh_points = create_mesh(img.copy(), mesh_rows, mesh_cols)
        meshed_image = display_mesh(img , mesh_points)
        cv2.imshow('meshed_image' , meshed_image )

        # # # Perturber la maille
        disturbed_mesh_points = distort_mesh(mesh_points, intensity=10)
        disturbed_mesh_points_image = display_mesh(img , disturbed_mesh_points )
        cv2.imshow('disturbed_mesh_points_image' , disturbed_mesh_points_image )

        # # Déformer l'image en utilisant la maille perturbée
        deformed_image = apply_mesh_to_image(img, mesh_points , disturbed_mesh_points)
        cv2.imshow('deformed_image' , deformed_image)


        # Générer aléatoirement une homographie
        transform_matrix = perspective_transform(deformed_image)
        transformed_image = apply_perspective_transform(deformed_image, transform_matrix)

        # Afficher l'image transformée
        cv2.imshow("Transformed Image", transformed_image)



        cv2.waitKey(0)  
        # cv2.destroyAllWindows()




