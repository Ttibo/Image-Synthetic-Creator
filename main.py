import os
import cv2
import argparse
import random
from src.processors import *


# def random_homography(image, translation_range=(-0.3, 0.3), rotation_range=(-0.3, 0.3), shear_range=(-0.1, 0.1), perspective_range=(-0.1, 0.1)):
#     height, width = image.shape[:2]

#     # Définir les plages pour les paramètres de l'homographie
#     translation = np.random.uniform(translation_range[0], translation_range[1], size=(2,))
#     rotation = np.random.uniform(rotation_range[0], rotation_range[1])
#     shear = np.random.uniform(shear_range[0], shear_range[1])
#     perspective = np.random.uniform(perspective_range[0], perspective_range[1])

#     # Calculer la matrice d'homographie
#     random_homography_matrix = np.array([
#         [1 + shear, rotation, translation[0]],
#         [0, 1 + perspective, translation[1]],
#         [0, 0, 1]
#     ])

#     # Appliquer une transformation perspective à l'image
#     warped_image = cv2.warpPerspective(image, random_homography_matrix, (width, height))

#     return warped_image


def random_homography(image, max_translation=0.2, max_rotation=30, max_shear=0.2, max_scale=0.2):
    """
    Génère aléatoirement une homographie pour transformer une image.

    Args:
        image (tuple): La forme de l'image (hauteur, largeur).
        max_translation (float, optional): La valeur maximale de translation par rapport à la taille de l'image. Par défaut, 0.2.
        max_rotation (float, optional): La valeur maximale de rotation en degrés. Par défaut, 30.
        max_shear (float, optional): La valeur maximale de cisaillement. Par défaut, 0.2.
        max_scale (float, optional): La valeur maximale de mise à l'échelle. Par défaut, 0.2.

    Returns:
        numpy.ndarray: La matrice d'homographie.
    """
    # Taille de l'image
    height, width = image.shape[:2]

    # Générer aléatoirement les paramètres de transformation
    tx = np.random.uniform(-max_translation * width, max_translation * width)
    ty = np.random.uniform(-max_translation * height, max_translation * height)
    angle = np.random.uniform(-max_rotation, max_rotation)
    shear = np.random.uniform(-max_shear, max_shear)
    scale_x = np.random.uniform(1 - max_scale, 1 + max_scale)
    scale_y = np.random.uniform(1 - max_scale, 1 + max_scale)

    # Calculer la matrice d'homographie
    center_x = width / 2
    center_y = height / 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    shear_matrix = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
    scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)


    # Calculer la matrice d'homographie
    random_homography_matrix = np.array([
        [1 + shear, rotation, translation[0]],
        [0, 1 + perspective, translation[1]],
        [0, 0, 1]
    ])


    print(translation_matrix)
    print(shear_matrix)
    print(scale_matrix)
    print(rotation_matrix)

    print()
    print(translation_matrix.shape)
    print(shear_matrix.shape)
    print(scale_matrix.shape)
    print(rotation_matrix.shape)

    # Combiner les transformations
    homography_matrix = translation_matrix @ shear_matrix @ scale_matrix @ rotation_matrix

    return homography_matrix


# def random_homography(image, translation_range=(-0.1, 0.1), rotation_range=(-0.1, 0.1), shear_range=(-0.1, 0.1), perspective_range=(-0.1, 0.1)):
#     height, width = image.shape[:2]

#     # Définir les plages pour les paramètres de l'homographie
#     translation_x = np.random.uniform(*translation_range)
#     translation_y = np.random.uniform(*translation_range)
#     rotation = np.random.uniform(*rotation_range)
#     shear = np.random.uniform(*shear_range)
#     perspective = np.random.uniform(*perspective_range)

#     # Construire la matrice d'homographie
#     H_translation = np.array([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
#     H_rotation = np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
#     H_shear = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]])
#     H_perspective = np.array([[1, 0, 0], [0, 1, 0], [perspective, perspective, 1]])
    
#     # Combiner les transformations
#     H = np.dot(H_translation, np.dot(H_rotation, np.dot(H_shear, H_perspective)))

#     # Appliquer une transformation perspective à l'image
#     warped_image = cv2.warpPerspective(image, H, (width, height))

#     return warped_image




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--background', type=str , nargs='?' , help='Path background' , default="/home/badcoconut/Documents/Dataset/IMAGES/")
    parser.add_argument('--image', type=str, nargs='?', help='Path images' , default="/home/badcoconut/Documents/Dataset/DOCUMENTS/")
    parser.add_argument('--output', type=str, nargs='?', help='Path output' , default="/home/badcoconut/Documents/Dataset/SyntheticDataset/")
    parser.add_argument('--numbers', type=str, nargs='?', help='numbers synthetic views' , default=4 )
    parser.add_argument('--dimension' , type=int , nargs='?' , help='dimension max image' , default=640)

    args = parser.parse_args()

    print(args)


    # get images and background files
    images_ =  get_files(args.image)
    background_ =  get_files(args.background)

    print("images : {} , background {}".format(len(images_) , len(background_)))


    for i in range(100):
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
        transformed_image = random_homography(deformed_image)

        # Afficher l'image transformée
        cv2.imshow("Transformed Image", transformed_image)



        cv2.waitKey(0)  
        # cv2.destroyAllWindows()




