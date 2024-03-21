import cv2
import os
import sys 
import math
import numpy as np
from PIL import Image
import imutils
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
import random
import copy


def rotate_points(points):
    # Rotation aléatoire des points
    rotation_index = random.choice([0, 1, 2, 3])
    four_points_rotated = points[rotation_index:] + points[:rotation_index]
    
    return four_points_rotated

def random_rotation_homography(theta_range=(-np.pi/4, np.pi/4), width=640, height=480):
    """
    Génère aléatoirement une homographie représentant une rotation.

    Args:
        theta_range (tuple): Plage de valeurs pour l'angle de rotation en radians.
        width (int): Largeur de l'image.
        height (int): Hauteur de l'image.

    Returns:
        numpy.ndarray: Matrice de l'homographie représentant la rotation.
    """
    # Générer un angle de rotation aléatoire dans la plage spécifiée
    theta = np.random.uniform(theta_range[0], theta_range[1])

    # Calcul des composantes de l'homographie
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Matrice d'homographie pour une rotation
    homography = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    return homography

def apply_homography_to_points(points, homography):
    """
    Applique une homographie à une liste de points.

    Args:
        points (numpy.ndarray): Liste de points sous forme de tableau NumPy de dimensions (n, 2),
                                où n est le nombre de points et chaque ligne contient les coordonnées (x, y) d'un point.
        homography (numpy.ndarray): Matrice d'homographie 3x3.

    Returns:
        numpy.ndarray: Liste de points transformés après l'application de l'homographie.
    """
    # Ajouter une colonne de 1 pour chaque point pour rendre les coordonnées homogènes
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])

    # Appliquer l'homographie à tous les points en une seule opération
    transformed_points_homogeneous = np.dot(homography, points_homogeneous.T).T

    # Normaliser les coordonnées homogènes en coordonnées cartésiennes
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]

    return transformed_points
    
def rotate_points_around_center(points, pivot, angle):
    # Conversion de l'angle en cos et sin
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Matrice de rotation
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Pour chaque point, appliquer la rotation autour du pivot
    rotated_points = []
    for point in points:
        # Translation pour faire tourner le point autour de l'origine
        translated_point = point - pivot
        # Rotation du point
        rotated_point = np.dot(rotation_matrix, translated_point)
        # Translation pour ramener le point à sa position d'origine
        rotated_point += pivot
        rotated_points.append(rotated_point)

    return np.array(rotated_points)

def zoom_points_around_center(points, factor):
    # Calcul du barycentre des points
    barycenter = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
    
    # Déplacement des points vers ou depuis le barycentre
    moved_points = []
    for point in points:
        moved_x = point[0] + factor * (barycenter[0] - point[0])
        moved_y = point[1] + factor * (barycenter[1] - point[1])
        moved_points.append((moved_x, moved_y))
    
    return moved_points




def perspective_transform(image):
    # Détermination des points de base
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    min_side_length = min(image.shape[1], image.shape[0]) // 3
    half_min_side_length = min_side_length / 2
    gap_x = center_x - half_min_side_length
    gap_y = center_y - half_min_side_length
    
    top_point = (gap_x, gap_y)
    left_point = (gap_x + min_side_length, gap_y)
    bottom_point = (gap_x + min_side_length, gap_y + min_side_length)
    right_point = (gap_x, gap_y + min_side_length)
    
    four_points_base = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = rotate_points_around_center(four_points_base , np.asarray([center_x , center_y]) , random.randint(-180 , 180))
    
    # Génération de points perturbés pour la perspective
    for point in perturbed_four_points:
        point = [point[0] + random.randint(-half_min_side_length // 2, half_min_side_length // 2),point[1] + random.randint(-half_min_side_length // 2, half_min_side_length // 2)]
        

    # perturbed_four_points = zoom_points_around_center(perturbed_four_points , random.uniform(0.3 , 0.5))
    
    return cv2.getPerspectiveTransform(np.float32(four_points_base), np.float32(perturbed_four_points))

def apply_perspective_transform(img, transform_):
    # Application de la transformation sur l'image et le masque
    warped_image = cv2.warpPerspective(img, transform_, (img.shape[1], img.shape[0]))
    return warped_image









def create_folder(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)

def get_files(path,add_path = True, pattern_ = None):
    if os.path.exists(path) == False:
        return []

    files_ =  os.listdir(path)
    files_ = [file_ for file_ in files_ if file_ not in [".DS_Store", "._.DS_Store"]]
    
    if pattern_ != None:
        print("pattern")
        files_ = [file_ + pattern_ for file_ in files_]

    if add_path:
        return [path + file_ for file_ in files_]
    else: 
        return files_


def distort_mesh(mesh_matrix, intensity=10):
    distorted_mesh_matrix = mesh_matrix.copy()
    print(mesh_matrix.shape)
    rows, cols, _ = mesh_matrix.shape

    # Appliquer une déformation à chaque point de la maille
    for i in range(rows):
        for j in range(cols):
            # Ajouter un bruit aléatoire à chaque coordonnée (x, y) de la maille
            distorted_mesh_matrix[i, j, 0] += np.random.randint(-intensity, intensity + 1)
            distorted_mesh_matrix[i, j, 1] += np.random.randint(-intensity, intensity + 1)

    return distorted_mesh_matrix

def apply_mesh_to_image(image, mesh_src, mesh_dst):
    tform = PiecewiseAffineTransform()
    tform.estimate(mesh_src.reshape(-1 , 2), mesh_dst.reshape(-1 , 2))
    return warp(image, tform, output_shape=(image.shape[0], image.shape[1]))

def create_mesh(image, rows, cols):

    height, width = image.shape[:2]
    step_x = width // cols
    step_y = height // rows

    # Créer les points de la maille
    mesh_points = []
    for y in range(0, height+1, step_y):
        for x in range(0, width+1, step_x):
            mesh_points.append([x, y])

    return np.asarray(mesh_points).reshape(rows+1 , cols+1 , -1)

def display_mesh(image, mesh_matrix):

    mesh_image = image.copy()

    # Dessiner les lignes horizontales
    for row in mesh_matrix:
        for col in range(len(row) - 1):
            start_point = tuple(row[col])
            end_point = tuple(row[col + 1])
            cv2.line(mesh_image, start_point, end_point, (0, 255, 0), 1)

    # Dessiner les lignes verticales
    for col in range(len(mesh_matrix[0])):
        for row in range(len(mesh_matrix) - 1):
            start_point = tuple(mesh_matrix[row][col])
            end_point = tuple(mesh_matrix[row + 1][col])
            cv2.line(mesh_image, start_point, end_point, (0, 255, 0), 1)

    return mesh_image


def resize_to_fit(image_A, image_B):

    height_A, width_A = image_B.shape[:2]
    height_B, width_B = image_A.shape[:2]

    # Calculer le rapport de redimensionnement pour chaque dimension
    ratio_height = height_A / height_B
    ratio_width = width_A / width_B

    # Sélectionner le rapport de redimensionnement le plus petit pour que l'image B rentre dans l'image A
    resize_ratio = max(ratio_height, ratio_width)

    # Redimensionner l'image A
    resized_A = cv2.resize(image_A, (int(width_B * resize_ratio), int(height_B * resize_ratio)))

    return resized_A

def resizer(image, max_dimension):
    height, width = image.shape[:2]

    if max(height, width) <= max_dimension:
        return image

    if height > width:
        return imutils.resize(image, height=max_dimension)
    else:
        return imutils.resize(image, width=max_dimension)

def add_background(img , mask , back):
    
    img_ =  Image.fromarray(img.astype(np.uint8))
    back_ =  Image.fromarray(mask.astype(np.uint8))
    mask_ =  Image.fromarray(cv2.resize(back , (img.shape[1] , img.shape[0])).astype(np.uint8))

    return np.asarray(Image.composite(img_, back_, mask_))

def normalize(center_ , centerS , shapeS, reshapeS):
    gh , gw = center_[0] - centerS[0] , center_[1] - centerS[1]
    cp_ = [(shapeS[0] // 2) + gh , (shapeS[1] // 2) + gw ]
    r_ = [shapeS[0] / reshapeS[0] , shapeS[1] / reshapeS[1]]
    cp_[0] /= r_[0]
    cp_[1] /= r_[1]
    
    return cp_

def denormalize(center_ , centerD , shapeD, reshapeD):
    r_ = [shapeD[0] / reshapeD[0] , shapeD[1] / reshapeD[1]]
    gw_, gh_ = centerD[0] - shapeD[0]//2 ,  centerD[1] - shapeD[1]//2
    center_[0] *= r_[0]
    center_[1] *= r_[1]
    center_[0] += gw_
    center_[1] += gh_
    return center_

def denormalize_homography(h,pcenter,pshape,preshape,wcenter,wshape,wreshape , img):
    
    # define borders
    # rr_ = copy.deepcopy(img)
    pborders_ = [[0 , 0] , [0 , preshape[1]] , [preshape[0] , preshape[1]] , [preshape[0] , 0]]
    # borders_ = [[0 , 0] , [0 , pshape[1]] , [pshape[0] , pshape[1]] , [pshape[0] , 0]]
    
    borders_ = [[pcenter[0] - pshape[0]//2 , pcenter[1] - pshape[1]//2] ,
                [pcenter[0] - pshape[0]//2 , pcenter[1] + pshape[1]//2] ,
                [pcenter[0] + pshape[0]//2 , pcenter[1] + pshape[1]//2] ,
                [pcenter[0] + pshape[0]//2 , pcenter[1] - pshape[1]//2]]
    
    pts_ = []
    for pt in pborders_:
        # pt_ = processors.project_point(h , pt , pcenter , pshape, preshape , wcenter , wshape, wreshape)
        
        pt_ = np.float32(pt).reshape(-1, 1, 2)
        pt_ = cv2.perspectiveTransform(pt_, h)[0,0,:]
        pt_ = denormalize(pt_ , wcenter , wshape , wreshape)
        
        # rr_ = cv2.circle(rr_ , ( int(pt_[0]) , int(pt_[1]) ) , 20 , (255, 0 , 0) , 3)
        pts_.append(pt_)
        
    src_pts = np.float32(borders_).reshape(-1,1,2)
    dst_pts = np.float32(pts_).reshape(-1,1,2)
    
    h_, _ = cv2.findHomography(src_pts, dst_pts) # prosac
    
    return h_
    