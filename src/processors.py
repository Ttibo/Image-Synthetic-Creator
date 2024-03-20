import cv2
import os
import sys 
import math
import numpy as np
from PIL import Image
import imutils
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data



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
    
    
    plt.figure()
    plt.imshow(rr_)
    
    return h_
