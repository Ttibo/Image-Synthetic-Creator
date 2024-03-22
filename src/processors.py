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


def apply_homography_to_points(points, homography):
    transformed_points = points.reshape(-1 , 2)

    points_homogeneous = np.hstack([transformed_points, np.ones((len(transformed_points), 1))])

    transformed_points_homogeneous = np.dot(homography, points_homogeneous.T).T

    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
    
    return transformed_points.reshape(points.shape).astype(int)
    
def rotate_points_around_center(points, pivot, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    rotated_points = []
    for point in points:
        translated_point = point - pivot
        rotated_point = np.dot(rotation_matrix, translated_point)
        rotated_point += pivot
        rotated_points.append(rotated_point)

    return np.array(rotated_points)

def zoom_points_around_center(points, factor):
    barycenter = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
    
    moved_points = []
    for point in points:
        moved_x = point[0] + factor * (barycenter[0] - point[0])
        moved_y = point[1] + factor * (barycenter[1] - point[1])
        moved_points.append((moved_x, moved_y))
    
    return moved_points


def move_points_perspectives(points, intensity):
    moved_points = []
    for point in points:
        moved_points.append([point[0] + random.randint(-intensity, intensity),point[1] + random.randint(-intensity, intensity)])
    return moved_points


def perspective_transform(image):
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
    perturbed_four_points = move_points_perspectives(perturbed_four_points , intensity = 15)
    perturbed_four_points = zoom_points_around_center(perturbed_four_points , random.uniform(0.3 , 0.65))
    
    return cv2.getPerspectiveTransform(np.float32(four_points_base), np.float32(perturbed_four_points))

def apply_perspective_transform(img, transform_):
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
    rows, cols, _ = mesh_matrix.shape
    for i in range(rows):
        for j in range(cols):
            distorted_mesh_matrix[i, j, 0] += np.random.randint(-intensity, intensity + 1)
            distorted_mesh_matrix[i, j, 1] += np.random.randint(-intensity, intensity + 1)

    return distorted_mesh_matrix













def create_mesh(image, rows, cols):

    height, width = image.shape[:2]
    step_x = (width-1) // cols
    step_y = (height-1) // rows

    mesh_points = []
    for y in range(0, height+1, step_y):
        for x in range(0, width+1, step_x):
            mesh_points.append([x, y])

    return np.asarray(mesh_points).reshape(rows+1 , cols+1 , -1)

def display_mesh(image, mesh_matrix):
    mesh_image = image.copy()

    for row in mesh_matrix:
        for col in range(len(row) - 1):
            if row[col][0] is None:
                continue

            if row[col+1][0] is None:
                continue

            start_point = (row[col][0] ,row[col][1] )
            end_point = (row[col + 1][0] , row[col + 1][1])
            cv2.line(mesh_image, start_point, end_point, (0, 255, 0), 2)

    for col in range(len(mesh_matrix[0])):
        for row in range(len(mesh_matrix) - 1):
            if mesh_matrix[row][col][0] is None:
                continue

            if mesh_matrix[row+1][col][0] is None:
                continue

            start_point = (mesh_matrix[row][col][0] ,mesh_matrix[row][col][1] )
            end_point = (mesh_matrix[row + 1][col][0] , mesh_matrix[row + 1][col][1])
            cv2.line(mesh_image, start_point, end_point, (0, 255, 0), 2)

    return mesh_image


def apply_deformation_to_image(img, perturbed_mesh):
    h,w = img.shape[:2]

    perturbed_mesh_x = perturbed_mesh[:,0]
    perturbed_mesh_y = perturbed_mesh[:,1]
    
    perturbed_mesh_x =perturbed_mesh_x.reshape((h,w))
    perturbed_mesh_y =perturbed_mesh_y.reshape((h,w))

    remapped = cv2.remap(img, perturbed_mesh_x, perturbed_mesh_y, cv2.INTER_AREA) 
    return  remapped

def apply_deformation_to_mesh(img , mesh, perturbed_mesh):
    moved_mesh = []
    h,w = img.shape[:2]

    perturbed_mesh_x = perturbed_mesh[:,0]
    perturbed_mesh_y = perturbed_mesh[:,1]
    
    perturbed_mesh_x =perturbed_mesh_x.reshape((h,w))
    perturbed_mesh_y =perturbed_mesh_y.reshape((h,w))

    for m in mesh.reshape(-1 , 2):
        if m[0] >= w or m[1] >=h:
            moved_mesh.append([None , None])
            continue

        if m[0] < 0 or m[1] <0 :
            moved_mesh.append([None , None])
            continue

        pt_x = int(m[0] + (m[0] - perturbed_mesh_x[m[1]][m[0]]))
        pt_y = int(m[1] + (m[1] - perturbed_mesh_y[m[1]][m[0]]))

        moved_mesh.append([pt_x , pt_y])

    return np.asarray(moved_mesh).reshape(mesh.shape)

def apply_mouvements(ms):
    vidx = np.random.randint(np.shape(ms)[0])
    vtex = ms[vidx, :]
    xv  = ms - vtex
    
    mv = (np.random.rand(1,2) - 0.5)*10
    hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] +1) )
    hxv[:, :-1] = xv
    hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0],1))
    d = np.cross(hxv, hmv)
    d = np.absolute(d[:, 2])
    d = d / (np.linalg.norm(mv, ord=2))
    wt = d

    curve_type = np.random.rand(1)
    if curve_type > 0.3:
        alpha = np.random.rand(1) * 50 + 50
        wt = alpha / (wt + alpha)
    else:
        alpha = np.random.rand(1) + 1
        wt = 1 - (wt / 100 )**alpha
    msmv = mv * np.expand_dims(wt, axis=1)

    ms = ms + msmv
    ms = ms.astype(np.float32)
    
    return ms


def distort_mesh_image(img, mesh , mask = None):
    height, width = img.shape[:2]

    map_y, map_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    mr,mc = img.shape[:2]
    xx, yy = np.arange(0, mr, 1), np.arange(0, mc, 1)

    [Y, X] = np.meshgrid(xx, yy)
    ms = np.transpose(np.asarray([X.flatten('F'), Y.flatten('F')]), (1,0)).astype(np.float32)
    for i in range(5):
        ms = apply_mouvements(ms)

    img = apply_deformation_to_image(img, ms)
    mesh = apply_deformation_to_mesh(img , mesh, ms)

    if mask is not None:
        mask = apply_deformation_to_image(mask , ms)
        return img, mesh, mask

    return img, mesh








def resize_to_fit(image_A, image_B):

    height_A, width_A = image_B.shape[:2]
    height_B, width_B = image_A.shape[:2]

    ratio_height = height_A / height_B
    ratio_width = width_A / width_B

    resize_ratio = max(ratio_height, ratio_width)

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
    mask_ =  Image.fromarray(mask.astype(np.uint8)).convert('L')
    back_ =  Image.fromarray(cv2.resize(back , (img.shape[1] , img.shape[0])).astype(np.uint8))

    return np.asarray(Image.composite(img_, back_, mask_))