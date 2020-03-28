import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import json
import sys
from mpl_toolkits.mplot3d import Axes3D

def load_camera_parameters():
    fs_l = cv.FileStorage("../../parameters/left_camera_intrinsics.xml", cv.FILE_STORAGE_READ)
    left_camera_intrinsics = fs_l.getNode("camera_intrinsic").mat()
    left_camera_distortion = fs_l.getNode("camera_distortion").mat()

    fs_r = cv.FileStorage("../../parameters/right_camera_intrinsics.xml", cv.FILE_STORAGE_READ)
    right_camera_intrinsics = fs_r.getNode("camera_intrinsic").mat()
    right_camera_distortion = fs_r.getNode("camera_distortion").mat()
    
    return left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion


def get_undistorted_image(original_image, intrinsic_matrix, distortion_coefficients,write_to_file=False):
    h, w = original_image.shape[:2]
    mapx, mapy = cv.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, None, intrinsic_matrix, (w,h), 5)
    undistorted_img = cv.remap(original_image, mapx, mapy, cv.INTER_LINEAR)
    if(write_to_file):
        cv.imwrite("../../output/task_7/undistorted_image.png", undistorted_img)
    return undistorted_img

def plot_image(image, write_to_file = True, filename = "Image_plot.png"):
    plt.figure(figsize=(24, 9))
    plt.imshow(image, cmap = 'gray')
    if(write_to_file):
        plt.savefig("../../output/task_7/"+ filename)
    plt.show()

def plot_images_2(image1, image2, write_to_file = True, filename = "Plotted_Images.png"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap = 'gray')
    ax1.set_title('Image 1', fontsize=30)
    ax2.imshow(image2, cmap = 'gray')
    ax2.set_title('Image 2', fontsize=30)
    if(write_to_file):
        plt.savefig("../../output/task_7/"+ filename)
    plt.show()


def get_localised_max_keypoints(keyp_desc, window_size = 10):
    kp_map = {}
    for kp_s_i in keyp_desc:
        window_x = kp_s_i[0].pt[0]//window_size
        window_y = kp_s_i[0].pt[1]//window_size
        if(window_x in kp_map):
            if(window_y in kp_map[window_x]):
                kp_map[window_x][window_y].append(kp_s_i)
            else:
                kp_map[window_x][window_y] = [kp_s_i]
        else:
            kp_map[window_x] = {}
    kp_local_maximas = []
    for hor_dict in kp_map.values():
        for vert_dict in hor_dict.values():
            kp_local_maximas.append(max(vert_dict, key=lambda kp: kp[0].response))
    return kp_local_maximas

def join_arrays(keypoints, descriptors):
    keyp_desc = []
    for i in range(len(keypoints)):
        keyp_desc.append((keypoints[i],descriptors[i]))
    return keyp_desc

def split_arrays(keyp_desc):
    keypoints = []
    descriptors = []
    for i in range(len(keyp_desc)):
        keypoints.append(keyp_desc[i][0])
        descriptors.append(keyp_desc[i][1])
    return np.array(keypoints), np.array(descriptors)

def scatter_plot(triangulated_pts):
    x,y,z,w = triangulated_pts
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)

    fig = plt.figure(figsize=(24, 9))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x/w,y/w,z/w)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def scatter_plot_with_cam(traingulated_pts, R, t, write_to_file = True, file_name = "scatter plot.png"):
    x,y,z,w = triangulated_pts
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)

    fig = plt.figure(figsize=(24, 9))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x/w,y/w,z/w)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # camera principle axis representation
    start = np.array([0,0,0,1])
    end = np.array([0,0,0.5,1])
    c1 = np.column_stack([start,end])
    T = np.column_stack([R,t])
    rotated_c1 = np.matmul(T, c1)
    ax.plot([c1[0,0],c1[0,1]], [c1[1,0],c1[1,1]], [c1[2,0],c1[2,1]])
    ax.text(c1[0,0],c1[1,0],c1[2,0]-0.05, "0")
    ax.plot([rotated_c1[0,0],rotated_c1[0,1]], [rotated_c1[1,0],rotated_c1[1,1]], [rotated_c1[2,0],rotated_c1[2,1]])
    ax.text(rotated_c1[0,0],rotated_c1[1,0],rotated_c1[2,0]-0.05, "1")
    if(write_to_file):
        plt.savefig("../../output/task_7/"+ file_name)
    plt.show()


def triangulate_scatter_plot(matches, kp_l, kp_r, P1, P2, R, t):
    points1 = np.zeros([len(matches),1,2])
    points2 = np.zeros([len(matches),1,2])
    for i in range(len(matches)):
        points1[i][0] = np.array(kp_l[matches[i].queryIdx].pt)
        points2[i][0] = np.array(kp_r[matches[i].trainIdx].pt)
    
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1_re = np.row_stack((points1[:,0,0], points1[:,0,1]))
    points2_re = np.row_stack((points2[:,0,0], points2[:,0,1]))

    triangulated_pts = cv.triangulatePoints(P1,P2,points1_re,points2_re)
    scatter_plot_with_cam(triangulated_pts, R, t)

def get_local_max_keypoints(image1_kp, image1_desc):
    kp_desc_l = join_arrays(image1_kp, image1_desc)
    kp_desc_l_max = get_localised_max_keypoints(kp_desc_l)
    kp_l_max, desc_l_max = split_arrays(kp_desc_l_max)
    return kp_l_max, desc_l_max

def save_results_into_file(essential_matrix, R, t, camera):
    with open("../../parameters/task_7_{0}_{1}_{2}.txt".format(camera[0], camera[1], camera[2]),"w") as f:
        f.write("Essential Matrix:\n")
        f.write(str(essential_matrix)+"\n\n")
        f.write("Rotation:\n")
        f.write(str(R)+"\n\n")
        f.write("Translation:\n")
        f.write(str(t)+"\n")


if __name__ == "__main__":
    IMG1_INDEX = 0
    IMG2_INDEX = 1
    side = "left"

    if(len(sys.argv) == 4):
        side = sys.argv[1].lower()
        IMG1_INDEX = sys.argv[2]
        IMG2_INDEX = sys.argv[3]

    LEFT = False if side == "right" else True


    side = "left" if LEFT else "right"
    img1_path = "../../images/task_7/{0}_{1}.png".format(side,IMG1_INDEX)
    img2_path = "../../images/task_7/{0}_{1}.png".format(side,IMG2_INDEX)
    # read images
    # img1 = cv.imread("../../images/task_7/left_1.png", cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread("../../images/task_7/left_2.png", cv.IMREAD_GRAYSCALE)

    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
    
    # get camera parameters
    left_intrinsic_matrix, left_distortion_coeffecients, right_intrinsic_matrix, right_distortion_coeffecients = load_camera_parameters()

    camera_intrinsic_matrix = left_intrinsic_matrix if LEFT else right_intrinsic_matrix
    camera_distortion_coeffecients = left_distortion_coeffecients if LEFT else right_distortion_coeffecients
    
    # undistort images
    undistorted_img_1 = get_undistorted_image(img1,camera_intrinsic_matrix, camera_distortion_coeffecients)
    undistorted_img_2 = get_undistorted_image(img2,camera_intrinsic_matrix, camera_distortion_coeffecients)
    plot_images_2(undistorted_img_1, undistorted_img_2, write_to_file=True, filename="Unidistorted_Images_{0}_{1}_{2}.png".format(side, IMG1_INDEX, IMG2_INDEX))
    
    # Get feature points
    orb = cv.ORB_create()
    kp_1, des_1 = orb.detectAndCompute(undistorted_img_1,None)
    kp_2, des_2 = orb.detectAndCompute(undistorted_img_2,None)
    
    # Obtain local maximas
    image1_kp_max, image1_desc_max = get_local_max_keypoints(kp_1, des_1)
    image2_kp_max, image2_desc_max = get_local_max_keypoints(kp_2, des_2)

    # Plot keypoints
    kp1_img = cv.drawKeypoints(undistorted_img_1, image1_kp_max, undistorted_img_1.copy(), color=(0,255,0), flags=0)
    kp2_img = cv.drawKeypoints(undistorted_img_2, image2_kp_max, undistorted_img_2.copy(), color=(0,255,0), flags=0)
    plot_images_2(kp1_img, kp2_img, write_to_file=True, filename="Keypoint_Images_{0}_{1}_{2}.png".format(side, IMG1_INDEX, IMG2_INDEX))

    # Match the new keypoints on both the images
    bf = cv.BFMatcher_create(normType = cv.NORM_HAMMING)
    matches = bf.match(image1_desc_max,image2_desc_max)
    image_with_all_matches = cv.drawMatches(undistorted_img_1,image1_kp_max,undistorted_img_2,image2_kp_max,matches, None, flags=2)
    plot_image(image_with_all_matches, True, "All_matches_{0}_{1}_{2}.png".format(side, IMG1_INDEX, IMG2_INDEX))


    # Get corresponding feature image points on both the images.
    img1_matches = []
    img2_matches = []
    for match in matches:
        img1_matches.append(image1_kp_max[match.queryIdx].pt)
        img2_matches.append(image2_kp_max[match.trainIdx].pt)
    img1_match_pts = np.asarray(img1_matches)
    img2_match_pts = np.asarray(img2_matches)

    # Calculate Essential matrix and get the mask for filtering matches
    essential_mat, mask = cv.findEssentialMat(img1_match_pts, img2_match_pts, camera_intrinsic_matrix)

    # Filter matches using masks and get the new image points and matches
    new_matches = []
    new_img1_matches = []
    new_img2_matches = []
    for i in range(len(mask)):
        if(mask[i][0]==1):
            new_matches.append(matches[i])
            new_img1_matches.append(img1_matches[i])
            new_img2_matches.append(img2_matches[i])
    new_img1_matches = np.asarray(new_img1_matches)
    new_img2_matches = np.asarray(new_img2_matches)

    # Draw Matches
    filtered_matches_img = cv.drawMatches(undistorted_img_1,image1_kp_max,undistorted_img_2,image2_kp_max, new_matches, None, flags=2)
    plot_image(filtered_matches_img, True, "Filtered_matches_{0}_{1}_{2}.png".format(side, IMG1_INDEX, IMG2_INDEX))

    # Triagulation method 1
    # Recover the camera pose using the essential matrix and get the triangulated points
    retval, R, t, mask2, triangulated_pts = cv.recoverPose(essential_mat, new_img1_matches, new_img2_matches, camera_intrinsic_matrix, 10, None, None, None)
    T = np.column_stack([R,t])
    T = np.append(T, [[0,0,0,1]], axis=0)
    inv_T = np.linalg.inv(T)
    new_r = inv_T[:3,:3]
    new_t = inv_T[:3,3]
    save_results_into_file(essential_mat, new_r, new_t, (side, IMG1_INDEX, IMG2_INDEX))
    scatter_plot_with_cam(triangulated_pts, new_r, new_t, True, "Pose_View_{0}_{1}_{2}.png".format(side, IMG1_INDEX, IMG2_INDEX))

    # Triagulation method 2. Calculate P1 and P2 and use triangulate points method
    # P1 = np.column_stack((np.identity(3,dtype=np.float64),np.zeros([3,1],dtype=np.float64)))
    # P1 = np.matmul(camera_intrinsic_matrix, P1)
    # P2 = np.column_stack((R, t))
    # P2 = np.matmul(camera_intrinsic_matrix, P2)
    # triangulate_scatter_plot(new_matches, image1_kp_max, image2_kp_max, P1, P2, R, t)