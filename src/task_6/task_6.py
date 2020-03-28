import numpy as np
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 1)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,0,255), 1)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,255), 1)
    return img

def plot(objp, T_s, img_idx, write_to_file = True, file_name = "camera_poses.png"):
    print('plotting awesome cameras in 3D')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(objp[0],objp[1])
    ax.plot(objp[1],objp[2])
    ax.plot(objp[2],objp[3])
    ax.plot(objp[3],objp[0])
    for i,T in zip(img_idx, T_s):
        start = np.array([0,0,0,1])
        end = np.array([0,0,0.5,1])
        c1 = np.column_stack([start,end])
        rotated_c1 = np.matmul(T, c1)
        ax.plot([rotated_c1[0,0],rotated_c1[0,1]], [rotated_c1[1,0],rotated_c1[1,1]], [rotated_c1[2,0],rotated_c1[2,1]])
        ax.text(rotated_c1[0,0],rotated_c1[1,0],rotated_c1[2,0]-0.05, i)
    if(write_to_file):
        plt.savefig("../../output/task_6/"+ file_name)
    plt.show()

def pose_estimate(M1,dist1,M2,dist2,camera):
    objp = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    M1 = np.asarray(M1)
    dist1 = np.asarray(dist1)
    M2 = np.asarray(M2)
    dist2 = np.asarray(dist2)
    if camera == 'left':
      camera_matrix=np.array(M1,dtype = "double")
      dist = np.asarray(dist1)
      images=glob.glob("../../images/task_6/left_*.png")
    else:
      camera_matrix=np.array(M2,dtype = "double")
      dist = np.asarray(dist2)
      images=glob.glob("../../images/task_6/right_*.png")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    xloc = []
    yloc = []
    zloc = []
    T_s = []
    img_idx = []
    i=0
    f = open("../../parameters/task_6_"+camera+".txt","w")
    while(i<len(images)):
        try:
            frame = cv2.imread(images[i])
            image_index = images[i][-6:-4].strip('_')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.aruco.detectMarkers(gray,dictionary)
            if len(res[0]) > 0:
                cv2.aruco.drawDetectedMarkers(frame,res[0],res[1],borderColor=(0, 255, 0))
            imgWithAruco=frame
            x1 = (res[0][0][0][0][0], res[0][0][0][0][1])
            x2 = (res[0][0][0][1][0], res[0][0][0][1][1])
            x3 = (res[0][0][0][2][0], res[0][0][0][2][1])
            x4 = (res[0][0][0][3][0], res[0][0][0][3][1])
            imgp = np.asarray([x1,x2,x3,x4])

            cv2.line(gray, x1, x2, (255, 0, 0), 1)
            cv2.line(gray, x2, x3, (255, 0, 0), 1)
            cv2.line(gray, x3, x4, (255, 0, 0), 1)
            cv2.line(gray, x4, x1, (255, 0, 0), 1)
            _,rvecs, tvecs = cv2.solvePnP(objp, imgp, camera_matrix, dist)

            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist)

            ZYX,_ = cv2.Rodrigues(rvecs)

            totalrotmax=np.array([[ZYX[0,0],ZYX[0,1],ZYX[0,2],tvecs[0]],[ZYX[1,0],ZYX[1,1],ZYX[1,2],tvecs[1]],[ZYX[2,0],ZYX[2,1],ZYX[2,2],tvecs[2]],[0.,0.,0.,1.]],dtype='float')
            WtoC=np.mat(totalrotmax)
            inverserotmax=np.linalg.inv(totalrotmax)
            # print(inverserotmax)
            xloc.append(inverserotmax[0][3])
            yloc.append(inverserotmax[1][3])
            zloc.append(inverserotmax[2][3])
            T_s.append(inverserotmax)
            R = inverserotmax[:3,:3]
            t = inverserotmax[:3,3]
            img_idx.append(image_index)

            filename='../../output/task_6/img_'+camera+str(image_index)+'.jpg'
            # cv2.imshow('frame',frame)
            # print('image {} saved in output folder'.format(img_name) )
            f.write(camera+"_"+image_index+'.png\n')
            f.write('Translation vector: \n')
            f.write(str(t)+'\n')
            f.write('Rotation matrix: \n')
            f.write(str(R)+'\n\n\n')
            cv2.imwrite(filename, frame)
            print('image {} saved in output folder'.format(image_index) )
            i+=1
        except:
            print('skipping image {}'.format(image_index))
            i+=1
            continue
    plot(objp, T_s, img_idx, True, "Camera_Poses.png")

def load_camera_parameters():
    fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
    left_camera_intrinsics = fs_l.getNode("camera_intrinsic").mat()
    left_camera_distortion = fs_l.getNode("camera_distortion").mat()

    fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
    right_camera_intrinsics = fs_r.getNode("camera_intrinsic").mat()
    right_camera_distortion = fs_r.getNode("camera_distortion").mat()
    
    return left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion

if __name__ == "__main__":
    import pathlib
    pathlib.Path('../../output/task_6').mkdir(parents=True, exist_ok=True)
    if(len(sys.argv)<2):
        camera = "left"
    else:
        camera = sys.argv[1]

    # M1=[[423.27381306,0,341.34626532],[0,421.27401756,269.28542111],[0,0,1]]
    # dist1=[-0.43394157423038077,0.26707717557547866,-0.00031144347020293427,0.000563893810148836,-0.10970452266148858]

    # M2=[[420.91160482,0,352.16135589],[0,418.72245958,264.50726699],[0,0,1]]
    # dist2=[-0.4145817681176909,0.19961273246897668,-0.0001483209114165653,-0.0013686760437966467,-0.05113584625015141]
    
    objp = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    M1, dist1, M2, dist2 = load_camera_parameters()
    pose_estimate(M1,dist1,M2,dist2,camera)