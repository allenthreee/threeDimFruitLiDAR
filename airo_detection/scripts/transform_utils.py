import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Transform
import tf.transformations as tf_trans
from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import Int32

import sys, os
airo_detection_path = os.path.dirname(__file__)
sys.path.append(airo_detection_path)

class TransformUtils:
    def __init__(self):
        # Camera intrinsics and T_cam_lidar
        self.bridge = CvBridge()
        self.fx = 624.4325
        self.fy = 624.4396
        self.cx = 320.5857
        self.cy = 219.9464
        self.intrinsic_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float32)
        # self.intrinsic_matrix = np.array([[381.36, 0, 320.5], [0.0, 381.36, 240.5], [0, 0, 1]])
        self.dist_coeffs = np.zeros((5, 1), np.float32)
        self.R_cam_lidar_z_neg90 =  np.array([[0,1,0],
                                [-1, 0, 0],
                                [0,0.,1] ])
                                
        # approximate
        self.R_cam_lidar_y_neg104 =  np.array([[-0.2419219,  0.0000000, -0.9702957],
                                        [0.0000000,  1.0000000,  0.0000000],
                                        [0.9702957,  0.0000000, -0.2419219] ])
        self.R_cam_lidar_y_neg101point5 =  np.array([[ -0.1993679,  0.0000000, -0.9799247],
                                            [0.0000000,  1.0000000,  0.0000000],
                                            [0.9799247,  0.0000000, -0.1993679 ]])

        # x:180 y:-11.5
        self.R_imu_lidar = np.array([[ 0.9799247,  0.0000000, -0.1993679],
                                [-0.0000000, -1.0000000, -0.0000000],
                                [-0.1993679,  0.0000000, -0.9799247 ]])

        self.R_cam_lidar = np.dot(self.R_cam_lidar_z_neg90, self.R_cam_lidar_y_neg101point5)
        self.R_lidar_imu = np.linalg.inv(self.R_imu_lidar)
        self.R_cam_IMU = np.dot(self.R_cam_lidar, self.R_lidar_imu)

        self.R_lidar_cam = np.linalg.inv(self.R_cam_lidar)
        # self.R_cam_lidar = np.dot(np.array([[0,1,0], [-1, 0, 0], [0,0.,1]]), np.array([[-0.2419219,  0.0000000, -0.9702957], [0.0000000,  1.0000000,  0.0000000], [0.9702957,  0.0000000, -0.2419219]]))
        self.rotvec_cam_lidar_g, _ = cv2.Rodrigues(self.R_cam_lidar)
        self.transvec_cam_lidar_g = np.array([0.0,0.15,-0.1])

    def uvd_to_cam_coor(self, u, v, depth):
        """
        计算物体在相机坐标系下的坐标
        参数:
        u,v: 物体在图像中的坐标 (u, v)
        depth: 物体的深度
        self.intrinsic_matrix: 相机的内参矩阵
        返回:
        物体在相机坐标系下的坐标 (X, Y, Z)
        """
        # 将图像坐标和深度合并为齐次坐标
        uv = np.array([u,v])
        uvd = np.append(uv,1)
        # print("uvd", uvd)
        # print("depth", depth)
        depth = depth  # NOTE apply extrinsic 1
        u_v_depth = uvd * depth
        # print("u_v_depth", u_v_depth)

        # 计算物体在相机坐标系下的坐标  @为矩阵乘法 dot
        XYZcam = np.dot(np.linalg.inv(self.intrinsic_matrix),u_v_depth)

        return XYZcam

    
    def Tlidar_cam(self, XYZ_cam):
        """
        将点从相机坐标系转换到body坐标系
        参数:
        XYZ_cam: 点在相机坐标系下的坐标 (X, Y, Z)
        rotation_matrix: 从相机坐标系到body坐标系的旋转矩阵
        [0, 0, 1
        -1, 0, 0
         0, -1, 0]
        返回:
        点在body坐标系下的坐标 (X, Y, Z)
        """
        # 将点从相机坐标系转换到body坐标系
        XYZ_lidar = np.dot(self.R_lidar_cam, XYZ_cam)
        XYZ_lidar[0] = XYZ_lidar[0] # - 0.4  # NOTE apply extrinsic 2
        return XYZ_lidar
    
    def Timu_lidar(self, XYZ_lidar):
        """
        将点从lidar坐标系转换到body坐标系
        参数:
        XYZ_lidar: 点在相机坐标系下的坐标 (X, Y, Z)
        rotation_matrix: 从lidar坐标系到body坐标系的旋转矩阵
        返回:
        点在body坐标系下的坐标 (X, Y, Z)
        """
        # 将点从相机坐标系转换到body坐标系
        XYZ_imu = np.dot(self.R_imu_lidar, XYZ_lidar)
        # XYZ_imu [0] = XYZ_lidar[0] # - 0.4  # NOTE apply extrinsic 2
        return XYZ_imu
    
    def odom_to_transformation_matrix(self, odom_msg):
        # 提取位置和方向
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation

        # 创建4x4变换矩阵
        transformation_matrix = np.eye(4)

        # 设置平移部分
        transformation_matrix[0, 3] = position.x
        transformation_matrix[1, 3] = position.y
        transformation_matrix[2, 3] = position.z

        # 设置旋转部分
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf_trans.quaternion_matrix(quaternion)
        transformation_matrix[:3, :3] = rotation_matrix[:3, :3]

        return transformation_matrix

    def Tworld_lidar(self, XYZ_lidar, odom_msg):
        """
        将点从body系转换到world坐标系
        参数:
        point_in_camera_frame: 点在相机坐标系下的坐标 (X, Y, Z)
        transform_matrix: 从 self.odom 中获得

        返回:
        点在world坐标系下的坐标 (X, Y, Z)
        """
        # 从 body coor to world coor
        # 从odometry消息中获取旋转矩阵和平移向量
        # 将点转换为齐次坐标
        transformation_matrix = self.odom_to_transformation_matrix(odom_msg)
        XYZ1_body = np.append(XYZ_lidar, 1)

        # 使用变换矩阵进行坐标变换
        point_in_world_coords_hom = np.dot(transformation_matrix, XYZ1_body)

        # 将结果转换回非齐次坐标
        XYZ_world = point_in_world_coords_hom[:3] / point_in_world_coords_hom[3]
        return XYZ_world
    
    def uvd_to_world(self, u, v, depth, odom_msg):
        """
        计算物体在相机坐标系下的坐标

        参数:
        u, v: 物体在图像中的坐标 (u, v)
        depth: 物体的深度
        intrinsic_matrix: 相机的内参矩阵

        返回:
        水果在世界坐标系下的坐标 (X, Y, Z)
        """
        XYZ_cam = self.uvd_to_cam_coor(u,v,depth)
        XYZ_lidar = self.Tlidar_cam(XYZ_cam)
        XYZ_imu = self.Timu_lidar(XYZ_lidar)
        XYZ_world = self.Tworld_lidar(XYZ_imu, odom_msg)
        # print("XYZ_world is", XYZ_world)

        return XYZ_world
    
    # def world_to_uv(self, XYZ_world_marker):
    #     # 将3D点从世界坐标系转换到相机坐标系
    #     XYZ_world = [0,0,0]
    #     XYZ_world[0] = XYZ_world_marker.pose.position.x
    #     XYZ_world[1] = XYZ_world_marker.pose.position.y
    #     XYZ_world[2] = XYZ_world_marker.pose.position.z
    #     XYZ_body = self.Tlidar_world(XYZ_world)            
    #     # 将3D点从相机坐标系投影到图像平面
    #     XYZ_cam = self.T_cam_body(XYZ_body)
    #     u,v = self.cam_coor_to_uv(XYZ_cam)
    #     if(abs(XYZ_body[0]+XYZ_body[1]+XYZ_body[2])>5):
    #         # 如果到body系距离太远，让u,v 不能接受
    #         u = -1000
    #         v = -1000
    #     return u, v
    

    def Timu_world(self, XYZ_world, odom_msg):
        """
        将点从world坐标系转换到body坐标系
        参数:
        XYZ_world: 点在世界坐标系下的坐标 (N x 3数组)
        odom_msg: odometry消息
        
        返回:
        点在body坐标系下的坐标 (N x 3数组)
        """
        # 获取从body到world的变换矩阵
        T_world_body = self.odom_to_transform_matrix(odom_msg)
        
        # 计算逆变换 - 从world到body
        T_body_world = np.linalg.inv(T_world_body)
        
        # 转换为齐次坐标
        ones_column = np.ones((XYZ_world.shape[0], 1))
        XYZ1_world = np.concatenate((XYZ_world, ones_column), axis=1)
        
        # 应用变换
        XYZ1_body = np.dot(T_body_world, XYZ1_world.T).T
        
        # 转换回非齐次坐标
        XYZ_body = XYZ1_body[:, :3]
        
        return XYZ_body

    def odom_to_transform_matrix(self, odom_msg):
        """
        从odometry消息创建4x4齐次变换矩阵(body到world)
        """
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        
        # 创建变换矩阵
        T = tf_trans.quaternion_matrix([orientation.x, orientation.y, 
                                    orientation.z, orientation.w])
        T[0:3, 3] = [position.x, position.y, position.z]
        
        return T


    # def Timu_world(self, XYZ_world, odom_msg):
    #     """
    #     将点从world坐标系转换到body坐标系
    #     参数:
    #     point_in_world_frame: 点在世界坐标系下的坐标 (X, Y, Z)
    #     transform_matrix: 从 self.odom 中获得

    #     返回:
    #     点在body坐标系下的坐标 (X, Y, Z)
    #     """
    #     # 从 world coor to body coor
    #     # 从odometry消息中获取旋转矩阵和平移向量
    #     # 将点转换为齐次坐标        
    #     position = odom_msg.pose.pose.position
    #     Tworld_lidar = self.odom_to_rotation_matrix(odom_msg)
    #     Timu_world4x4 = np.linalg.inv(Tworld_lidar)
    #     # print(f"Rlidar_world4x4:\n{Timu_world4x4}")
    #     # 设置平移部分
    #     Timu_world4x4[0, 3] = -position.x
    #     Timu_world4x4[1, 3] = -position.y
    #     Timu_world4x4[2, 3] = -position.z

    #     # # 使用 np.concatenate 在矩阵的最后一列添加一列全为1的列
    #     ones_column = np.ones((XYZ_world.shape[0],1))
    #     XYZ1_world = np.concatenate((XYZ_world, ones_column), axis=1)
    #     # print(f"Timu_world4x4:\n{Timu_world4x4}")

    #     XYZ_imu = np.dot(Timu_world4x4, XYZ1_world.T)
    #     XYZ_imu = XYZ_imu.T
    #     # # 将第一行减去number_to_subtract
    #     # XYZ_lidar[0] = XYZ_lidar_no_trans[0] - trans_x
    #     # XYZ_lidar = XYZ_lidar_no_trans[1] - trans_y
    #     # XYZ_lidar = XYZ_lidar_no_trans[2] - trans_z
    #     # XYZ_lidar = XYZ_lidar.T


    #     # # 使用逆变换矩阵进行坐标变换
    #     # point_in_body_coords_hom_T = np.dot(inv_transformation_matrix, XYZ1_world.T)
    #     # print(f"point_in_body_coords_hom_T:\n{point_in_body_coords_hom_T}")
    #     # point_in_body_coords_hom = point_in_body_coords_hom_T.T
    #     # print(f"point_in_body_coords_hom:\n{point_in_body_coords_hom}")
    #     # # 将结果转换回非齐次坐标
    #     # # 使用切片去掉最后一列
    #     XYZ_imu = XYZ_imu[:, :-1]

    #     # print(f"XYZ_imu:\n{XYZ_imu}")

    #     return XYZ_imu
    
    # def odom_to_rotation_matrix(self, odom_msg):
        # 提取位置和方向
        orientation = odom_msg.pose.pose.orientation

        # 创建3x3变换矩阵
        transformation_matrix = np.eye(4)

        # 设置旋转部分
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf_trans.quaternion_matrix(quaternion)

        return rotation_matrix

    def Tbody_world(self, XYZ_world):
        """
        将点从 world 系转换到 body 坐标系
        参数:
        XYZ_world: 点在世界坐标系下的坐标 (X, Y, Z)
        transform_matrix: 从 self.odom 中获得

        返回:
        点在world坐标系下的坐标 (X, Y, Z)
        """
        # 从 body coor to world coor
        # 从odometry消息中获取旋转矩阵和平移向量
        # 将点转换为齐次坐标
        transformation_matrix = self.odom_to_transformation_matrix()
        XYZ1_world = np.append(XYZ_world, 1)
        # 使用变换矩阵进行坐标变换
        point_in_body_coords_hom = np.dot(np.linalg.inv(transformation_matrix), XYZ1_world)

        # 将结果转换回非齐次坐标
        XYZ_body = point_in_body_coords_hom[:3] / point_in_body_coords_hom[3]
        return XYZ_body
    
    def Tcam_body(self, XYZ_body):
        """
        将点从 body 坐标系 转换到 cam 坐标系
        参数:
        XYZ_body: 点在相机坐标系下的坐标 (X, Y, Z)
        rotation_matrix: 从相机坐标系到body坐标系的旋转矩阵
        [0, 0, 1
        -1, 0, 0
         0, -1, 0]
        返回:
        点在body坐标系下的坐标 (X, Y, Z)
        """
        # 将点从相机坐标系转换到body坐标系
        XYZ_cam = np.dot(np.linalg.inv(self.R_body_cam), XYZ_body)
        return XYZ_cam
    
    def DetectCallback(self, rgb_msg, depth_msg, odom_msg):
        try:
            self.cv_rgb_image_raw = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        try:
            self.cv_depth_image_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except CvBridgeError as e:
            print(e)
            return
        # 计算平均深度
        # average_depth = np.nanmean(self.cv_depth_image_raw)
        # print('average depth for this depth image is: ', average_depth)

        self.odom = odom_msg
    
        # Convert BGR to HSV
        hsv = cv2.cvtColor(self.cv_rgb_image_raw, cv2.COLOR_BGR2HSV)

        # Draw the largest 3 contours on the original image
        green_contours = self.findGreenContours(hsv)
        contour_image = self.cv_rgb_image_raw.copy()
        cv2.drawContours(contour_image, green_contours, -1, (255, 255, 0), 2)

        for cnt in sorted(green_contours, key=cv2.contourArea, reverse=True)[:3]:
            
            # Calculate the area of the contour
            area = cv2.contourArea(cnt)
            # Calculate the convex hull for the current contour
            green_hull = cv2.convexHull(cnt)
            # 计算mask部分的像素平均值
            x_mean_green = np.mean(green_hull[:,:,0])
            y_mean_green = np.mean(green_hull[:,:,1])
            # If the area is smaller than the maximum area for a single green area
            # only for plant at the center of iamge(distort)
            orientation_ywy = self.odom.pose.pose.orientation
            quaternion_ywy = [orientation_ywy.x, orientation_ywy.y, orientation_ywy.z, orientation_ywy.w]
            rpy_ywy = tf_trans.euler_from_quaternion(quaternion_ywy)
            rpy_ywy = [i * 57.29 for i in rpy_ywy]

            # print(f'area is: {area}')
            if self.min_area_green < area < self.max_area_green and 270 < x_mean_green < 370 and 10 < y_mean_green < 450:
            # if self.min_area_green < area < self.max_area_green:
                # print(f'area is: {area}')
                
                # Draw the convex hull on the original image
                cv2.drawContours(self.cv_rgb_image_raw, [green_hull], -1, (0, 255, 0), 3)
                
                cv2.drawContours(self.cv_depth_image_raw, [green_hull], -1, (10,10,10), 3)

                # print(f'\nmean x of green_area_depth_mask is: {x_mean_green}')
                # print(f'mean y of green_area_depth_mask is: {y_mean_green}')

                # CALCULATE GREEN DEPTH BY FUNCTION 
                depth_mean_green = self.calcMeanDepthInGreenContour(cnt)
                # depth_mean_green_3 = self.calcMeanDepthInGreenContour_copilot(cnt)
                # print(f"depth_mean_green is {depth_mean_green},======= ")

                if(abs(rpy_ywy[2]) <self.yaw_threshold):
                    depth_mean_green = depth_mean_green + 0.11
                elif(abs(abs(rpy_ywy[2])-180) <self.yaw_threshold):
                    depth_mean_green = depth_mean_green + 0.11
                else:
                    print("yaw is ", rpy_ywy[2])
                    print("NOT good yaw, return")
                    return

                # print("mean depth of green area is: ", depth_mean_green)
                len_fruit_arr = len(self.plant_fruit_database.fruit_arr_.markers)
                print(f"\nthe detected {self.fruit_name} number is: ============================================== {len_fruit_arr}")
                # real_pepper_num = dict_pepper[int(self.bed_ids[0])] + dict_pepper[int(self.bed_ids[1])] + dict_pepper[int(self.bed_ids[2])] + dict_pepper[int(self.bed_ids[3])] + dict_pepper[int(self.bed_ids[4])] + dict_pepper[int(self.bed_ids[5])] 
                # real_tomato_num = dict_tomato[int(self.bed_ids[0])] + dict_tomato[int(self.bed_ids[1])] + dict_tomato[int(self.bed_ids[2])] + dict_tomato[int(self.bed_ids[3])] + dict_tomato[int(self.bed_ids[4])] + dict_tomato[int(self.bed_ids[5])] 
                # real_eggplant_num = dict_eggplant[int(self.bed_ids[0])] + dict_eggplant[int(self.bed_ids[1])] + dict_eggplant[int(self.bed_ids[2])] + dict_eggplant[int(self.bed_ids[3])] + dict_eggplant[int(self.bed_ids[4])] + dict_eggplant[int(self.bed_ids[5])] 
                # print("the real number is: ================================ pepper:%d"%(real_pepper_num)+"   tomato:%d"%(real_tomato_num)+"   eggplant:%d"%(real_eggplant_num))

                XYZ_green = self.uv_to_world(x_mean_green, y_mean_green, depth_mean_green)
                # print("depth_mean_green is : ", depth_mean_green)
                # print("XYZ_green", XYZ_green)
                curr_plant_id = len(self.plant_fruit_database.green_arr_.markers) + 100
                plant_bed_num = XYZ_to_bed_num(XYZ_green)
                if(plant_bed_num in self.bed_ids):
                    if(abs(rpy_ywy[0]) <self.roll_threshold):
                        # print(f"roll is : {rpy_ywy[0]}, okay")
                        pass
                    else:
                        print("roll is =================================: ", rpy_ywy[0])
                        print("==================================NOT good roll, return")
                        return
                    print("in bed_ids, id is: ", plant_bed_num)
                    self.plant_fruit_database.add_plant_marker(curr_plant_id, XYZ_green, abs(rpy_ywy[0]))
                else:
                    print("NOT in bed_ids, id is: ", plant_bed_num)
                    return