#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import message_filters
import statistics
import matplotlib.pyplot as plt
import io
import tf.transformations as tf_trans
import queue
from ultralytics import YOLO
import time
from scipy.spatial import KDTree, ConvexHull
from tf.transformations import quaternion_matrix, translation_matrix, concatenate_matrices

import sys, os
import open3d as o3d
airo_detection_path = os.path.dirname(__file__)
sys.path.append(airo_detection_path)

from threeDim_fruit_database import *
from transform_utils import *
from yolo_detect import *
from tracker_manager import *
import tf

# np.import_array()
_EPS = np.finfo(float).eps  # Define small epsilon value

def quaternion_to_rotation_matrix(quat):
    return tf.transformations.quaternion_matrix(quat)[:3, :3]


def odom_msg_to_rpy(odom_msg):
    orientation = odom_msg.pose.pose.orientation
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rpy_rad = tf_trans.euler_from_quaternion(quaternion)
    # rpy_deg = [i * 57.29 for i in rpy_rad]
    rpy_deg = np.rad2deg(rpy_rad)
    return rpy_deg



class LidarReprojector:
    def __init__(self):
        # Camera intrinsics and T_cam_lidar
        self.queue_size = 2
        self.knn_mode_percent = 80
        # self.knn_dist = 2
        self.bridge = CvBridge()
        # icuas intrinsic
        # self.fx = 624.325
        # self.fy = 624.439
        # self.cx = 300.585
        # self.cy = 239.946
        # w614, realsense d435 rgb intrinsic
        self.fx = 911.325
        self.fy = 911.439
        self.cx = 639.585
        self.cy = 361.946
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float32)
        self.dist_coeffs = np.zeros((5, 1), np.float32)
        self.R_cam_lidar_z_neg90 =  np.array([[0,1,0],
                                [-1, 0, 0],
                                [0,0.,1] ])

        # @TODO 1     move this extrinsic to config.yml            
        # approximate
        # self.R_cam_lidar_y_neg104 =  np.array([[-0.2419219,  0.0000000, -0.9702957],
        #                                 [0.0000000,  1.0000000,  0.0000000],
        #                                 [0.9702957,  0.0000000, -0.2419219] ])
        
        # # approximate
        # self.R_cam_lidar_y_neg102 =  np.array([[ -0.2079117,  0.0000000, -0.9781476],
        #                                     [0.0000000,  1.0000000,  0.0000000],
        #                                     [0.9781476,  0.0000000, -0.2079117 ]])
        
        # self.R_cam_lidar_y_neg101point5 =  np.array([[ -0.1993679,  0.0000000, -0.9799247],
        #                                     [0.0000000,  1.0000000,  0.0000000],
        #                                     [0.9799247,  0.0000000, -0.1993679 ]])

        # x:180 y:-11.5
        # self.R_imu_lidar = np.array([[ 0.9799247,  0.0000000, -0.1993679],
        #                         [-0.0000000, -1.0000000, -0.0000000],
        #                         [-0.1993679,  0.0000000, -0.9799247 ]])

        self.R_imu_lidar = np.array([[1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0 ]])


        # self.R_cam_lidar = np.dot(self.R_cam_lidar_z_neg90, self.R_cam_lidar_y_neg101point5)
        self.R_cam_lidar = self.R_cam_lidar_z_neg90
        self.R_lidar_imu = np.linalg.inv(self.R_imu_lidar)
        self.R_cam_imu = np.dot(self.R_cam_lidar, self.R_lidar_imu)
        # self.R_cam_lidar = np.dot(np.array([[0,1,0], [-1, 0, 0], [0,0.,1]]), np.array([[-0.2419219,  0.0000000, -0.9702957], [0.0000000,  1.0000000,  0.0000000], [0.9702957,  0.0000000, -0.2419219]]))
        self.rotvec_cam_lidar_g, _ = cv2.Rodrigues(self.R_cam_lidar)
        self.transvec_cam_lidar_g = np.array([0.0,0.15,-0.1])
        self.transvec_lidar_imu_g = np.array([0.0,0.15,-0.25])
        self.transvec_cam_imu_g = np.array([-0.07,0.0,-0.15])
        self.lidar_projected_image_pub = rospy.Publisher('/fused_image_ooad', Image, queue_size=10)
        self.yolo_tracking_pub = rospy.Publisher('/yolo_tracking', Image, queue_size=10)
        self.fruit_detections_pub = rospy.Publisher('/fruit_detections', Image, queue_size=10)
        self.yolo_fov_pub = rospy.Publisher('/yolo_fov', MarkerArray, queue_size=10)
        self.points_in_yolo_fov_pub = rospy.Publisher('/points_in_yolo_fov', PointCloud2, queue_size=1)

        self.fruit_database = PlantFruitDatabase()
        self.transform_utils = TransformUtils()
        
        # self.yolo_model = YOLO("/home/allen/Downloads/w614_yolo/airo_ws/src/airo_detection/scripts/icuas_best.pt")
        self.yolo_model = YOLO("/home/allen/Downloads/w614_yolo/airo_ws/src/airo_detection/scripts/strawberry_best.pt")
        self.tracker_manager = TrackManager()

        # ROS node and subscriber
        rospy.init_node('lidar_reprojector', anonymous=True)
        # rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)

        # Subscribe to both topics
        self.sub_image = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        self.sub_odom = message_filters.Subscriber('/Odometry', Odometry)
        
        self.cam_fov_pub = rospy.Publisher('cam_fov_visualization', Marker, queue_size=10)
        self.quad_rviz_pub = rospy.Publisher('quadrotor', MarkerArray, queue_size=10)
        # Synchronize the topics with a slop of 0.1 seconds
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_image, self.sub_odom], 3000, 0.1)
        self.ts.registerCallback(self.callback)

        # Create queues for lidar_msg and odom_msg
        self.lidar_msg_queue = queue.Queue(maxsize=self.queue_size)
        self.odom_msg_queue = queue.Queue(maxsize=self.queue_size)
        self.image_msg_queue = queue.Queue(maxsize=10)
        self.frame_count = 0
        self.min_yellow_size = 10
        self.max_yellow_size = 10
        self.min_red_size = 10
        self.max_red_size = 10

        self.last_odom = Odometry()
        self.last_odom.pose.pose.position = Point(0, 0, 0)
        self.curr_odom = Odometry()
        self.path_length = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0


        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.color_y = (0, 255, 255)  # BGR color for blue
        self.color_r = (0, 0, 255)  # BGR color for blue

        # Initialize with PLY file
        self.ply_path = "/home/allen/Downloads/w614_yolo/PCD/w614_apr_9th_01-shelfOnly_subsample.ply"
        # self.ply_path = "/home/allen/Downloads/w614_yolo/PCD/bag2_scans_treesOnly20percent.ply"
        self.pcd = o3d.io.read_point_cloud(self.ply_path)
        self.prebuild_map_points_arr = self.pointcloud2_to_array()
        # self.preBuildMapKDTree = KDTree(self.prebuild_map_points_arr)

        # Existing initialization code
        self.voxel_size = 0.1  # meters - adjust based on your needs
        self.voxel_grid = None
        self.filtered_points = None
        self.last_pointcloud_update = None
        self.update_voxel_grid(self.prebuild_map_points_arr)

    def callback(self, image_msg, odom_msg):
        rpy_deg = odom_msg_to_rpy(odom_msg)
        self.roll = rpy_deg[0]
        self.pitch = rpy_deg[1]
        self.yaw = rpy_deg[2]
        
        # self.publish_camera_fov_marker(odom_msg)
        self.publish_camera_fov_marker_small(odom_msg)
        self.publish_quad_rviz(odom_msg)

        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # TODO 2: decide if we should project lidar point for visualization, it depends on 计算效率
        # Reproject points
        # uvd_points = self.lidar_points_to_uvd(points_arr)  # YWY noly need uvd
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.yolo_model.track(image, persist=True, verbose=False)
        yolo_tracker_yellows = []
        yolo_tracker_reds = []
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            class_indices = boxes.cls  # Class indices of each detected object
            confidences = boxes.conf  # Confidence scores of each detected object
            track_ids = boxes.id
            xywhs = boxes.xywh
            if xywhs is not None and class_indices is not None and confidences is not None and track_ids is not None:
                for xywh, class_index, confidence, id  in zip(xywhs, class_indices, confidences, track_ids):
                    class_name = result.names[int(class_index)]
                    # print("yolo detect one yellow")
                    darknet_fruit_bbx = BoundingBox()
                    darknet_fruit_bbx.probability = confidence
                    darknet_fruit_bbx.x = float(xywh[0])
                    darknet_fruit_bbx.y = float(xywh[1])
                    darknet_fruit_bbx.radius = float((xywh[2]+xywh[3])/2)
                    darknet_fruit_bbx.id = id
                    darknet_fruit_bbx.Class = class_name
                    point = Point()
                    point.x = float(xywh[0])
                    point.y = float(xywh[1])
                    point.z = float((xywh[2]+xywh[3])/2)
                    darknet_fruit_bbx.lidar_d = 6
                    self.tracker_manager.update(id.item(), darknet_fruit_bbx)
                    if(class_name == "unripe" and confidence > 0.1):
                        yolo_tracker_yellows.append(point)
                    if(class_name == "ripe" and confidence > 0.1):
                        yolo_tracker_reds.append(point)

                    # not that much tracking in w614
                    # for track_id, track in self.tracker_manager.tracks.items():
                    #     if((track_id == id) and (track.tracked_counts >= 1)):
                    #         print(f"track count > 10, ready too add fruit")
                    #         point.z = track.mean_size
                    #         print(f" the bbx size is: ", {point.z})
                    #         if(class_name == "ripe" and confidence > 0.1 and (point.z > 1)):
                    #             yolo_tracker_yellows.append(point)
                    #         if(class_name == "unripe" and confidence > 0.1 and (point.z > 1)):
                    #             yolo_tracker_reds.append(point)

        self.tracker_manager.get_all_tracks()
        yolo_fov_markers = self.publish_yolo_fov_markers(odom_msg, yolo_tracker_yellows, yolo_tracker_reds)
        # self.publish_points_in_yolo_fov_kdtree_cone(yolo_fov_markers, odom_msg)
        self.publish_points_in_yolo_fov_kdtree_cone_voxel(yolo_fov_markers, odom_msg)
        # Visualize the results on the frame
        yolo_tracking_image = results[0].plot()
        yolo_tracking_msg = self.bridge.cv2_to_imgmsg(yolo_tracking_image, "bgr8")
        yolo_tracking_msg.header = odom_msg.header
        self.yolo_tracking_pub.publish(yolo_tracking_msg)

        image_fruit_detections = image.copy()
        image_with_lidar_points = image.copy()
        print(f"yolo_tracker_yellows:", yolo_tracker_yellows)
        self.add_yellow_fruits_from_yolo_detection(yolo_tracker_yellows, "yellow", image_with_lidar_points, odom_msg, image_fruit_detections)
        self.add_red_fruits_from_yolo_detection(yolo_tracker_reds, "red", image_with_lidar_points, odom_msg, image_fruit_detections)
        print("add red and yellow fruit\n")
        image_fruit_detections = image.copy()
        if(len(self.fruit_database.red_fruit_arr_.markers)>0 or len(self.fruit_database.yellow_fruit_arr_.markers)>0):
            uvd_fruits = self.fruit_markers_to_uvd(odom_msg)
            image_with_lidar_points = self.colorize_fruits_uvd(uvd_fruits, image_with_lidar_points)

        image_stamp_str = "image_time" + str("{:.2f}".format(image_msg.header.stamp.to_sec()))
        odom_stamp_str = "odom_time" + str("{:.2f}".format(odom_msg.header.stamp.to_sec()))
        cv2.putText(image_with_lidar_points, image_stamp_str, (5,120), self.font, self.font_scale, self.color_y, thickness=2)
        cv2.putText(image_with_lidar_points, odom_stamp_str, (5,200), self.font, self.font_scale, self.color_y, thickness=2)
        
        imageMinusOdom = image_msg.header.stamp.to_sec() - odom_msg.header.stamp.to_sec()
        imageMinusOdom_str = "imageMinusOdom_str" + str("{:.4f}".format(imageMinusOdom)) 
        cv2.putText(image_with_lidar_points, imageMinusOdom_str, (5,280), self.font, self.font_scale, self.color_y, thickness=2)

        

        colored_image_msg = self.bridge.cv2_to_imgmsg(image_with_lidar_points, "bgr8")
        colored_image_msg.header = odom_msg.header
        self.lidar_projected_image_pub.publish(colored_image_msg)

        self.calculate_path_length_3d()
        self.last_odom = self.curr_odom
        self.curr_odom = odom_msg
        # image.clear()
        # uvd_points.clear()

    def mean_point_in_one_bbx(self, odom_msg, yolo_tracker_bbx_one):
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "dirty_code"
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        # bug, no orientation
        marker.pose = odom_msg.pose.pose
        
        # Set color and style
        marker.scale.x = 0.03

        marker.color.a = 0.7        
        # Camera intrinsic parameters
        # image_width = 640
        # image_height = 480
        # cam_fov_width = 5.0  # meters at 5m distance
        # cam_fov_height = 5.0  # meters at 5m distance

        image_width = 1280
        image_height = 720
        cam_fov_width = 3.0  # meters at 5m distance
        cam_fov_height = 2.0  # meters at 5m distance
        
        # Calculate pixel-to-meter conversion factors
        px_to_meter_width = cam_fov_width / image_width
        px_to_meter_height = cam_fov_height / image_height
                
        # Convert from pixel offset to angular offset
        # CHANGED: Removed the negative sign for x_offset_px to fix left/right inversion
        x_offset_px = (image_width/2) - yolo_tracker_bbx_one.x  # Corrected left/right
        y_offset_px = (image_height/2) - yolo_tracker_bbx_one.y  # Flip Y axis
        
        # Calculate angular width/height at detection distance
        length = 3.5
        detection_width = yolo_tracker_bbx_one.z * 2
        detection_height = yolo_tracker_bbx_one.z * 2
        
        # Convert to meters using camera FOV scaling
        fov_width = detection_width * px_to_meter_width
        fov_height = detection_height * px_to_meter_height
        
        # Calculate direction vector components
        dir_x = length
        dir_y = x_offset_px * (cam_fov_width/image_width) * (length/5.0)
        dir_z = y_offset_px * (cam_fov_height/image_height) * (length/5.0)
        
        # Base pyramid points
        p0 = Point(0, 0, 0)
        
        # Far plane points (at 5m)
        half_w = fov_width/2
        half_h = fov_height/2
        # Corrected left/right points:
        p1 = Point(dir_x, dir_y - half_w, dir_z + half_h)  # Top-left (was Top-right)
        p2 = Point(dir_x, dir_y + half_w, dir_z + half_h)  # Top-right (was Top-left)
        p3 = Point(dir_x, dir_y + half_w, dir_z - half_h)  # Bottom-right (was Bottom-left)
        p4 = Point(dir_x, dir_y - half_w, dir_z - half_h)  # Bottom-left (was Bottom-right)
        
        # Create pyramid lines
        marker.points = [
            p0, p1, p0, p2, p0, p3, p0, p4,  # Pyramid edges
            p1, p2, p2, p3, p3, p4, p4, p1    # Far plane rectangle
        ]
        
        for i in range(0, len(marker.points)):  # Use () and dynamic length
            original_point = marker.points[i]  # Store original before modification
            transformed_point = self.transform_marker_points_to_world_frame(marker, original_point)
            
            # Debug print before assignment
            # print(f"Original point {i}: ({original_point.x:.3f}, {original_point.y:.3f}, {original_point.z:.3f})")
            # print(f"Transformed point {i}: ({transformed_point.x:.3f}, {transformed_point.y:.3f}, {transformed_point.z:.3f})")
            
            marker.points[i] = transformed_point  # Assign after verification            # Get pyramid edges (in world frame)

        # Compute bounding box center and size
        p0_world = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z])  # Camera origin
        bb_points = np.array([
            [marker.points[i].x, marker.points[i].y, marker.points[i].z] 
            for i in [1, 3, 5, 7]  # Use the 4 corners of the FoV
        ])
        bb_center = np.mean(bb_points, axis=0)
        print(f"bb_center is: {bb_center.tolist()}")  # Explicitly convert to list
        bb_size = np.linalg.norm(np.max(bb_points, axis=0) - np.min(bb_points, axis=0))  # sqrt(size_x² + size_y² + size_z²)
        print(f"bb_size is: ", {bb_size})
        
        # Cone parameters (k=1.0 for radius scaling)
        k = 1
        cone_axis = bb_center - p0_world  # Cone direction (from camera to BB center)
        cone_axis /= np.linalg.norm(cone_axis)  # Normalize
        cone_radius = k * bb_size  # Use max horizontal BB size for radius
        
        # # Query KD-Tree for points in a sphere around the cone (for initial filtering)
        query_radius = np.linalg.norm(bb_center - p0_world) + cone_radius
        # candidate_indices = self.preBuildMapKDTree.query_ball_point(p0_world, query_radius)
        print("we are searching point for fruit depth in the bbx cone")
                                       
        # Get all candidate points at once
        # candidate_points = self.prebuild_map_points_arr[candidate_indices]

        # Vectorized calculations
        # point_vecs = self.prebuild_map_points_arr - p0_world
        point_vecs = self.filtered_points - p0_world
        dot_products = np.dot(point_vecs, cone_axis)

        # Filter points in front of camera
        front_mask = dot_products > 0
        front_points = self.filtered_points[front_mask]
        front_dots = dot_products[front_mask]
        # front_vecs = point_vecs[front_mask]

        # Projections and perpendicular distances
        proj_points = p0_world + front_dots[:, None] * cone_axis
        perp_dists = np.linalg.norm(front_points - proj_points, axis=1)

        # Allowed radii
        allowed_radii = (front_dots / np.linalg.norm(bb_center - p0_world)) * cone_radius

        # Final mask
        in_cone_mask = perp_dists <= allowed_radii
        in_fov_points = front_points[in_cone_mask]
        # When in_fov_points is a list of numpy arrays (shape (3,))
        if len(in_fov_points) > 0:
            mean_point = np.mean(np.vstack(in_fov_points), axis=0)
            print(f"mean_point is:", mean_point)
        else:
            mean_point = None  # Or handle empty case appropriately
            print(f"mean_point is None")
            # print(f"in_fov_points:\n", in_fov_points)
        return mean_point


    
    def publish_yolo_fov_markers(self, odom_msg, yolo_tracker_yellows, yolo_tracker_reds):
        marker_array = MarkerArray()
        marker_id = 0
        duration = rospy.Duration(0.1)

        # Camera intrinsic parameters
        # image_width = 640
        # image_height = 480
        # cam_fov_width = 5.0  # meters at 5m distance
        # cam_fov_height = 5.0  # meters at 5m distance

        # Camera intrinsic parameters
        image_width = 1280
        image_height = 720
        cam_fov_width = 3.0  # meters at 5m distance
        cam_fov_height = 2.0  # meters at 5m distance
        
        # Calculate pixel-to-meter conversion factors
        px_to_meter_width = cam_fov_width / image_width
        px_to_meter_height = cam_fov_height / image_height
        
        for color_group, color in [(yolo_tracker_yellows, (1.0, 1.0, 0.0)), 
                                (yolo_tracker_reds, (1.0, 0.0, 0.0))]:
            
            for point in color_group:
                marker = Marker()
                marker.header.frame_id = "camera_init"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "yolo_fov"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.lifetime = duration
                # bug, no orientation
                marker.pose = odom_msg.pose.pose
                
                # Set color and style
                marker.scale.x = 0.03
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.7
                
                # Convert from pixel offset to angular offset
                # CHANGED: Removed the negative sign for x_offset_px to fix left/right inversion
                x_offset_px = (image_width/2) - point.x  # Corrected left/right
                y_offset_px = (image_height/2) - point.y  # Flip Y axis
                
                # Calculate angular width/height at detection distance
                length = 3.5
                detection_width = point.z * 2
                detection_height = point.z * 2
                
                # Convert to meters using camera FOV scaling
                fov_width = detection_width * px_to_meter_width
                fov_height = detection_height * px_to_meter_height
                
                # Calculate direction vector components
                dir_x = length
                dir_y = x_offset_px * (cam_fov_width/image_width) * (length/5.0)
                dir_z = y_offset_px * (cam_fov_height/image_height) * (length/5.0)
                
                # Base pyramid points
                p0 = Point(0, 0, 0)
                
                # Far plane points (at 5m)
                half_w = fov_width/2
                half_h = fov_height/2
                # Corrected left/right points:
                p1 = Point(dir_x, dir_y - half_w, dir_z + half_h)  # Top-left (was Top-right)
                p2 = Point(dir_x, dir_y + half_w, dir_z + half_h)  # Top-right (was Top-left)
                p3 = Point(dir_x, dir_y + half_w, dir_z - half_h)  # Bottom-right (was Bottom-left)
                p4 = Point(dir_x, dir_y - half_w, dir_z - half_h)  # Bottom-left (was Bottom-right)
                
                # Create pyramid lines
                marker.points = [
                    p0, p1, p0, p2, p0, p3, p0, p4,  # Pyramid edges
                    p1, p2, p2, p3, p3, p4, p4, p1    # Far plane rectangle
                ]
                
                marker_array.markers.append(marker)
        
        self.yolo_fov_pub.publish(marker_array)
        # print("yolo_fov_published")

        return marker_array


    def publish_camera_fov_marker(self, odom_msg):
        marker = Marker()
        marker.header.frame_id = "camera_init"  # Change to your camera frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_fov"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Set marker pose (same as camera)
        marker.pose = odom_msg.pose.pose

        # Set scale and color
        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.20
        marker.color.b = 1.0
        marker.color.g = 0.99
        marker.color.a = 0.7

        # Define FoV dimensions
        fov_height = 3.0  # Change to your FoV height
        fov_width = 4.0  # Change to your FoV width

        # Define points of the pyramid (assuming camera at origin)
        p1 = Point(0, 0, 0)  # Camera position
        p10 = Point(5, 0, 0)
        p2 = Point(5, fov_width / 2, fov_height / 2)
        p3 = Point(5, -fov_width / 2, fov_height / 2)
        p4 = Point(5, -fov_width / 2, -fov_height / 2)
        p5 = Point(5, fov_width / 2, -fov_height / 2)



        p6 = Point(5, 0, -fov_height / 2)
        p7 = Point(5, 0, fov_height / 2)
        p8 = Point(5, fov_width / 2, 0)
        p9 = Point(5, -fov_width / 2, 0)

        p11 = Point(3, fov_width / 2, fov_height / 2)
        p12 = Point(3, -fov_width / 2, fov_height / 2)
        p13 = Point(3, -fov_width / 2, -fov_height / 2)
        p14 = Point(3, fov_width / 2, -fov_height / 2)

        p15 = Point(7, fov_width / 2, fov_height / 2)
        p16 = Point(7, -fov_width / 2, fov_height / 2)
        p17 = Point(7, -fov_width / 2, -fov_height / 2)
        p18 = Point(7, fov_width / 2, -fov_height / 2)
        # Add lines to the marker
        marker.points = [p1, p10, p1, p2, p1, p3, p1, p4, p1, p5, p2, p3, p3, p4, p4, p5, p5, p2, p6, p7, p8, p9, p11, p12, p12, p13, p13, p14, p14, p11, p15,p16,p16,p17,p17,p18,p18,p15]

        # Publish the marker
        self.cam_fov_pub.publish(marker)

    def publish_camera_fov_marker_small(self, odom_msg):
        # 1280*720
        marker = Marker()
        marker.header.frame_id = "camera_init"  # Change to your camera frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_fov"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Set marker pose (same as camera)
        marker.pose = odom_msg.pose.pose

        # Set scale and color
        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.20
        marker.color.b = 1.0
        marker.color.g = 0.99
        marker.color.a = 0.7

        # Define FoV dimensions (reduced to 1/10th of original)
        fov_height = 0.2  # Changed from 3.0
        fov_width = 0.3  # Changed from 4.0

        # Define points of the pyramid (assuming camera at origin)
        p1 = Point(0, 0, 0)  # Camera position
        p10 = Point(0.5, 0, 0)  # Changed from 5.0
        p2 = Point(0.5, fov_width / 2, fov_height / 2)
        p3 = Point(0.5, -fov_width / 2, fov_height / 2)
        p4 = Point(0.5, -fov_width / 2, -fov_height / 2)
        p5 = Point(0.5, fov_width / 2, -fov_height / 2)

        p6 = Point(0.5, 0, -fov_height / 2)
        p7 = Point(0.5, 0, fov_height / 2)
        p8 = Point(0.5, fov_width / 2, 0)
        p9 = Point(0.5, -fov_width / 2, 0)

        p11 = Point(0.3, fov_width / 2, fov_height / 2)  # Changed from 3.0
        p12 = Point(0.3, -fov_width / 2, fov_height / 2)
        p13 = Point(0.3, -fov_width / 2, -fov_height / 2)
        p14 = Point(0.3, fov_width / 2, -fov_height / 2)

        p15 = Point(0.7, fov_width / 2, fov_height / 2)  # Changed from 7.0
        p16 = Point(0.7, -fov_width / 2, fov_height / 2)
        p17 = Point(0.7, -fov_width / 2, -fov_height / 2)
        p18 = Point(0.7, fov_width / 2, -fov_height / 2)
        
        # Add lines to the marker
        marker.points = [p1, p10, p1, p2, p1, p3, p1, p4, p1, p5, p2, p3, p3, p4, p4, p5, p5, p2, p6, p7, p8, p9, p11, p12, p12, p13, p13, p14, p14, p11, p15,p16,p16,p17,p17,p18,p18,p15]

        # Publish the marker
        self.cam_fov_pub.publish(marker)


        # rospy.init_node('camera_fov_visualizer')
        # marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    
    def publish_quad_rviz(self, odom_msg):
        # 创建 MarkerArray
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "camera_init"  # Change to your camera frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "quad"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Set marker pose (same as camera)
        marker.pose = odom_msg.pose.pose

        # Set scale and color
        marker.scale.x = 0.021  # Line width
        marker.color.r = 0.20
        marker.color.b = 1.0
        marker.color.g = 0.199
        marker.color.a = 1.0

        # Define FoV dimensions

        # Define points of the pyramid (assuming camera at origin)
        p1 = Point(0, 0, 0)  # Camera position
        p2 = Point(1/10, 1/10, 0)
        p3 = Point(1/10, -1/10, 0)
        p4 = Point(-1/10, 1/10, 0)
        p5 = Point(-1/10, -1/10, 0)
        # Add lines to the marker
        marker.points = [p1, p2, p1, p3, p1, p4, p1, p5]
        marker_array.markers.append(marker)

        # 创建四个圆盘
        quad_center = odom_msg.pose.pose.position
        quad_orientation = odom_msg.pose.pose.orientation
        for i, pos in enumerate([(-1/10, -1/10, 0), (1/10, 1/10, 0), (-1/10, 1/10, 0), (1/10, -1/10, 0)]):
            marker_disc = Marker()
            marker_disc.header.frame_id = "camera_init"
            marker_disc.ns = "quadrotor"
            marker_disc.id = i + 1
            marker_disc.type = Marker.CYLINDER
            marker_disc.action = Marker.ADD
            marker_disc.pose.position = Point(quad_center.x+pos[0], quad_center.y+pos[1], quad_center.z+pos[2])
            marker_disc.pose.orientation = Quaternion(quad_orientation.x, quad_orientation.y, quad_orientation.z, quad_orientation.w)
            marker_disc.scale.x = 0.9/10  # 直径
            marker_disc.scale.y = 0.9/10  # 直径
            marker_disc.scale.z = 0.31/10  # 高度
            marker_disc.color.a = 1.0  # alpha
            marker_disc.color.r = 1.0  # red
            marker_array.markers.append(marker_disc)


        # 发布 MarkerArray
        self.quad_rviz_pub.publish(marker_array)

    
    def pointcloud2_to_array(self):
        """Convert Open3D point cloud to numpy array (Nx3) with float64 dtype"""
        points = np.asarray(self.pcd.points, dtype=np.float64)  # Ensure float64
        return points  # Shape: (N, 3)

    def point_in_hull(self, point, hull_points, hull):
        """Check if a point is inside a convex hull"""
        eq = np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1]
        return np.all(eq <= 1e-10)
    
    def transform_marker_points_to_world_frame(self, marker, point):
        """
        Corrected function to transform a point from marker-local frame to world frame.
        
        Args:
            marker: The marker containing pose information
            point: geometry_msgs/Point to transform
            
        Returns:
            geometry_msgs/Point in world frame
        """
        # Extract position and orientation
        position = marker.pose.position
        orientation = marker.pose.orientation

        # Create rotation matrix from quaternion
        rot_matrix = quaternion_matrix([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])

        # Create translation matrix (using different variable name)
        trans_matrix = translation_matrix([
            position.x,
            position.y,
            position.z
        ])

        # Combine transformations (translation first, then rotation)
        transform_matrix = concatenate_matrices(trans_matrix, rot_matrix)

        # Convert point to homogeneous coordinates
        point_homog = np.array([point.x, point.y, point.z, 1.0])
        
        # Apply transformation
        transformed_point = np.dot(transform_matrix, point_homog)

        # Return as Point message
        world_point = Point(
            x=transformed_point[0],
            y=transformed_point[1],
            z=transformed_point[2]
        )

        return world_point

    def update_voxel_grid(self, point_cloud):
        """Update the voxel grid when the point cloud changes"""
        if len(point_cloud) == 0:
            return
            
        # Calculate voxel indices for all points
        voxel_indices = (point_cloud / self.voxel_size).astype(np.int32)
        
        # Use a dictionary to keep one point per voxel
        voxel_dict = {}
        for idx, voxel in enumerate(voxel_indices):
            voxel_tuple = tuple(voxel)
            if voxel_tuple not in voxel_dict:
                voxel_dict[voxel_tuple] = idx
        
        # Store the filtered points
        filtered_indices = list(voxel_dict.values())
        self.filtered_points = point_cloud[filtered_indices]
        self.last_pointcloud_update = rospy.Time.now()

    def publish_points_in_yolo_fov_kdtree_cone(self, yolo_fov_markers, odom_msg):
        in_fov_points = np.empty((0, 3))  # Initialize as empty array with 3 columns
        
        # Process each FoV marker (already in world frame)
        for marker in yolo_fov_markers.markers:
            if len(marker.points) < 8:
                continue
            for i in range(0, len(marker.points)):  # Use () and dynamic length
                original_point = marker.points[i]  # Store original before modification
                transformed_point = self.transform_marker_points_to_world_frame(marker, original_point)
                
                # Debug print before assignment
                # print(f"Original point {i}: ({original_point.x:.3f}, {original_point.y:.3f}, {original_point.z:.3f})")
                # print(f"Transformed point {i}: ({transformed_point.x:.3f}, {transformed_point.y:.3f}, {transformed_point.z:.3f})")
                
                marker.points[i] = transformed_point  # Assign after verification            # Get pyramid edges (in world frame)
            p0_marker = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z])  # Camera origin
            p0_world = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z])  # Camera origin
            print(f"p0_marker is: {p0_marker.tolist()}")  # Explicitly convert to list

            # Compute bounding box center and size
            bb_points = np.array([
                [marker.points[i].x, marker.points[i].y, marker.points[i].z] 
                for i in [1, 3, 5, 7]  # Use the 4 corners of the FoV
            ])
            bb_center = np.mean(bb_points, axis=0)
            print(f"bb_center is: {bb_center.tolist()}")  # Explicitly convert to list
            bb_size = np.linalg.norm(np.max(bb_points, axis=0) - np.min(bb_points, axis=0))  # sqrt(size_x² + size_y² + size_z²)
            print(f"bb_size is: ", {bb_size})
            
            # Cone parameters (k=1.0 for radius scaling)
            k = 1.0
            cone_axis = bb_center - p0_world  # Cone direction (from camera to BB center)
            cone_axis /= np.linalg.norm(cone_axis)  # Normalize
            cone_radius = k * bb_size  # Use max horizontal BB size for radius
            cone_length = np.linalg.norm(cone_axis)

            # Vectorized calculations
            # point_vecs = candidate_points - p0_world
            point_vecs = self.prebuild_map_points_arr - p0_world
            dot_products = np.dot(point_vecs, cone_axis)

            # Filter points in front of camera
            front_mask = dot_products > 0
            front_points = self.prebuild_map_points_arr[front_mask]
            front_dots = dot_products[front_mask]
            # front_vecs = point_vecs[front_mask]

            # Projections and perpendicular distances
            proj_points = p0_world + front_dots[:, None] * cone_axis
            perp_dists = np.linalg.norm(front_points - proj_points, axis=1)

            # Allowed radii
            allowed_radii = (front_dots / np.linalg.norm(bb_center - p0_world)) * cone_radius

            # Final mask
            in_cone_mask = perp_dists <= allowed_radii
            in_fov_point_one_bbx = front_points[in_cone_mask]
            # print("in_fov_point_one_bbx type:", type(in_fov_point_one_bbx))
            # print("in_fov_point_one_bbx contents:", in_fov_point_one_bbx)
            # print("in_fov_point_one_bbx dtype:", in_fov_point_one_bbx.dtype)
            # print("in_fov_point_one_bbx shape:", in_fov_point_one_bbx.shape)
            # in_fov_points.append(in_fov_point_one_bbx)
            in_fov_points = np.vstack((in_fov_points, in_fov_point_one_bbx))  # Stack vertically

            
        # Publish results (in world frame)
        if in_fov_points.any():
            header = odom_msg.header
            header.stamp = odom_msg.header.stamp
            header.frame_id = "camera_init"
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            
            structured_array = np.zeros(len(in_fov_points), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
            for i, p in enumerate(in_fov_points):
                structured_array[i] = (p[0], p[1], p[2])
            
            out_msg = PointCloud2()
            out_msg.header = header
            out_msg.height = 1
            out_msg.width = len(in_fov_points)
            out_msg.fields = fields
            out_msg.is_bigendian = False
            out_msg.point_step = 12
            out_msg.row_step = 12 * len(in_fov_points)
            out_msg.is_dense = True
            out_msg.data = structured_array.tobytes()
            
            self.points_in_yolo_fov_pub.publish(out_msg)
        return in_fov_points

    def publish_points_in_yolo_fov_kdtree_cone_voxel(self, yolo_fov_markers, odom_msg):
        in_fov_points = np.empty((0, 3))  # Initialize as empty array with 3 columns
        
        # Check if we have filtered points available
        if self.filtered_points is None or len(self.filtered_points) == 0:
            return in_fov_points
            
        # Process each FoV marker (already in world frame)
        for marker in yolo_fov_markers.markers:
            if len(marker.points) < 8:
                continue
                
            # Transform marker points to world frame
            for i in range(0, len(marker.points)):
                original_point = marker.points[i]
                transformed_point = self.transform_marker_points_to_world_frame(marker, original_point)
                marker.points[i] = transformed_point
                
            # Get camera origin
            p0_world = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z])
            
            # Compute bounding box center and size
            bb_points = np.array([
                [marker.points[i].x, marker.points[i].y, marker.points[i].z] 
                for i in [1, 3, 5, 7]  # Use the 4 corners of the FoV
            ])
            bb_center = np.mean(bb_points, axis=0)
            bb_size = np.linalg.norm(np.max(bb_points, axis=0) - np.min(bb_points, axis=0))
            
            # Cone parameters
            k = 1.0
            cone_axis = bb_center - p0_world
            cone_axis /= np.linalg.norm(cone_axis)
            cone_radius = k * bb_size
            cone_length = np.linalg.norm(cone_axis)

            # Vectorized calculations using pre-filtered points
            point_vecs = self.filtered_points - p0_world
            dot_products = np.dot(point_vecs, cone_axis)

            # Filter points in front of camera
            front_mask = dot_products > 0
            front_points = self.filtered_points[front_mask]
            front_dots = dot_products[front_mask]

            # Projections and perpendicular distances
            proj_points = p0_world + front_dots[:, None] * cone_axis
            perp_dists = np.linalg.norm(front_points - proj_points, axis=1)

            # Allowed radii
            allowed_radii = (front_dots / np.linalg.norm(bb_center - p0_world)) * cone_radius

            # Final mask
            in_cone_mask = perp_dists <= allowed_radii
            in_fov_point_one_bbx = front_points[in_cone_mask]
            in_fov_points = np.vstack((in_fov_points, in_fov_point_one_bbx))
        
        # Publish results
        if in_fov_points.any():
            header = odom_msg.header
            header.stamp = odom_msg.header.stamp
            header.frame_id = "camera_init"
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            
            structured_array = np.zeros(len(in_fov_points), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
            for i, p in enumerate(in_fov_points):
                structured_array[i] = (p[0], p[1], p[2])
            
            out_msg = PointCloud2()
            out_msg.header = header
            out_msg.height = 1
            out_msg.width = len(in_fov_points)
            out_msg.fields = fields
            out_msg.is_bigendian = False
            out_msg.point_step = 12
            out_msg.row_step = 12 * len(in_fov_points)
            out_msg.is_dense = True
            out_msg.data = structured_array.tobytes()
            
            self.points_in_yolo_fov_pub.publish(out_msg)
        
        return in_fov_points

    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion"""
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3], q[1, 2]-q[3, 0], q[1, 3]+q[2, 0], 0.0],
            [q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3], q[2, 3]-q[1, 0], 0.0],
            [q[1, 3]-q[2, 0], q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0]])

    def plane_from_points(self, p1, p2, p3):
        """Calculate plane equation from 3 points (ax + by + cz + d = 0)"""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize
        d = -np.dot(normal, p1)
        return np.append(normal, d)

    def calculate_path_length_3d(self):
        dx = self.curr_odom.pose.pose.position.x - self.last_odom.pose.pose.position.x
        dy = self.curr_odom.pose.pose.position.y - self.last_odom.pose.pose.position.y
        dz = self.curr_odom.pose.pose.position.z - self.last_odom.pose.pose.position.z
        self.path_length += (dx**2 + dy**2 + dz**2)**0.5
        return self.path_length

    def fruit_reprojection(self, fruit_XYZ_world):
        fruit_XYZ_lidar = self.transform_utils.Tlidar_world(fruit_XYZ_world)
        return fruit_XYZ_lidar

    def add_yellow_fruits_from_yolo_detection(self, yolo_fruit_tracks, color, image_with_lidar_points, odom_msg, image_fruit_detections):
        for fruit_point in yolo_fruit_tracks:
            image_with_lidar_points = self.draw_bbx(image_with_lidar_points, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z))

            print("\nafter mean_point_in_one_bbx")
            XYZ_yellow = self.mean_point_in_one_bbx(odom_msg, fruit_point)
            print("after mean_point_in_one_bbx\n")
            if(XYZ_yellow is None):
                continue
            # XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color_g = (0, 255, 0)  # BGR color for blue
            color_r = (0, 0, 0)  # BGR color for blue
            # Position for the text
            position = (int(fruit_point.x), int(fruit_point.y))  # You can adjust this according to your needs
            # curr_red_id = len(self.fruit_database.red_fruit_arr_.markers)+100
            curr_yellow_id = len(self.fruit_database.yellow_fruit_arr_.markers)
            print(f"we have============================= {curr_yellow_id} ============================yellow fruits now")
            yellow_id = self.fruit_database.add_yellow_fruit_marker(color, curr_yellow_id, XYZ_yellow, abs(8), fruit_point.z)
            image_fruit_detections = self.draw_yellow_bbx(image_fruit_detections, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z), yellow_id)
            
            fruit_detections_image_msg = self.bridge.cv2_to_imgmsg(image_fruit_detections, "bgr8")
            fruit_detections_image_msg.header = odom_msg.header
            self.fruit_detections_pub.publish(fruit_detections_image_msg)

            self.fruit_database.publish_markers()


    def add_red_fruits_from_yolo_detection(self, yolo_fruit_tracks, color, image_with_lidar_points, odom_msg, image_fruit_detections):
        for fruit_point in yolo_fruit_tracks:
            # print(f"rpy_deg is : {rpy_deg}")
            # print(f"red fuit is at ({int(fruit_point.x)},{int(fruit_point.y)}, {int(fruit_point.z)})")
            image_with_lidar_points = self.draw_bbx(image_with_lidar_points, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z))

            XYZ_red = self.mean_point_in_one_bbx(odom_msg, fruit_point)
            if(XYZ_red is None):
                continue
            curr_red_id = len(self.fruit_database.red_fruit_arr_.markers)+100
            # print(f"we have============================= {curr_red_id} ============================red fruits now")
            red_id = self.fruit_database.add_red_fruit_marker(color, curr_red_id, XYZ_red, abs(8), fruit_point.z)
            image_fruit_detections = self.draw_red_bbx(image_fruit_detections, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z), curr_red_id)
            
            fruit_detections_image_msg = self.bridge.cv2_to_imgmsg(image_fruit_detections, "bgr8")
            fruit_detections_image_msg.header = odom_msg.header
            self.fruit_detections_pub.publish(fruit_detections_image_msg)

            self.fruit_database.publish_markers()


    def colorize_uvd_visualization(self, uvd_points, image):
        # Convert xy from homogeneous points to image points, keep z as its real dist
        mask = (uvd_points[:, 0] >= 40) & (uvd_points[:, 0] <= 600)  #changed but not tested

        # Apply the mask to the uvd_points array to get the filtered array
        filtered_uvd_points = uvd_points[mask]
        filtered_z_values = filtered_uvd_points[:, 2]
        image_mean_depth = np.mean(filtered_z_values)
        # # Get the first column of uvd_points
        # first_column = uvd_points[:, 0]

        right_u_mask = (uvd_points[:, 0] >= 580) & (uvd_points[:, 0] <= 640)  #changed but not tested
        right_filtered_uvd_points = uvd_points[right_u_mask]
        right_filtered_z_values = right_filtered_uvd_points[:, 2]
        right_image_mean_depth = np.mean(right_filtered_z_values)
        # print(f"the fov_mean_depth is:{image_mean_depth}")
        # Clip the z_values to the range [3, 8]
        z_values = uvd_points[:,2]
        z_values = np.clip(z_values, 3, 9)
        # Normalize the z_values to the range [0, 1]
        z_values = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        # Define the colors for close and far points
        close_color = np.array([0, 0, 255])  # Red in BGR
        far_color = np.array([255, 125, 0])  # Blue in BGR

        # Calculate the colors for the points
        colors = (1 - z_values[:, np.newaxis]) * close_color + z_values[:, np.newaxis] * far_color
        # Concatenate the image points and colors
        colored_image_points = np.hstack((uvd_points, colors))

        for x, y, z, b,g,r in np.int32(colored_image_points):
            # print(f"b,g,r:{b},{g},{r}")
            cv2.circle(image, (x, y), 1, color=(int(b), int(g), int(r)), thickness=2)
        
        # 画出分割线
        width_step = 100
        height_step = 100
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        path_length_str = "path_length: " + str(self.path_length)
        cv2.putText(image, path_length_str,(300, 50), font, font_scale, (0,0,0), thickness)
        str_right_image_mean_depth = "540-640 depth: " + str(right_image_mean_depth)
        cv2.putText(image, str_right_image_mean_depth,(300, 80), font, font_scale, (0,0,0), thickness)
        str_roll = "roll: " + str(self.roll)
        cv2.putText(image, str_roll,(300, 120), font, font_scale, (0,0,0), thickness)
        str_roll = "picth: " + str(self.pitch)
        cv2.putText(image, str_roll,(300, 150), font, font_scale, (0,0,0), thickness)
        str_roll = "yaw: " + str(self.yaw)
        cv2.putText(image, str_roll,(300, 180), font, font_scale, (0,0,0), thickness)
        

        # 中轴十字线
        cv2.line(image, (320, 0), (320, image.shape[0]), (255, 255, 50), 2)
        cv2.line(image, (0, 240), (640, 240), (255, 255, 50), 2)
        for i in range(1, 7):
            cv2.line(image, (i * width_step, 0), (i * width_step, image.shape[0]), (255, 255, 255), 1)
            cv2.line(image, (0, i * height_step), (image.shape[1], i * height_step), (255, 255, 255), 1)

        return image, image_mean_depth, right_image_mean_depth
    
    def fruit_markers_to_uvd(self, odom_msg):
        worldXYZ_fruit = np.empty((0,3))
        for i in range(0,len(self.fruit_database.yellow_fruit_arr_.markers)):
            marker = self.fruit_database.yellow_fruit_arr_.markers[i]
            marker_position = marker.pose.position
            new_fruit = np.array([float(marker_position.x), marker_position.y, marker_position.z])
            worldXYZ_fruit = np.append(worldXYZ_fruit, [new_fruit], axis=0)
        
        for i in range(0,len(self.fruit_database.red_fruit_arr_.markers)):
            marker = self.fruit_database.red_fruit_arr_.markers[i]
            marker_position = marker.pose.position
            new_fruit = np.array([float(marker_position.x), marker_position.y, marker_position.z])
            worldXYZ_fruit = np.append(worldXYZ_fruit, [new_fruit], axis=0)
        # print(f"worldXYZ_fruit is:\n{worldXYZ_fruit}")
        # print(f"odom is: \n", odom_msg.pose.pose)
        imuXYZ_fruits = self.transform_utils.Timu_world(worldXYZ_fruit, odom_msg)
        rotated_lidarXYZ_fruits = np.dot(self.R_lidar_imu, imuXYZ_fruits.T)
        translated_rotated_lidarXYZ_fruits = rotated_lidarXYZ_fruits.T + self.transvec_lidar_imu_g
        uvd_fruits = self.lidar_fruits_to_uvd(translated_rotated_lidarXYZ_fruits)
        # print(f"uvd_fruits is:\n{uvd_fruits}")
        return uvd_fruits
    
    def lidar_fruits_to_uvd(self, points_arr):
        # this is rotated to cam_coor
        rotated_points_arr = np.dot(self.R_cam_lidar, points_arr.T)
        translated_rotated_points_arr = rotated_points_arr.T + self.transvec_cam_lidar_g
        # print(f"translated_rotated_points_arr is\n {translated_rotated_points_arr}")
        homo_points = np.dot(self.camera_matrix, translated_rotated_points_arr.T).T
        # image_points -> (x,y,depth)
        uvd_points = homo_points[:, :3] / homo_points[:, 2, np.newaxis]
        uvd_points[:,2] = homo_points[:,2]
        return uvd_points

    def lidar_points_to_uvd(self, points_arr):
        rotated_points_arr = np.dot(self.R_cam_lidar, points_arr.T)
        translated_rotated_points_arr = rotated_points_arr.T + self.transvec_cam_lidar_g
        homo_points = np.dot(self.camera_matrix, translated_rotated_points_arr.T).T
        # image_points -> (x,y,depth)
        uvd_points = homo_points[:, :3] / homo_points[:, 2, np.newaxis]
        uvd_points[:,2] = homo_points[:,2]
        return uvd_points

    def colorize_fruits_uvd(self, uvd_fruits, image):
        # Convert xy from homogeneous points to image points, keep z as its real dist
        z_values = uvd_fruits[:, 2]
        # Define the colors for close and far points
        close_color = np.array([0, 0, 0])  # Red in BGR
        far_color = np.array([0, 0, 0])  # Blue in BGR
        # Calculate the colors for the points
        
        # 2025 THIS COLOR IS NOT WORKING
        colors = (1 - z_values[:, np.newaxis]) * close_color + z_values[:, np.newaxis] * far_color
        # Concatenate the image points and colors
        colored_image_points = np.hstack((uvd_fruits, colors))
        for x, y, z, b,g,r in np.int32(colored_image_points):
            # print(f"b,g,r:{b},{g},{r}")
            # print(f"the fruit ponit is at({x},{y})")
            if(z>0):
                cv2.circle(image, (x, y), 15, color=(255, int(g), int(r)), thickness=11)
            else:
                cv2.circle(image, (x, y), 5, color=(255, int(255), int(r)), thickness=1)

        return image
    
    def draw_bbx(self, image, color_string, x, y, size):
        # draw the fruit first
        color = (0, 255, 0)

        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w = size
        h = size
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        # Add a text label with a background color
        label = color_string
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1-15, y1
        if(color_string == 'yellow'):
            image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,255,255), thickness=3)
            cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0,255,255), -1)
            cv2.putText(image, "Yellow", (text_x, text_y), font, font_scale, (0,0,0), thickness)
        if(color_string == 'red'):
            image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,0,255), thickness=3)
            red_text_x =text_x+3
            cv2.rectangle(image, (red_text_x, text_y - text_size[1]), (red_text_x + text_size[0], text_y), (0,0,255), -1)
            cv2.putText(image, "Red", (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
    def draw_red_bbx(self, image, color_string, x, y, size, red_id):
        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w = size
        h = size
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
        # Add a text label with a background color
        label = "Red"+str(red_id - 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1-10, y1

        # image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,0,255), thickness=3)
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0,0,255), -1)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
    def draw_yellow_bbx(self, image, color_string, x, y, size, yellow_id):
        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w = size
        h = size
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 255), 2)
        # Add a text label with a background color
        label = "Yellow"+str(yellow_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1-20, y1
        # image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,255,255), thickness=3)
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0,255,255), -1)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
    
if __name__ == '__main__':
    try:
        lr = LidarReprojector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass