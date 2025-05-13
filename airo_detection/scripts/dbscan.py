#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import numpy as np
import random
import struct
from sklearn.linear_model import RANSACRegressor

class PointCloudClusterer:
    def __init__(self):
        rospy.init_node('pointcloud_clusterer', anonymous=True)
        
        # Subscriber
        self.sub = rospy.Subscriber('/cloud_registered', PointCloud2, self.cloud_callback)
        
        # Publisher for colored point cloud
        self.pub = rospy.Publisher('/cloud_colored', PointCloud2, queue_size=10)
        
        # DBSCAN parameters
        self.eps = 0.25
        self.min_samples = 8
        
        # Ground removal parameters
        self.ground_threshold = 0.2  # Distance threshold for ground points
        self.ransac_threshold = 0.1  # Inlier threshold for RANSAC
        
        rospy.loginfo("PointCloud Clusterer initialized with ground removal")

    def remove_ground(self, points):
        """Remove ground points using RANSAC plane fitting"""
        if len(points) < 3:
            return points, np.array([])
            
        # Fit plane using RANSAC
        X = points[:, :2]  # Use only x,y coordinates for ground fitting
        z = points[:, 2]
        ransac = RANSACRegressor(residual_threshold=self.ransac_threshold)
        ransac.fit(X, z)
        
        # Calculate distances to the fitted plane
        plane_z = ransac.predict(X)
        distances = np.abs(z - plane_z)
        
        # Separate ground and non-ground points
        ground_mask = distances < self.ground_threshold
        non_ground_points = points[~ground_mask]
        ground_points = points[ground_mask]
        
        return non_ground_points, ground_points

    def cloud_callback(self, cloud_msg):
        # Convert PointCloud2 to numpy array
        points = []
        for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        if len(points) == 0:
            return
            
        points = np.array(points)
        
        # Remove ground points
        non_ground_points, ground_points = self.remove_ground(points)
        rospy.loginfo(f"Original: {len(points)} points | After ground removal: {len(non_ground_points)} points")
        
        if len(non_ground_points) == 0:
            return
            
        # Perform DBSCAN clustering on non-ground points
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(non_ground_points)
        labels = db.labels_
        
        # Number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        rospy.loginfo(f"Found {n_clusters} clusters")
        
        # Generate colors
        unique_labels = set(labels)
        colors = {}
        for label in unique_labels:
            if label == -1:
                colors[label] = (0.5, 0.5, 0.5)  # Gray for noise
            else:
                colors[label] = (random.random(), random.random(), random.random())
        
        # Create colored point cloud (non-ground points)
        colored_points = []
        for i, point in enumerate(non_ground_points):
            label = labels[i]
            r, g, b = colors[label]
            rgb = struct.unpack('I', struct.pack('BBBB', int(b*255), int(g*255), int(r*255), 255))[0]
            colored_points.append([point[0], point[1], point[2], rgb])
        
        # Add ground points (in brown)
        ground_color = struct.unpack('I', struct.pack('BBBB', 102, 51, 0, 255))[0]  # Brown color
        for point in ground_points:
            colored_points.append([point[0], point[1], point[2], ground_color])
        
        # Create output cloud
        header = cloud_msg.header
        header.frame_id = "camera_init"
        
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1)
        ]
        
        colored_cloud = pc2.create_cloud(header, fields, colored_points)
        self.pub.publish(colored_cloud)

if __name__ == '__main__':
    try:
        clusterer = PointCloudClusterer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass