import rospy
from nav_msgs.msg import Odometry

def callback(data):
    with open('gps_tum1.txt', 'a') as f:
        f.write(f"{data.header.stamp.to_sec()} {data.pose.pose.position.x} {data.pose.pose.position.y} {data.pose.pose.position.z} {data.pose.pose.orientation.x} {data.pose.pose.orientation.y} {data.pose.pose.orientation.z} {data.pose.pose.orientation.w}\n")

rospy.init_node('odometry_recorder')

rospy.Subscriber('/gps_odometry', Odometry, callback)
rospy.spin()
