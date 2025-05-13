import rospy
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler
import math

class GPSToOdometry:
    def __init__(self):
        rospy.init_node('gps_to_odometry', anonymous=True)
        self.pub = rospy.Publisher('gps_odometry', Odometry, queue_size=10)
        rospy.Subscriber('/hawkblue/mavros/global_position/global', NavSatFix, self.callback)
        self.frame_id = 'world'
        self.child_frame_id = 'base_link'
        self.init_point_set = False
        self.init_lat = 0
        self.init_lon = 0
        self.init_alt = 0

    def callback(self, data):
        if not self.init_point_set:
            self.init_lat = data.latitude
            self.init_lon = data.longitude
            self.init_alt = data.altitude
            self.init_point_set = True

        odom = Odometry()
        # odom.header.stamp = rospy.Time.now()
        odom.header.stamp = data.header.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.latlon_to_xy(data.latitude, data.longitude)
        odom.pose.pose.position.z = data.altitude - self.init_alt
        quaternion = quaternion_from_euler(0, 0, 0)  # Assuming no orientation information
        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]
        self.pub.publish(odom)

    def latlon_to_xy(self, lat, lon):
        R = 6378137  # Earth's radius in meters
        x = R * (math.radians(lon) - math.radians(self.init_lon)) * math.cos(math.radians(self.init_lat))
        y = R * (math.radians(lat) - math.radians(self.init_lat))
        return x, y

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = GPSToOdometry()
        node.run()
    except rospy.ROSInterruptException:
        pass
