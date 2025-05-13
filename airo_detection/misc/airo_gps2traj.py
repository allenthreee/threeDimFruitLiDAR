import rospy
import math
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf

class NavSatFixToTrajectory:
    def __init__(self):
        rospy.init_node('navsatfix_to_trajectory', anonymous=True)
        self.pub = rospy.Publisher('trajectory', Path, queue_size=10)
        rospy.Subscriber('/hawkblue/mavros/global_position/global', NavSatFix, self.callback)
        self.frame_id = 'world'
        self.init_point_set = False
        self.init_ecef_x = 0
        self.init_ecef_y = 0
        self.init_ecef_z = 0
        self.path = Path()

    def callback(self, data):
        ecef_x, ecef_y, ecef_z = self.geodetic_to_ecef(data.latitude, data.longitude, data.altitude)
        if not self.init_point_set:
            self.init_ecef_x = ecef_x
            self.init_ecef_y = ecef_y
            self.init_ecef_z = ecef_z
            self.init_point_set = True

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self.frame_id
        pose_stamped.pose.position.x = ecef_x - self.init_ecef_x
        pose_stamped.pose.position.y = ecef_y - self.init_ecef_y
        pose_stamped.pose.position.z = ecef_z - self.init_ecef_z
        self.path.poses.append(pose_stamped)
        self.path.header = pose_stamped.header
        self.pub.publish(self.path)

    def geodetic_to_ecef(self, lat, lon, h):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        a = 6378137.0  # semi-major axis
        f = 1 / 298.257223563  # inverse flattening
        e_sq = 2 * f - f ** 2  # square of eccentricity
        N = a / math.sqrt(1 - e_sq * math.sin(lat_rad) ** 2)  # prime vertical radius of curvature
        x = (N + h) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + h) * math.cos(lat_rad) * math.sin(lon_rad)
        z = ((1 - e_sq) * N + h) * math.sin(lat_rad)
        return x, y, z

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = NavSatFixToTrajectory()
        node.run()
    except rospy.ROSInterruptException:
        pass
