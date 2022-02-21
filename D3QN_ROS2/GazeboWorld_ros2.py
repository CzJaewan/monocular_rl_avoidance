import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor

import math
import time
import numpy as np
import cv2
import copy
import random

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class GazeboWorld():
	def __init__(self):
		# initiliaze

		rclpy.init(args=None)
		self.node = rclpy.create_node('GazeboWorld')
		qos_profile = QoSProfile(depth=10)
		self.rate = self.node.create_rate(0.1)
		
		#-----------Default Robot State-----------------------
		self.set_self_state = ModelState()
		self.set_self_state.model_name = 'turtlebot3_waffle_depth' 
		self.set_self_state.pose.position.x = 0.5
		self.set_self_state.pose.position.y = 0.
		self.set_self_state.pose.position.z = 0.
		self.set_self_state.pose.orientation.x = 0.0
		self.set_self_state.pose.orientation.y = 0.0
		self.set_self_state.pose.orientation.z = 0.0
		self.set_self_state.pose.orientation.w = 1.0
		self.set_self_state.twist.linear.x = 0.
		self.set_self_state.twist.linear.y = 0.
		self.set_self_state.twist.linear.z = 0.
		self.set_self_state.twist.angular.x = 0.
		self.set_self_state.twist.angular.y = 0.
		self.set_self_state.twist.angular.z = 0.
		self.set_self_state.reference_frame = 'world'

		#------------Params--------------------
		self.depth_image_size = [160, 128]
		self.rgb_image_size = [304, 228]
		self.bridge = CvBridge()

		self.object_state = [0, 0, 0, 0]
		self.object_name = []

		# 0. | left 90/s | left 45/s | right 45/s | right 90/s | acc 1/s | slow down -1/s
		self.action_table = [0.4, 0.2, np.pi/6, np.pi/12, 0., -np.pi/12, -np.pi/6]
		
		self.self_speed = [.4, 0.0]
		self.default_states = None
		
		self.start_time = time.time()
		self.max_steps = 10000

		self.depth_image = None
		self.rgb_image = None

		self.bump = False

		#-----------Publisher and Subscriber-------------

		self.cmd_vel = self.node.create_publisher(Twist, 'cmd_vel', qos_profile)
		self.set_state = self.node.create_publisher(ModelState, 'gazebo/set_model_state', qos_profile)
		self.resized_depth_img = self.node.create_publisher(Image, 'camera/depth/image_resized', qos_profile)
		self.resized_rgb_img = self.node.create_publisher(Image, 'camera/rgb/image_resized', qos_profile)
		
		self.object_state_sub = self.node.create_subscription(ModelStates, 'gazebo/model_states', self.ModelStateCallBack, qos_profile)
		#self.depth_image_sub = self.node.create_subscription(Image, 'camera/depth/image_raw', self.DepthImageCallBack, qos_profile)
		#self.rgb_image_sub = self.node.create_subscription(Image, 'camera/rgb/image_raw', self.RGBImageCallBack, qos_profile)
		self.depth_image_sub = self.node.create_subscription(Image, 'intel_realsense_r200_depth/depth/image_raw', self.DepthImageCallBack, qos_profile)
		self.rgb_image_sub = self.node.create_subscription(Image, 'intel_realsense_r200_depth/image_raw', self.RGBImageCallBack, qos_profile)
		
		self.laser_sub = self.node.create_subscription(LaserScan, 'scan', self.LaserScanCallBack, qos_profile)
		self.odom_sub = self.node.create_subscription(Odometry, 'odom', self.OdometryCallBack, qos_profile)
		#self.bumper_sub = self.node.create_subscription(BumperEvent, 'turtlebot3_waffle/events/bumper', self.BumperCallBack, qos_profile)
		
		#----------client---------------------------------
		self.reset_cli = self.node.create_client(Empty, 'reset_world')
		self.req_reset = Empty.Request()

		#rclpy.time.sleep(1.0)
		print("[GazeboWorld] finish init")
		# What function to call when you ctrl + c    
		#rclpy.on_shutdown(self.shutdown)
		while self.depth_image is None or self.rgb_image is None:
			rclpy.spin_once(self.node)
			pass

	def euler_from_quaternion(self, quaternion):
		"""
		Converts quaternion (w in last place) to euler roll, pitch, yaw
		quaternion = [x, y, z, w]
		Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
		"""
		x = quaternion.x
		y = quaternion.y
		z = quaternion.z
		w = quaternion.w

		sinr_cosp = 2 * (w * x + y * z)
		cosr_cosp = 1 - 2 * (x * x + y * y)
		roll = np.arctan2(sinr_cosp, cosr_cosp)

		sinp = 2 * (w * y - z * x)
		pitch = np.arcsin(sinp)

		siny_cosp = 2 * (w * z + x * y)
		cosy_cosp = 1 - 2 * (y * y + z * z)
		yaw = np.arctan2(siny_cosp, cosy_cosp)

		return roll, pitch, yaw

	def ModelStateCallBack(self, data):
		# self state
		idx = data.name.index("turtlebot3_waffle_depth")
		quaternion = (data.pose[idx].orientation.x,
					  data.pose[idx].orientation.y,
					  data.pose[idx].orientation.z,
					  data.pose[idx].orientation.w)
		
		#euler = tf2.transformations.euler_from_quaternion(quaternion)
		euler = self.euler_from_quaternion(quaternion)
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]
		self.self_state = [data.pose[idx].position.x, 
					  	  data.pose[idx].position.y,
					  	  yaw,
					  	  data.twist[idx].linear.x,
						  data.twist[idx].linear.y,
						  data.twist[idx].angular.z]
		for lp in range(len(self.object_name)):
			idx = data.name.index(self.object_name[lp])
			quaternion = (data.pose[idx].orientation.x,
						  data.pose[idx].orientation.y,
						  data.pose[idx].orientation.z,
						  data.pose[idx].orientation.w)

			euler = self.euler_from_quaternion(quaternion)
			#euler = tf2.transformations.euler_from_quaternion(quaternion)
			roll = euler[0]
			pitch = euler[1]
			yaw = euler[2]

			self.object_state[lp] = [data.pose[idx].position.x, 
									 data.pose[idx].position.y,
									 yaw]

		if self.default_states is None:
			self.default_states = copy.deepcopy(data)

	
	def DepthImageCallBack(self, img):
		print("get DepthImageCallBack")
		#print(img)
		self.depth_image = img

	def RGBImageCallBack(self, img):
		print("rgb image")
		self.rgb_image = img
	
	def LaserScanCallBack(self, scan):
		#print("get laser")

		self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
						   scan.scan_time, scan.range_min, scan. range_max]
		self.scan = np.array(scan.ranges)

	def OdometryCallBack(self, odometry):
		self.self_linear_x_speed = odometry.twist.twist.linear.x
		self.self_linear_y_speed = odometry.twist.twist.linear.y
		self.self_rotation_z_speed = odometry.twist.twist.angular.z

	#def BumperCallBack(self, bumper_data):
	#	if bumper_data.state == BumperEvent.PRESSED:
	#		self.bump = True
	#	else:
	#		self.bump = False
	
	def GetDepthImageObservation(self):
		rclpy.spin_once(self.node)
		# ros image to cv2 image
		try:
			#cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
			cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="32FC1")

		except Exception as e:
			raise e
		try:
			cv_rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		cv_img = np.array(cv_img, dtype=np.float32)
		# resize
		dim = (self.depth_image_size[0], self.depth_image_size[1])
		cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

		cv_img[np.isnan(cv_img)] = 0.
		cv_img[cv_img < 0.4] = 0.
		cv_img/=(10./255.)

		# cv_img/=(10000./255.)
		# print 'max:', np.amax(cv_img), 'min:', np.amin(cv_img)
		# cv_img[cv_img > 5.] = -1.

		# # inpainting
		# mask = copy.deepcopy(cv_img)
		# mask[mask == 0.] = 1.
		# mask[mask != 1.] = 0.
		# mask = np.uint8(mask)
		# cv_img = cv2.inpaint(np.uint8(cv_img), mask, 3, cv2.INPAINT_TELEA)

		# # guassian noise
		# gauss = np.random.normal(0., 0.5, dim)
		# gauss = gauss.reshape(dim[1], dim[0])
		# cv_img = np.array(cv_img, dtype=np.float32)
		# cv_img = cv_img + gauss
		# cv_img[cv_img<0.00001] = 0.

		# # smoothing
		# kernel = np.ones((4,4),np.float32)/16
		# cv_img = cv2.filter2D(cv_img,-1,kernel)

		cv_img = np.array(cv_img, dtype=np.float32)
		cv_img*=(10./255.)

		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)
		return(cv_img/5.)

	def GetRGBImageObservation(self):
		rclpy.spin_once(self.node)

		# ros image to cv2 image
		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		# resize
		dim = (self.rgb_image_size[0], self.rgb_image_size[1])
		cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
		except Exception as e:
			raise e
		self.resized_rgb_img.publish(resized_img)
		return(cv_resized_img)

	def PublishDepthPrediction(self, depth_img):
		rclpy.spin_once(self.node)

		# cv2 image to ros image and publish
		cv_img = np.array(depth_img, dtype=np.float32)
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)

	def GetLaserObservation(self):
		rclpy.spin_once(self.node)

		scan = copy.deepcopy(self.scan)
		scan[np.isnan(scan)] = 30.
		return scan

	def GetSelfState(self):
		rclpy.spin_once(self.node)

		return self.self_state;

	def GetSelfLinearXSpeed(self):
		rclpy.spin_once(self.node)

		return self.self_linear_x_speed

	def GetSelfOdomeSpeed(self):
		rclpy.spin_once(self.node)

		v = np.sqrt(self.self_linear_x_speed**2 + self.self_linear_y_speed**2)
		return [v, self.self_rotation_z_speed]

	def GetTargetState(self, name):
		rclpy.spin_once(self.node)

		return self.object_state[self.TargetName.index(name)]

	def GetSelfSpeed(self):
		rclpy.spin_once(self.node)

		return np.array(self.self_speed)

	def GetBump(self):
		rclpy.spin_once(self.node)

		return self.bump

	def quaternion_from_euler(self, roll, pitch, yaw):
		"""
		Converts euler roll, pitch, yaw to quaternion (w in last place)
		quat = [x, y, z, w]
		Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
		"""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		cp = math.cos(pitch * 0.5)
		sp = math.sin(pitch * 0.5)
		cr = math.cos(roll * 0.5)
		sr = math.sin(roll * 0.5)

		q = [0] * 4
		q[0] = cy * cp * cr + sy * sp * sr
		q[1] = cy * cp * sr - sy * sp * cr
		q[2] = sy * cp * sr + cy * sp * cr
		q[3] = sy * cp * cr - cy * sp * sr

		return q

	def SetObjectPose(self, name='turtlebot3_waffle_depth', random_flag=False):
		rclpy.spin_once(self.node)
		print("reset")
		quaternion = self.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))

		
		#quaternion = tf2.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
		if name is 'turtlebot3_waffle_depth' :
			object_state = copy.deepcopy(self.set_self_state)
			object_state.pose.orientation.x = quaternion[0]
			object_state.pose.orientation.y = quaternion[1]
			object_state.pose.orientation.z = quaternion[2]
			object_state.pose.orientation.w = quaternion[3]
		else:
			object_state = self.States2State(self.default_states, name)

		self.set_state.publish(object_state)

	def States2State(self, states, name):
		to_state = ModelState()
		from_states = copy.deepcopy(states)
		idx = from_states.name.index(name)
		to_state.model_name = name
		to_state.pose = from_states.pose[idx]
		to_state.twist = from_states.twist[idx]
		to_state.reference_frame = 'world'
		return to_state


	def ResetWorld(self):
		self.reset_cli.call_async(self.req_reset)

		self.SetObjectPose() # reset robot
		for x in range(len(self.object_name)): 
			self.SetObjectPose(self.object_name[x]) # reset target
		self.self_speed = [.4, 0.0]
		self.step_target = [0., 0.]
		self.step_r_cnt = 0.
		self.start_time = time.time()
		#rate.sleep(0.5)


	def Control(self, action):
		rclpy.spin_once(self.node)

		if action <2:
			self.self_speed[0] = self.action_table[action]
			# self.self_speed[1] = 0.
		else:
			self.self_speed[1] = self.action_table[action]
			
		move_cmd = Twist()
		move_cmd.linear.x = self.self_speed[0]
		move_cmd.linear.y = 0.
		move_cmd.linear.z = 0.
		move_cmd.angular.x = 0.
		move_cmd.angular.y = 0.
		move_cmd.angular.z = self.self_speed[1]
		self.cmd_vel.publish(move_cmd)

	def shutdown(self):
		rclpy.spin_once(self.node)

		# stop turtlebot
		#rospy.loginfo("Stop Moving") #check
		self.cmd_vel.publish(Twist())
		#rate.sleep(1.)
		
	def GetRewardAndTerminate(self, t):
		rclpy.spin_once(self.node)

		terminate = False
		reset = False
		result = "Reach Goal"
		[v, theta] = self.GetSelfOdomeSpeed()
		laser = self.GetLaserObservation()
		reward = v * np.cos(theta) * 0.2 - 0.01

		if self.GetBump() or np.amin(laser) < 0.30 or np.amin(laser) == 30.:
			reward = -10.
			terminate = True
			reset = True
			result = 'Crashed'
		if t > 500:
			reset = True
			result = 'timeout'

		return reward, terminate, reset, result	
	
def main(args=None):
    #rclpy.init(args=args)
    environment = GazeboWorld()
    rclpy.spin(environment.node)

    environment.destroy()
    #rclpy.shutdown()


if __name__ == '__main__':
    main()