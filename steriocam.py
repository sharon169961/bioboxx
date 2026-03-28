import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import numpy as np
import requests
import threading
import time

# --- MJPEG Streamer Helper ---
class CameraStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.bytes = b''
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while True:
            try:
                stream = requests.get(self.url, stream=True, timeout=5)
                for chunk in stream.iter_content(chunk_size=1024):
                    self.bytes += chunk
                    a = self.bytes.find(b'\xff\xd8')
                    b = self.bytes.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = self.bytes[a:b+2]
                        self.bytes = self.bytes[b+2:]
                        if len(jpg) > 0:
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                self.frame = img
            except Exception as e:
                print(f"Stream {self.url} error: {e}")
                time.sleep(2) # Retry delay

class StereoEspRosBridge(Node):
    def __init__(self):
        super().__init__('stereo_esp_bridge')
        
        # 1. QoS & Publishers
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.br = CvBridge()
        
        self.pub_l = self.create_publisher(Image, '/camera/left/image_raw', qos)
        self.pub_r = self.create_publisher(Image, '/camera/right/image_raw', qos)
        self.info_l = self.create_publisher(CameraInfo, '/camera/left/camera_info', qos)
        self.info_r = self.create_publisher(CameraInfo, '/camera/right/camera_info', qos)
        self.pub_imu = self.create_publisher(Imu, '/imu/data', qos)

        # 2. Camera Streams
        self.left_stream = CameraStream("http://left-cam.local/stream")
        self.right_stream = CameraStream("http://right-cam.local/stream")
        self.imu_url = "http://left-cam.local/imu"

        # 3. Timer (15-20 Hz is realistic for ESP32 streams)
        self.timer = self.create_timer(0.05, self.sync_callback)

        # 4. Calibration Data (Placeholders - Update these after calibration!)
        self.left_info = self.get_placeholder_info("camera_left_optical_frame")
        self.right_info = self.get_placeholder_info("camera_right_optical_frame")

    def get_placeholder_info(self, frame_id):
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width, msg.height = 320, 240
        # K: Intrinsic matrix, P: Projection matrix (Placeholders)
        msg.k = [250.0, 0.0, 160.0, 0.0, 250.0, 120.0, 0.0, 0.0, 1.0]
        msg.p = [250.0, 0.0, 160.0, 0.0, 0.0, 250.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def sync_callback(self):
        img_l = self.left_stream.frame
        img_r = self.right_stream.frame

        if img_l is not None and img_r is not None:
            now = self.get_clock().now().to_msg()

            # --- Process IMU ---
            try:
                data = requests.get(self.imu_url, timeout=0.03).json()
                imu_msg = Imu()
                imu_msg.header.stamp = now
                imu_msg.header.frame_id = "imu_link"
                imu_msg.linear_acceleration.x = float(data['accel']['x'])
                imu_msg.linear_acceleration.y = float(data['accel']['y'])
                imu_msg.linear_acceleration.z = float(data['accel']['z'])
                # BNO055 orientation (convert Euler to Quaternion if possible)
                self.pub_imu.publish(imu_msg)
            except: pass

            # --- Process Left Eye ---
            msg_l = self.br.cv2_to_imgmsg(img_l, encoding='bgr8')
            msg_l.header.stamp = now
            msg_l.header.frame_id = "camera_left_optical_frame"
            self.left_info.header.stamp = now
            
            # --- Process Right Eye ---
            # Ensure sizes match if sensors differ
            if img_r.shape != img_l.shape:
                img_r = cv2.resize(img_r, (img_l.shape[1], img_l.shape[0]))
            
            msg_r = self.br.cv2_to_imgmsg(img_r, encoding='bgr8')
            msg_r.header.stamp = now
            msg_r.header.frame_id = "camera_right_optical_frame"
            self.right_info.header.stamp = now

            # --- Publish Everything ---
            self.pub_l.publish(msg_l)
            self.info_l.publish(self.left_info)
            self.pub_r.publish(msg_r)
            self.info_r.publish(self.right_info)

def main(args=None):
    rclpy.init(args=args)
    node = StereoEspRosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()