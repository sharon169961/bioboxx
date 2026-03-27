import cv2
import numpy as np
import requests
import threading
import time

# --- CONFIGURATION ---
LEFT_URL = "http://left-cam.local/stream"
RIGHT_URL = "http://right-cam.local/stream"
IMU_URL = "http://left-cam.local/imu"

class CameraStream:
    def __init__(self, url):
        self.stream = requests.get(url, stream=True)
        self.bytes = b''
        self.frame = None
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                self.bytes += chunk
                a = self.bytes.find(b'\xff\xd8') # JPEG Start
                b = self.bytes.find(b'\xff\xd9') # JPEG End
                
                if a != -1 and b != -1:
                    jpg = self.bytes[a:b+2]
                    self.bytes = self.bytes[b+2:]
                    
                    # --- THE FIX: Guard against empty or corrupt buffers ---
                    if len(jpg) > 0:
                        data = np.frombuffer(jpg, dtype=np.uint8)
                        new_frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        
                        if new_frame is not None:
                            self.frame = new_frame
        except Exception as e:
            print(f"Stream error: {e}")

# --- INITIALIZE ---
print("Connecting to cameras...")
left_stream = CameraStream(LEFT_URL)
right_stream = CameraStream(RIGHT_URL)

stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=11)

pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])
last_time = time.time()

while True:
    img_l = left_stream.frame
    img_r = right_stream.frame

    if img_l is not None and img_r is not None:
        # 1. Align Sizes (Safety check)
        if img_l.shape != img_r.shape:
            img_r = cv2.resize(img_r, (img_l.shape[1], img_l.shape[0]))

        # 2. Compute Disparity
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(gray_l, gray_r)

        # 3. IMU Update (Async-style)
        try:
            imu_data = requests.get(IMU_URL, timeout=0.05).json()
            curr_time = time.time()
            dt = curr_time - last_time
            acc = np.array([imu_data['accel']['x'], imu_data['accel']['y'], imu_data['accel']['z']])
            
            # Integration
            vel += acc * dt
            pos += vel * dt
            last_time = curr_time
            print(f"Z-Pos: {pos[2]:.2f}m | Yaw: {imu_data['angles']['z']:.1f}")
        except:
            pass # Skip IMU if it's too slow this frame

        # 4. Show Output
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("Depth Map", cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))
        cv2.imshow("Left Eye", img_l)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()