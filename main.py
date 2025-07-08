import cv2
import mediapipe as mp
import math
import numpy as np
import platform
import time

# Platform-specific volume control imports
system = platform.system()
if system == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        WINDOWS_VOLUME = True
    except ImportError:
        print("pycaw not installed. Install with: pip install pycaw")
        WINDOWS_VOLUME = False
elif system == "Darwin":  # macOS
    import os
    WINDOWS_VOLUME = False
elif system == "Linux":
    import subprocess
    WINDOWS_VOLUME = False

class GestureVolumeController:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam with fallback options
        self.cap = None
        for i in range(3):  # Try camera indices 0, 1, 2
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.cap = cap
                    print(f"Camera {i} initialized successfully")
                    break
                cap.release()
        
        if self.cap is None:
            print("No camera found or permission denied!")
            print("Please check camera permissions in System Settings > Privacy & Security > Camera")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Volume control setup
        self.setup_volume_control()
        
        # Variables for smooth volume control
        self.min_distance = 30
        self.max_distance = 200
        # Get current system volume as starting point instead of hardcoded 50
        self.current_volume = self.get_current_volume()
        self.volume_smoothing = 0.2  # Increased for faster response
        
        # FPS calculation
        self.fps = 0
        self.prev_time = 0
        
    def setup_volume_control(self):
        """Setup volume control based on operating system"""
        if system == "Windows" and WINDOWS_VOLUME:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.min_vol, self.max_vol = self.volume.GetVolumeRange()[:2]
                print("Windows volume control initialized")
            except Exception as e:
                print(f"Error initializing Windows volume: {e}")
                self.volume = None
        else:
            self.volume = None
            print(f"{system} volume control initialized")
    
    def get_current_volume(self):
        """Get current system volume as percentage"""
        try:
            if system == "Windows" and self.volume:
                current_vol = self.volume.GetMasterVolumeLevel()
                percentage = ((current_vol - self.min_vol) / (self.max_vol - self.min_vol)) * 100
                return max(0, min(100, percentage))
            elif system == "Darwin":  # macOS
                result = os.popen("osascript -e 'output volume of (get volume settings)'").read().strip()
                return float(result) if result.isdigit() else 50
            elif system == "Linux":
                result = subprocess.run(['amixer', '-D', 'pulse', 'sget', 'Master'], 
                                      capture_output=True, text=True)
                # Parse amixer output to get volume percentage
                import re
                match = re.search(r'\[(\d+)%\]', result.stdout)
                return float(match.group(1)) if match else 50
        except Exception as e:
            print(f"Error getting current volume: {e}")
        return 50  # Default fallback
    
    def set_volume(self, percentage):
        """Set system volume based on percentage (0-100)"""
        try:
            if system == "Windows" and self.volume:
                # Convert percentage to Windows volume range
                vol_level = self.min_vol + (self.max_vol - self.min_vol) * (percentage / 100)
                self.volume.SetMasterVolumeLevel(vol_level, None)
            elif system == "Darwin":  # macOS
                os.system(f"osascript -e 'set volume output volume {int(percentage)}'")
            elif system == "Linux":
                subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{int(percentage)}%'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error setting volume: {e}")
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def draw_volume_bar(self, img, volume_percent):
        """Draw volume bar on the image"""
        bar_x, bar_y = 50, 50
        bar_width, bar_height = 200, 20
        
        # Background bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                      (50, 50, 50), -1)
        
        # Volume bar
        fill_width = int(bar_width * volume_percent / 100)
        color = (0, 255, 0) if volume_percent > 20 else (0, 165, 255)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                      color, -1)
        
        # Volume text
        cv2.putText(img, f'Volume: {int(volume_percent)}%', (bar_x, bar_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_distance_indicator(self, img, distance, x1, y1, x2, y2):
        """Draw distance indicator between fingers"""
        # Line between fingers
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # Circles on finger tips
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)  # Thumb
        cv2.circle(img, (x2, y2), 10, (0, 255, 0), -1)  # Index
        
        # Distance text
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(img, f'{int(distance)}px', (mid_x - 30, mid_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def process_frame(self):
        """Process a single frame from webcam"""
        ret, img = self.cap.read()
        if not ret:
            return None
        
        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmark_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                
                if len(landmark_list) >= 9:
                    # Get thumb tip (4) and index finger tip (8)
                    thumb_tip = landmark_list[4]
                    index_tip = landmark_list[8]
                    
                    x1, y1 = thumb_tip[1], thumb_tip[2]
                    x2, y2 = index_tip[1], index_tip[2]
                    
                    # Calculate distance
                    distance = self.calculate_distance((x1, y1), (x2, y2))
                    
                    # Map distance to volume (0-100)
                    target_volume = np.interp(distance, 
                                            [self.min_distance, self.max_distance], 
                                            [0, 100])
                    target_volume = max(0, min(100, target_volume))
                    
                    # Smooth volume changes
                    self.current_volume = (self.current_volume * (1 - self.volume_smoothing) + 
                                         target_volume * self.volume_smoothing)
                    
                    # Set system volume
                    self.set_volume(self.current_volume)
                    
                    # Draw visual indicators
                    self.draw_distance_indicator(img, distance, x1, y1, x2, y2)
                    
                    # Debug info
                    cv2.putText(img, f'Distance: {int(distance)}', (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(img, f'Target: {int(target_volume)}%', (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        # Draw volume bar
        self.draw_volume_bar(img, self.current_volume)
        
        # Calculate and display FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        
        cv2.putText(img, f'FPS: {int(self.fps)}', (img.shape[1] - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(img, 'Bring thumb and index finger closer/farther to control volume', 
                    (10, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, 'Press Q to quit, R to reset, C to calibrate', 
                    (10, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f'Range: {self.min_distance}-{self.max_distance}px', 
                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def calibrate_range(self):
        """Calibrate the min/max distance range"""
        print("\nCalibration Mode:")
        print("1. Make the smallest distance with your fingers and press 'S'")
        print("2. Make the largest distance with your fingers and press 'L'")
        print("3. Press 'D' when done")
        
        calibrating = True
        while calibrating:
            ret, img = self.cap.read()
            if not ret:
                break
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            current_distance = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    landmark_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmark_list.append([id, cx, cy])
                    
                    if len(landmark_list) >= 9:
                        thumb_tip = landmark_list[4]
                        index_tip = landmark_list[8]
                        x1, y1 = thumb_tip[1], thumb_tip[2]
                        x2, y2 = index_tip[1], index_tip[2]
                        current_distance = self.calculate_distance((x1, y1), (x2, y2))
                        self.draw_distance_indicator(img, current_distance, x1, y1, x2, y2)
            
            # Display calibration info
            cv2.putText(img, 'CALIBRATION MODE', (200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, f'Current Distance: {int(current_distance)}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f'Min Distance: {self.min_distance}', (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f'Max Distance: {self.max_distance}', (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Gesture Volume Control', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.min_distance = max(10, int(current_distance))
                print(f"Min distance set to: {self.min_distance}")
            elif key == ord('l'):
                self.max_distance = max(self.min_distance + 50, int(current_distance))
                print(f"Max distance set to: {self.max_distance}")
            elif key == ord('d'):
                calibrating = False
                print("Calibration complete!")
    
    def run(self):
        """Main loop"""
        if self.cap is None:
            print("Cannot start - no camera available")
            return
            
        print("Starting Gesture Volume Control...")
        print("Hold your hand in front of the camera")
        print("Move your thumb and index finger closer/farther to control volume")
        print("Press 'q' to quit, 'r' to reset, 'c' to calibrate")
        
        while True:
            img = self.process_frame()
            if img is None:
                break
                
            cv2.imshow('Gesture Volume Control', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset to current system volume
                self.current_volume = self.get_current_volume()
                print(f"Volume reset to current system volume: {int(self.current_volume)}%")
            elif key == ord('c'):  # Calibrate distance range
                self.calibrate_range()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

def main():
    controller = GestureVolumeController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        controller.cleanup()

if __name__ == "__main__":
    main()