import cv2
import mediapipe as mp
import math
import numpy as np
import platform
import time

# Check what OS we're running on and import the right libraries
current_os = platform.system()
if current_os == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        windows_audio_available = True
    except ImportError:
        print("Warning: pycaw not installed. Run 'pip install pycaw' for Windows volume control")
        windows_audio_available = False
elif current_os == "Darwin":  # macOS
    import os
    windows_audio_available = False
elif current_os == "Linux":
    import subprocess
    windows_audio_available = False

class HandGestureVolumeControl:
    def __init__(self):
        # Set up MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only track one hand to avoid confusion
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Try to find and connect to a working camera
        self.camera = None
        self._initialize_camera()
        
        if self.camera is None:
            print("ERROR: No working camera found!")
            print("Check that your camera is connected and permissions are granted")
            return
            
        # Set up reasonable video resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize volume control for the current OS
        self._setup_volume_control()
        
        # These control how sensitive the gesture detection is
        self.closest_distance = 30    # Minimum finger distance (volume = 0%)
        self.farthest_distance = 200  # Maximum finger distance (volume = 100%)
        
        # Start with the current system volume instead of a fixed value
        self.current_volume_level = self._get_system_volume()
        self.smoothing_factor = 0.2  # How fast volume changes (0.1 = slow, 0.5 = fast)
        
        # For calculating FPS display
        self.frame_rate = 0
        self.last_frame_time = 0
        
    def _initialize_camera(self):
        """Try different camera indices to find one that works"""
        for camera_index in range(3):  # Check cameras 0, 1, 2
            test_cap = cv2.VideoCapture(camera_index)
            if test_cap.isOpened():
                # Test if we can actually read frames
                success, test_frame = test_cap.read()
                if success:
                    self.camera = test_cap
                    print(f"Successfully connected to camera {camera_index}")
                    return
                test_cap.release()
        
    def _setup_volume_control(self):
        """Initialize volume control based on the operating system"""
        if current_os == "Windows" and windows_audio_available:
            try:
                # Get the default audio device
                audio_devices = AudioUtilities.GetSpeakers()
                audio_interface = audio_devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume_controller = cast(audio_interface, POINTER(IAudioEndpointVolume))
                
                # Get the volume range (usually around -65 to 0 dB)
                self.min_volume_db, self.max_volume_db = self.volume_controller.GetVolumeRange()[:2]
                print(f"Windows audio control ready (range: {self.min_volume_db:.1f} to {self.max_volume_db:.1f} dB)")
            except Exception as error:
                print(f"Failed to initialize Windows volume control: {error}")
                self.volume_controller = None
        else:
            self.volume_controller = None
            print(f"Volume control ready for {current_os}")
    
    def _get_system_volume(self):
        """Read the current system volume level"""
        try:
            if current_os == "Windows" and self.volume_controller:
                # Convert dB level to percentage
                current_db = self.volume_controller.GetMasterVolumeLevel()
                volume_percentage = ((current_db - self.min_volume_db) / 
                                   (self.max_volume_db - self.min_volume_db)) * 100
                return max(0, min(100, volume_percentage))
            elif current_os == "Darwin":  # macOS
                # Use AppleScript to get volume
                result = os.popen("osascript -e 'output volume of (get volume settings)'").read().strip()
                return float(result) if result.isdigit() else 50
            elif current_os == "Linux":
                # Use amixer to get ALSA volume
                result = subprocess.run(['amixer', '-D', 'pulse', 'sget', 'Master'], 
                                      capture_output=True, text=True)
                # Parse the output to find volume percentage
                import re
                volume_match = re.search(r'\[(\d+)%\]', result.stdout)
                return float(volume_match.group(1)) if volume_match else 50
        except Exception as error:
            print(f"Couldn't read system volume: {error}")
        
        return 50  # Safe default if we can't read the volume
    
    def _change_system_volume(self, percentage):
        """Set the system volume to a specific percentage"""
        try:
            if current_os == "Windows" and self.volume_controller:
                # Convert percentage back to dB
                db_level = self.min_volume_db + (self.max_volume_db - self.min_volume_db) * (percentage / 100)
                self.volume_controller.SetMasterVolumeLevel(db_level, None)
            elif current_os == "Darwin":  # macOS
                # Use AppleScript to set volume
                os.system(f"osascript -e 'set volume output volume {int(percentage)}'")
            elif current_os == "Linux":
                # Use amixer to set ALSA volume
                subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{int(percentage)}%'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as error:
            print(f"Failed to set volume: {error}")
    
    def _calculate_finger_distance(self, point1, point2):
        """Calculate the pixel distance between two finger positions"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def _draw_volume_indicator(self, frame, volume_level):
        """Draw a volume bar on the screen"""
        # Volume bar dimensions and position
        bar_left = 50
        bar_top = 50
        bar_width = 200
        bar_height = 20
        
        # Draw the background bar (gray)
        cv2.rectangle(frame, (bar_left, bar_top), 
                     (bar_left + bar_width, bar_top + bar_height), 
                     (50, 50, 50), -1)
        
        # Draw the filled portion based on volume level
        fill_width = int(bar_width * volume_level / 100)
        bar_color = (0, 255, 0) if volume_level > 20 else (0, 165, 255)  # Green or orange
        cv2.rectangle(frame, (bar_left, bar_top), 
                     (bar_left + fill_width, bar_top + bar_height), 
                     bar_color, -1)
        
        # Add text showing the exact percentage
        cv2.putText(frame, f'Volume: {int(volume_level)}%', 
                   (bar_left, bar_top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_finger_tracking(self, frame, distance, thumb_x, thumb_y, index_x, index_y):
        """Draw visual feedback for finger tracking"""
        # Draw line between the two fingers
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
        
        # Draw circles on the fingertips
        cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)      # Blue for thumb
        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)     # Green for index
        
        # Show the distance in the middle of the line
        middle_x = (thumb_x + index_x) // 2
        middle_y = (thumb_y + index_y) // 2
        cv2.putText(frame, f'{int(distance)}px', 
                   (middle_x - 30, middle_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _process_video_frame(self):
        """Process one frame from the camera"""
        success, frame = self.camera.read()
        if not success:
            return None
        
        # Flip horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Look for hands in the frame
        hand_results = self.hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw all the hand connections
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract the landmark positions
                landmarks = []
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    # Convert from normalized coordinates to pixel coordinates
                    height, width, _ = frame.shape
                    pixel_x = int(landmark.x * width)
                    pixel_y = int(landmark.y * height)
                    landmarks.append([landmark_id, pixel_x, pixel_y])
                
                # We need at least 9 landmarks to get thumb and index finger
                if len(landmarks) >= 9:
                    # Get thumb tip (landmark 4) and index finger tip (landmark 8)
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    
                    thumb_x, thumb_y = thumb_tip[1], thumb_tip[2]
                    index_x, index_y = index_tip[1], index_tip[2]
                    
                    # Calculate distance between fingertips
                    finger_distance = self._calculate_finger_distance(
                        (thumb_x, thumb_y), (index_x, index_y)
                    )
                    
                    # Map the distance to a volume percentage
                    target_volume = np.interp(finger_distance, 
                                            [self.closest_distance, self.farthest_distance], 
                                            [0, 100])
                    target_volume = max(0, min(100, target_volume))  # Clamp to 0-100
                    
                    # Apply smoothing to avoid jumpy volume changes
                    self.current_volume_level = (
                        self.current_volume_level * (1 - self.smoothing_factor) + 
                        target_volume * self.smoothing_factor
                    )
                    
                    # Actually change the system volume
                    self._change_system_volume(self.current_volume_level)
                    
                    # Draw the visual feedback
                    self._draw_finger_tracking(frame, finger_distance, thumb_x, thumb_y, index_x, index_y)
                    
                    # Show debug information
                    cv2.putText(frame, f'Distance: {int(finger_distance)}px', 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f'Target: {int(target_volume)}%', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        # Always draw the volume bar
        self._draw_volume_indicator(frame, self.current_volume_level)
        
        # Calculate and show FPS
        current_time = time.time()
        if self.last_frame_time > 0:
            self.frame_rate = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        cv2.putText(frame, f'FPS: {int(self.frame_rate)}', 
                   (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add instructions for the user
        cv2.putText(frame, 'Pinch thumb and index finger together/apart to control volume', 
                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, 'Press Q=quit, R=reset volume, C=calibrate range', 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Detection range: {self.closest_distance}-{self.farthest_distance} pixels', 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _calibrate_detection_range(self):
        """Let the user calibrate the min/max finger distances"""
        print("\n=== CALIBRATION MODE ===")
        print("Step 1: Put your fingers as CLOSE as possible, then press 'S' to set minimum")
        print("Step 2: Put your fingers as FAR as possible, then press 'L' to set maximum")
        print("Step 3: Press 'D' when you're done calibrating")
        
        calibrating = True
        while calibrating:
            success, frame = self.camera.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            
            current_distance = 0
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = []
                    for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                        height, width, _ = frame.shape
                        pixel_x = int(landmark.x * width)
                        pixel_y = int(landmark.y * height)
                        landmarks.append([landmark_id, pixel_x, pixel_y])
                    
                    if len(landmarks) >= 9:
                        thumb_tip = landmarks[4]
                        index_tip = landmarks[8]
                        thumb_x, thumb_y = thumb_tip[1], thumb_tip[2]
                        index_x, index_y = index_tip[1], index_tip[2]
                        current_distance = self._calculate_finger_distance((thumb_x, thumb_y), (index_x, index_y))
                        self._draw_finger_tracking(frame, current_distance, thumb_x, thumb_y, index_x, index_y)
            
            # Show calibration status
            cv2.putText(frame, '=== CALIBRATION MODE ===', 
                       (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f'Current distance: {int(current_distance)} pixels', 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Minimum distance: {self.closest_distance} pixels', 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Maximum distance: {self.farthest_distance} pixels', 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Hand Gesture Volume Control', frame)
            
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('s'):
                # Set minimum distance (add small buffer to avoid zero)
                self.closest_distance = max(10, int(current_distance))
                print(f"Minimum distance set to: {self.closest_distance} pixels")
            elif key_pressed == ord('l'):
                # Set maximum distance (ensure it's bigger than minimum)
                self.farthest_distance = max(self.closest_distance + 50, int(current_distance))
                print(f"Maximum distance set to: {self.farthest_distance} pixels")
            elif key_pressed == ord('d'):
                calibrating = False
                print("Calibration finished! You can now control volume with gestures.")
    
    def start_gesture_control(self):
        """Main loop that processes video and controls volume"""
        if self.camera is None:
            print("Cannot start - no camera available")
            return
            
        print("\n=== Hand Gesture Volume Control Started ===")
        print("Hold your hand in front of the camera")
        print("Pinch your thumb and index finger together/apart to control volume")
        print("Available commands:")
        print("  Q = Quit the program")
        print("  R = Reset to current system volume")
        print("  C = Calibrate finger distance range")
        print()
        
        while True:
            frame = self._process_video_frame()
            if frame is None:
                break
                
            cv2.imshow('Hand Gesture Volume Control', frame)
            
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                print("Quitting...")
                break
            elif key_pressed == ord('r'):
                # Reset to current system volume
                self.current_volume_level = self._get_system_volume()
                print(f"Volume reset to system level: {int(self.current_volume_level)}%")
            elif key_pressed == ord('c'):
                # Enter calibration mode
                self._calibrate_detection_range()
    
    def cleanup(self):
        """Clean up camera and windows"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    # Create the gesture controller
    gesture_controller = HandGestureVolumeControl()
    
    try:
        # Start the main control loop
        gesture_controller.start_gesture_control()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
    except Exception as error:
        print(f"An error occurred: {error}")
    finally:
        # Always clean up resources
        gesture_controller.cleanup()

if __name__ == "__main__":
    main()