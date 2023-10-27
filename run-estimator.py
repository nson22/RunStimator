import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class RunEstimator:
    def __init__(self, file_name="run.mp4", model_asset_path="pose_landmarker_full.task"):
        self.file_name = file_name
        self.model_asset_path = model_asset_path
        self.counter = 0
        self.options = PoseLandmarkerOptions(base_options=BaseOptions(
            model_asset_path), running_mode=VisionRunningMode.VIDEO)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.,
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.,
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(
            x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def display_video(self, show_annotation=True):
        with PoseLandmarker.create_from_options(self.options) as landmark:
            capture = cv2.VideoCapture(filename=self.file_name)
            fps = capture.get(cv2.CAP_PROP_FPS)
            calc_timestamp = 0

            while capture.isOpened():
                ret, frame = capture.read()

                if ret == True:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    calc_timestamp = int(calc_timestamp + 1000 / fps)
                    detection_result = landmark.detect_for_video(
                        mp_image, calc_timestamp)
                    
                    if show_annotation:
                        frame = self.draw_landmarks_on_image(frame, detection_result)

                    cv2.imshow(f"{self.file_name}", frame)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    
                    self.calc_steps(detection_result, 25, 26)
                else:
                    break

            capture.release()
            cv2.destroyAllWindows()

    def calc_steps(self, detection_result, left_knee_point, right_knee_point):
        land = detection_result.pose_landmarks[0]
        left_knee_values, right_knee_values = (land[left_knee_point].x, land[right_knee_point].x)
        
        if left_knee_values == right_knee_values:
            flag = "gtr"
            self.counter = self.counter + 1
        print(f"{right_knee_values:.4f} {left_knee_values:.4f}")
                
    
if __name__ == "__main__":
    app = RunEstimator()
    app.display_video()