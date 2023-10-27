import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

file_name = "run.mp4"
model_path = "pose_landmarker_full.task"

options = python.vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=model_path),
    running_mode=python.vision.RunningMode.VIDEO)


def draw_landmarks_on_image(rgb_image, detection_result):
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


with python.vision.PoseLandmarker.create_from_options(options) as landmark:
    capture = cv2.VideoCapture(filename=file_name)
    fps = capture.get(cv2.CAP_PROP_FPS)
    calc_timestamp = 0

    while capture.isOpened():
        ret, frame = capture.read()

        if ret == True:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            calc_timestamp = int(calc_timestamp + 1000 / fps)
            detection_result = landmark.detect_for_video(
                mp_image, calc_timestamp)
            annotated_image = draw_landmarks_on_image(frame, detection_result)

            cv2.imshow("Frame", annotated_image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    capture.release()
    cv2.destroyAllWindows()
