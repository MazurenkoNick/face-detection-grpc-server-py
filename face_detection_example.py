from concurrent import futures
import grpc
import cv2
import numpy as np
import face_detection_pb2
import face_detection_pb2_grpc

# Constants
MIN_AREA_RATIO = 0.15

# Load the pre-trained face detection model
faceCascade = cv2.CascadeClassifier('model/haarcascade_default.xml')

class FaceDetectionService(face_detection_pb2_grpc.FaceDetectionServiceServicer):

    def ValidateFace(self, request, context):
        # Decode the image from the request
        nparr = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False, message="Invalid image format."
            )

        # Get image dimensions
        image_height, image_width = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        detections = faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20)
        )

        if len(detections) == 0:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False, message="No faces detected."
            )

        if len(detections) > 1:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False, message="Multiple faces detected."
            )

        x, y, w, h = detections[0]
        face_area = w * h
        image_area = image_width * image_height
        area_ratio = face_area / image_area

        if area_ratio < MIN_AREA_RATIO:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False, message="Face is too small."
            )

        # If all checks pass, return success
        return face_detection_pb2.FaceValidationResponse(
            is_valid=True, message=f"Face is valid. Image dimensions: {image_width}x{image_height}."
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    face_detection_pb2_grpc.add_FaceDetectionServiceServicer_to_server(FaceDetectionService(), server)
    server.add_insecure_port('[::]:50051')
    print("Face Detection Service is running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
