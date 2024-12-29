import grpc
import cv2
import numpy as np
from concurrent import futures
import face_detection_pb2
import face_detection_pb2_grpc

# Face detection logic
class FaceDetectionService(face_detection_pb2_grpc.FaceDetectionServiceServicer):

    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('model/haarcascade_default.xml')
        self.MIN_AREA = 0.15
        self.size = (640, 480)

    def ValidateFace(self, request_iterator, context):
        print("validating face")
        # Receive the image chunks
        image_data = b''
        for chunk in request_iterator:
            print("read chunk")
            image_data += chunk.image
        
        print("decode to img")
        # Decode the received image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False,
                message="Invalid image format"
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        detections = self.faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

        if len(detections) > 1:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False,
                message=f"Multiple faces detected ({len(detections)})"
            )
        elif len(detections) == 0:
            return face_detection_pb2.FaceValidationResponse(
                is_valid=False,
                message="No faces detected"
            )

        for x, y, w, h in detections:
            percent = (w * h) / (self.size[0] * self.size[1])
            if percent < self.MIN_AREA:
                return face_detection_pb2.FaceValidationResponse(
                    is_valid=False,
                    message=f"Face too small! Area={percent*100:.2f}% < {self.MIN_AREA*100:.2f}%"
                )

        return face_detection_pb2.FaceValidationResponse(
            is_valid=True,
            message=f"Face detected. Area={percent*100:.2f}%"
        )

# Server setup
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    face_detection_pb2_grpc.add_FaceDetectionServiceServicer_to_server(FaceDetectionService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
