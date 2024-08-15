import math
import cv2 as cv
import numpy as np
import onnxruntime as ort
from time import perf_counter

from functions import resize_image, distance2bbox, softmax, align_face, make_anchors


class Onnx:
    def __init__(self, path: str, verbose=False):
        self.path = path
        self.verbose = verbose

        self._session = self.get_session(path)
        self._input_details = self.get_input_details()

    @property
    def input_details(self) -> ort.NodeArg:
        return self._input_details

    def get_input_details(self) -> ort.NodeArg:
        return self.session.get_inputs()[0]

    @property
    def session(self) -> ort.InferenceSession:
        return self._session

    def get_session(self, path: str) -> ort.InferenceSession:
        tic = perf_counter()

        session = ort.InferenceSession(
            path,
            providers=[
                'CPUExecutionProvider',
            ],
        )

        if self.verbose:
            print(f"Initialization ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000} ms.")

        return session


class YOLOv8nFace(Onnx):
    def __init__(self, path: str, verbose=False):
        super().__init__(path, verbose)

        self.iou_threshold = .5
        self.conf_threshold = .45
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.strides = (8, 16, 32)

        self._project = self.get_project()
        self._feats_hw = self.get_feats_hw()
        self._anchors = self.get_anchors()

    @staticmethod
    def preprocess(img: np.ndarray) -> tuple[np.ndarray, int, int, float, float]:
        img_height, img_width = img.shape[:2]

        img, new_height, new_width, pad_height, pad_width = resize_image(img)
        scale_h, scale_w = img_height / new_height, img_width / new_width

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0

        return img, pad_height, pad_width, scale_h, scale_w

    def inference(self, img: np.ndarray, align=True) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        tic = perf_counter()

        img_input, padh, padw, scale_h, scale_w = self.preprocess(img)
        if self.verbose:
            print(f"Preprocess time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")
            tic = perf_counter()

        outputs = self.session.run(None, {self.input_details.name: img_input})
        if self.verbose:
            print(f"Inference time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")
            tic = perf_counter()

        detected_face, (x, y, w, h) = self.postprocess(outputs, scale_h, scale_w, padh, padw, img, align=align)
        if self.verbose:
            print(f"Postprocess time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")

        return detected_face, (x, y, w, h)

    def postprocess(
            self, preds, scale_h, scale_w, padh, padw, img: np.ndarray, align=True
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        boxes_xyxy, scores, landmarks = [], [], []

        # TODO: breakdown this loop into function.
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))    # sigmoid
            kpts = pred[..., -15:].reshape((-1, 15))     # x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            box = box.reshape(-1, 4, self.reg_max)
            box = softmax(box, axis=-1)
            box = np.dot(box, self.project).reshape((-1, 4))
            bbox = distance2bbox(
                self.anchors[stride], box,
                max_shape=(self.input_height, self.input_width)
            ) * stride

            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))    # sigmoid

            # --- adjust based on pad and scale when resize
            bbox -= np.array([[padw, padh, padw, padh]])     # 合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            boxes_xyxy.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        boxes_xyxy = np.concatenate(boxes_xyxy, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        # --- convert boxes in xyxy format to xywh format
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2:4] = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)     # max_class_confidence

        mask = confidences > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        landmarks = landmarks[mask]

        indices = cv.dnn.NMSBoxes(
            boxes_xywh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold
        )
        if not len(indices) > 0:
            return np.array([]), (0, 0, 0, 0)

        # mlvl : multi-level bounding boxes
        boxes = boxes_xywh[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
        kpts = landmarks[indices]

        if boxes.size == 0:
            return np.array([]), (0, 0, 0, 0)

        x, y, w, h = [abs(int(_)) for _ in boxes[0]]
        detected_face = img[y:y + h, x:x + w]

        if align:
            left_eye_coor, left_eye_conf = kpts[0][:2], kpts[0][2]
            right_eye_coor, right_eye_conf = kpts[0][3:5], kpts[0][5]

            if left_eye_conf > .5 and right_eye_conf > .5:
                detected_face = align_face(
                    detected_face, left_eye_coor, right_eye_coor
                )

        return detected_face, (x, y, w, h)
    
    def draw_detections(self, image, boxes):
        x, y, w, h = boxes
        
        # Draw rectangle
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        
        return image

    def get_project(self) -> np.ndarray:
        return np.arange(self.reg_max)

    @property
    def project(self) -> np.ndarray:
        return self._project

    def get_feats_hw(self) -> list[tuple[int, int]]:
        return [
            (math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i]))
            for i in range(len(self.strides))
        ]

    @property
    def feats_hw(self) -> list[tuple[int, int]]:
        return self._feats_hw

    def get_anchors(self) -> dict:
        return make_anchors(self.strides, self.feats_hw)

    @property
    def anchors(self) -> dict:
        return self._anchors


class YOLOv8mMaskOpenCV:
    def __init__(self, path: str, verbose=False):
        self.model: cv.dnn.Net = cv.dnn.readNetFromONNX(path)
        self.verbose = verbose

        self.classes = ['with_mask', 'without_mask', 'mask_weared _incorrect']
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.iou_threshold = .5
        self.conf_threshold = .45
        self.input_height = 640
        self.input_width = 640

    def inference(self, img: np.ndarray, show=False) -> list[dict]:
        # Read the input image
        [height, width, _] = img.shape

        # Prepare a square image for inference
        length = max((height, width))
        img_input = np.zeros((length, length, 3), np.uint8)
        img_input[0:height, 0:width] = img

        # Calculate scale factor
        scale = length / 640

        # Preprocess the img_input and prepare blob for model
        blob = cv.dnn.blobFromImage(img_input, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()

        # Prepare output array
        outputs = np.array([cv.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": self.classes[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            self.draw_bounding_box(
                img,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        if show:
            # Display the image with bounding boxes
            cv.imshow("img", img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return detections

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        label = f"{self.classes[class_id]} ({confidence:.2f})"
        color = self.colors[class_id]
        cv.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class MiniFASNet(Onnx):
    def __init__(self, path: str, verbose=False):
        super().__init__(path, verbose)

        self._class_names = self.get_class_names()

    def inference(self, img: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
        tic = perf_counter()

        img_input = self.preprocess(img, bbox)
        if self.verbose:
            print(f"Preprocess time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")
            tic = perf_counter()

        outputs = self.session.run(None, {self.input_details.name: img_input})
        if self.verbose:
            print(f"Inference time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")
            tic = perf_counter()

        label, score, cls = self.postprocess(outputs)
        if self.verbose:
            print(f"Postprocess time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")

        return label

    def preprocess(self, img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        img = self.crop(img, bbox)
        img = np.expand_dims(img.transpose(2, 0, 1).astype(np.float32), axis=0)

        return img

    def postprocess(self, outputs: list[np.ndarray]) -> tuple[int, float, str]:
        prediction = softmax(outputs[0].squeeze(), axis=0)

        label = np.argmax(prediction)
        score, cls = prediction[label], self.class_names[label]

        return label, score, cls

    def crop(
            self, img: np.ndarray, bbox: tuple[int, int, int, int],
            scale: float = 2.7, out_height: int = 80, out_width: int = 80
    ) -> np.ndarray:
        img_height, img_width = img.shape[0], img.shape[1]

        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self.get_new_bbox(img_height, img_width, bbox, scale)
        img = img[left_top_y: right_bottom_y + 1, left_top_x: right_bottom_x + 1]
        img = cv.resize(img, (out_width, out_height))

        return img

    @staticmethod
    def get_new_bbox(img_height: int, img_width: int, bbox: tuple[int, int, int, int], scale: float):
        x, y, w, h = bbox

        scale = min(
            (img_height - 1) / h,
            min(
                (img_width - 1) / w,
                scale
            )
        )

        new_width = w * scale
        new_height = h * scale
        center_x, center_y = w / 2 + x, h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > img_width - 1:
            left_top_x -= right_bottom_x - img_width + 1
            right_bottom_x = img_width - 1

        if right_bottom_y > img_height - 1:
            left_top_y -= right_bottom_y - img_height + 1
            right_bottom_y = img_height - 1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    @property
    def class_names(self):
        return self._class_names

    @staticmethod
    def get_class_names():
        return '2DSpoof', 'Real', '3DSpoof'


class Facenet512(Onnx):
    def __init__(self, path: str, verbose=False):
        super().__init__(path, verbose)
        self.verbose = verbose

    def preprocess(self, img: np.ndarray, out_height=160, out_width=160) -> np.ndarray:
        # --- Preprocessing to Match Model Input Shape
        img, new_height, new_width, top, left = resize_image(img, True, out_height, out_width)
        img = np.expand_dims(img.astype(np.float32), 0) / 255.0

        # --- Facenet2018 Normalization
        img *= 255
        img /= 127.5
        img -= 1

        # --- exception for original facenet because trained facenet with custom dataset produce error if we specify
        # --- input data shape on onnx same as original facenet. so, just keep this specialization if you don't want to
        # --- change trained facenet input data shape.
        # if 'Facenet512.onnx' in self.path:
        return img

        # img = img.transpose(0, 3, 1, 2)
        # return img

    def inference(self, img: np.ndarray) -> np.ndarray:
        tic = perf_counter()

        img_input = self.preprocess(img)
        if self.verbose:
            print(f"Preprocess time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")
            tic = perf_counter()

        outputs = self.session.run(None, {self.input_details.name: img_input})
        if self.verbose:
            print(f"Inference time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")

        embedding = self.postprocess(outputs)
        if self.verbose:
            print(f"Postprocess time ({self.__class__.__name__}) : {(perf_counter() - tic) * 1000:.2f} ms")

        return embedding

    @staticmethod
    def postprocess(outputs: list[np.ndarray]) -> np.ndarray:
        return outputs[0].squeeze()


if __name__ == '__main__':
    import cv2 as cv
    from PIL import ImageEnhance, Image

    run_img = cv.imread("data/Attendance Photos/Test/CLIVE SWEETLY YORRISEN DIRK/CLIVE SWEETLY YORRISEN DIRK.jpeg")
    cv.imshow('img1', run_img)
    
    # https://stackoverflow.com/questions/33022578/improve-image-quality
    # try this : https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066
    run_img = cv.detailEnhance(run_img, sigma_s=100, sigma_r=0.55)
    
    run_detector_face = YOLOv8nFace('model/yolov8n-face-hpc203.onnx')
    # run_detector_liveness = MiniFASNet('model/2.7_80x80_MiniFASNetV2.onnx')
    # run_verificator_face = Facenet512('model/facenet512_weights.onnx')

    run_face, run_bbox = run_detector_face.inference(run_img)
    print(run_face.shape)
    # run_label = run_detector_liveness.inference(run_img, run_bbox)
    # run_embedding = run_verificator_face.inference(run_face).squeeze()
    
    cv.imshow('img2', run_img)
    cv.imshow('face', run_face)
    cv.waitKey(0)
    
    raise

    # --- check
    run_img = cv.imread('D:/projects/EATSAI/server/hras/attendance/15/1d74691f-05ce-469c-a6b9-b6e3e0f8ee6e.jpg')
    run_face, run_bbox = run_detector_face.inference(run_img)
    run_embedding2 = run_verificator_face.inference(run_face).squeeze()

    from Liveness.functions import verify_embeddings

    run_is_verified, run_distance = verify_embeddings(run_embedding, run_embedding2)

    print(run_is_verified, run_distance)
