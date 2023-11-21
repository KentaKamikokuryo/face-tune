import copy
import numpy as np
from retinaface import RetinaFace


def _convert_bbox2sbbox(xmin, ymin, xmax, ymax):

    # enlarge the bbox a little and do a square crop
    HCenter = (ymin + ymax) / 2
    WCenter = (xmin + xmax) / 2

    len_h = ymax - ymin
    len_w = xmax - xmin

    if len_h > len_w:
        side_len = len_h
    else:
        side_len = len_w

    margin = side_len * 1.0 // 2

    x1 = int(WCenter - margin)
    y1 = int(HCenter - margin)
    x2 = int(WCenter + margin)
    y2 = int(HCenter + margin)

    return x1, y1, x2, y2


class BBoxDetector():

    def __init__(self, path_model: str = "", threshold: float = .6):

        self.path_model = path_model
        self.threshold = threshold

    def infer(self, image: np.ndarray):

        img = copy.deepcopy(image)
        infer_result = RetinaFace.detect_faces(img, threshold = 0.5)

        detections = self._get_detections(infer_result=infer_result, image=img)
        faces = self._get_faces(detections=detections)
        square_faces = self._get_square_faces(detections=detections)

        return detections, faces, square_faces

    def _get_detections(self, infer_result, image):

        detections = []
        height, width = image.shape[:2]

        if len(infer_result) > 0:
            
            for key in infer_result:
                
                detection = infer_result[key]
                conf = detection["score"]
                
                if (conf > self.threshold):
                    
                    bbox = detection["facial_area"]

                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = bbox[2]
                    y2 = bbox[3]
                    
                    detections.append([x1, y1, x2, y2, conf])

        return detections

    def _get_faces(self, detections):

        persons = []
        self._conf = []
        if (len(detections) > 0):
            for detection in detections:
                conf = detection[4]
                if (conf > self.threshold):
                    x1 = int(detection[0])
                    y1 = int(detection[1])
                    x2 = int(detection[2])
                    y2 = int(detection[3])
                    persons.append([x1, y1, x2, y2])
                    self._conf.append(float(conf))

        return persons

    def _get_square_faces(self, detections):

        persons = []
        self._conf = []

        if len(detections) > 0:

            for detection in detections:
                conf = detection[4]

                if conf > self.threshold:

                    xmin = int(detection[0])
                    ymin = int(detection[1])
                    xmax = int(detection[2])
                    ymax = int(detection[3])

                    x1, y1, x2, y2 = _convert_bbox2sbbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                    persons.append([x1, y1, x2, y2])
                    self._conf.append(float(conf))

        return persons