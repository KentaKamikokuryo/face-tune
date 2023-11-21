import cv2
import numpy as np


def draw_face_simple_emotion(frame, results, faces):
    """
    Params:
    ---------
    frame : np.ndarray

    results : list of dict.keys('xmin', 'xmax', 'ymin', 'ymax', 'emo_label', 'emo_proba')

    Returns:
    ---------
    frame : np.ndarray
    """
    for i, result in enumerate(results):
        
        face = faces[i]

        xmin = face[0]
        xmax = face[2]
        ymin = face[1]
        ymax = face[3]
        emo_label = result["emo_label"]
        emo_proba = result["emo_proba"]

        label_size, base_line = cv2.getTextSize(
            f"{emo_label}: 000", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        # draw face
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (179, 255, 179), 2)

        cv2.rectangle(
            frame,
            (xmax, ymin + 1 - label_size[1]),
            (xmax + label_size[0], ymin + 1 + base_line),
            (223, 128, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            f"{emo_label} {int(emo_proba * 100)}",
            (xmax, ymin + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

    return frame