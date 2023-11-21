import os
import numpy as np
from PIL import Image

from src.detector import BBoxDetector
from src.enet import FaceEmotionRecognizer, emotion_class_dict_7, get_whole_emotion
from src.utilities import draw_face_simple_emotion

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            # 画像を開く
            with Image.open(img_path) as img:
                # PIL.Image を numpy.ndarray に変換
                img_array = np.array(img)
                images.append(img_array)
        except IOError:
            # ファイルが開けない、または画像ファイルでない場合
            pass
    return images

path_parent = os.getcwd()
path_model = path_parent + "/models/"
path_data = path_parent + "/data/"
print(path_parent)

bbox_detector = BBoxDetector(path_model=path_model, threshold=.6)
face_emotion_recognizer = FaceEmotionRecognizer(path_model=path_model)

images = load_images_from_folder(path_data)

detections, faces, square_faces = bbox_detector.infer(image=images[0])
results = face_emotion_recognizer.infer(image=images[0], faces=square_faces)
# image = draw_face_simple_emotion(frame=images[0], results=results, faces=faces)
# image = Image.fromarray(image)
# image.save(f"{path_data}test-1_result.jpg")

mean_emotion, max_value = get_whole_emotion(results)


print("")
    

