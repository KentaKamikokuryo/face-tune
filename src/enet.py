import os, urllib
from hsemotion_onnx.facial_emotions import *

emotion_class_dict_7 = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
emotion_class_dict_8 = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Neutral', 6: 'Sad', 7: 'Surprise'}

def _get_model_path(path_model, model_name):
    model_file = model_name + '.onnx'
    cache_dir = os.path.join(path_model, '.hsemotion')
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        url = 'https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/' + model_file + '?raw=true'
        print('Downloading', model_name, 'from', url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


class ExtendedHSEmotionRecognizer(HSEmotionRecognizer):
    # supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, path_model, model_name='enet_b2_7'):
        self.is_mtl = '_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class = emotion_class_dict_7
        else:
            self.idx_to_class = emotion_class_dict_8

        self.img_size = 224 if '_b0_' in model_name else 260

        path = _get_model_path(path_model=path_model, model_name=model_name)
        self.ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        

class FaceEmotionRecognizer():

    def __init__(self, path_model):

        self.path_model = path_model
        self.model_name = "HSEmotion"

        self.emo_model = ExtendedHSEmotionRecognizer(path_model=self.path_model, model_name="enet_b2_7")

    def detect_emotions(self, face_image):
        # the face image should be rgb

        emo_label, probas = self.emo_model.predict_emotions(face_img=face_image, logits=False)
        probas = list(probas)
        emo_proba = max(probas)

        proba_dict = {}
        for emo_idx, emo_name in self.emo_model.idx_to_class.items():
            proba_dict[emo_name] = probas[emo_idx]

        return emo_label, emo_proba, proba_dict, probas

    def infer(self, image, faces):

        results = []

        for i, face in enumerate(faces):

            xmin = int(face[0])
            ymin = int(face[1])
            xmax = int(face[2])
            ymax = int(face[3])

            img = image[ymin:ymax, xmin:xmax]
            h, w = img.shape[:2]

            if h == 0 or w == 0:
                continue

            emo_label, emo_proba, proba_dict, emo_probas = self.detect_emotions(face_image=img)

            results.append(
                {
                    "emo_label": emo_label,
                    "emo_proba": emo_proba,
                    "emo_probas": emo_probas,
                    "proba_dict": proba_dict,
                }
            )

        return results

    def infer_from_images(self, images):

        results = []

        for i, img in enumerate(images):

            h, w = img.shape[:2]

            if h == 0 or w == 0:
                continue

            emo_label, emo_proba, proba_dict, emo_probas = self.detect_emotions(face_image=img)

            results.append(
                {
                    "emo_label": emo_label,
                    "emo_proba": emo_proba,
                    "emo_probas": emo_probas,
                    "proba_dict": proba_dict,
                }
            )

        return results


def get_whole_emotion(results: dict):
    
    result_emotion = []

    for result in results:
        emo_probas = result["emo_probas"]
        result_emotion.append(emo_probas)
        
    result_emotion_array = np.array(result_emotion)
    mean_emotion_list = np.mean(result_emotion_array, axis=0).tolist()

    max_index = mean_emotion_list.index(max(mean_emotion_list))
    mean_emotion = emotion_class_dict_7[max_index]
    max_value = mean_emotion_list[max_index]
    
    return mean_emotion, max_value