from flask import Flask, request, abort
from src.detector import BBoxDetector
from src.enet import FaceEmotionRecognizer, get_whole_emotion
from src.utilities import draw_face_simple_emotion
from src.spotify import extract_the_most_played_song_in_each_country, FILENAME
from src.recommend import get_song_information
from dotenv import load_dotenv

# os内のenvironmentを扱うライブラリ
import os, requests, json, time
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,ImageMessage,
    ImageSendMessage,
    )

load_dotenv()

# BBOX検知モデル ＋ 感情認識モデル
bbox_detector = BBoxDetector(path_model=os.environ["DEEPFACE_HOME"], threshold=.6)
face_emotion_recognizer = FaceEmotionRecognizer(path_model=os.environ["DEEPFACE_HOME"])

# 各国の再生回数画一番多い曲（毎週更新）を取得し保存
extract_the_most_played_song_in_each_country()


#環境変数取得
YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

header = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + YOUR_CHANNEL_ACCESS_TOKEN
}


app = Flask(__name__)



@app.route("/")
def hello_world():
    return "hello!"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = "顔画像を送ってください．"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=text))
    

@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):

    message_id = event.message.id
    print(message_id)
    
    try: 
        message_content = line_bot_api.get_message_content(message_id)
        image_array = _get_image_from_line(message_content)
        _, faces, square_faces = bbox_detector.infer(image=image_array)
        results = face_emotion_recognizer.infer(image=image_array, faces=square_faces)
        image = draw_face_simple_emotion(frame=image_array, results=results, faces=faces)
        image = Image.fromarray(image)
        image.save(f"./data/output_{message_id}.jpg")
        
        
        # with open(Path(f"static/images/{message_id}.jpg").absolute(), "wb") as f:
        #     # バイナリを1024バイトずつ書き込む
        #     for chunk in message_content.iter_content():
        #         f.write(chunk)

        # # Save the processed image to a cloud storage or server and get the URL
        # # This part is pseudo-code and needs to be replaced with actual implementation
        # image_url = _save_image_to_cloud(image)  # Implement this function
        # preview_url = image_url  # You can use the same URL for preview, or create a different one

        # # Send the image as a reply
        # line_bot_api.reply_message(
        #     event.reply_token,
        #     ImageSendMessage(
        #         original_content_url=image_url,
        #         preview_image_url=preview_url
        #     )
        # )
        
        # time.sleep(1)
        
        mean_emotion, prob = get_whole_emotion(results)
        artist, title, uri = get_song_information(mean_emotion, prob, FILENAME)
        
        url = _convert_spotify_uri_to_url(uri)
        
        kg = "\n"
        text_to_send = f"画像全体の感情は{str(mean_emotion)}で確率{str(int(prob * 100))}%なので，以下の曲がおすすめです．{kg}アーティスト: {artist} {kg}曲名: {title} {kg}URI: {url}"
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=text_to_send)
        )
        
    except Exception as e:
        
        # Log the exception for debugging
        print(f"An error occurred: {e}")

        # Send a text message to the user notifying them of the error
        error_message = "認識できる顔がありません．"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=error_message)
        )
        
def _convert_spotify_uri_to_url(spotify_uri):
    # Split the URI by ':' and check if it's a valid Spotify track URI
    parts = spotify_uri.split(':')
    if len(parts) != 3 or parts[0] != 'spotify' or parts[1] != 'track':
        raise ValueError("Invalid Spotify track URI")

    # Construct the URL using the track ID
    track_id = parts[2]
    url = f"http://open.spotify.com/track/{track_id}"
    return url
    

def _get_image_from_line(message_content) -> np.ndarray:
    

    # line_url = 'https://api.line.me/v2/bot/message/' + id + '/content/'
    # result = requests.get(line_url, headers=header)
    # print(result)
    im = Image.open(BytesIO(message_content.content))
    im_array = np.array(im)
    
    return im_array

# Helper function to save image to cloud storage (this is pseudo-code)
def _save_image_to_cloud(image):
    # Convert PIL Image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Upload bytes to cloud storage and return the URL
    # This should be replaced with actual code to upload the image
    # For example, using AWS S3, Google Cloud Storage, or other services
    uploaded_url = _upload_to_cloud_service(img_byte_arr)  # Implement this function
    return uploaded_url

# Another helper function for uploading to a cloud service (pseudo-code)
def _upload_to_cloud_service(image_bytes):
    # Code to upload image bytes to a cloud service and return the URL
    # This is highly dependent on the cloud service you are using
    return "URL of the uploaded image"




if __name__ == "__main__":
    app.run()
