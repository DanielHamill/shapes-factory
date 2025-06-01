import requests
import base64
url = 'http://127.0.0.1:8000/predict'

with open('./images/drawing.png', "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

body = {
    "image_b64": encoded_string,
}
requests.post(url, json=body)