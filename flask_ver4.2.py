from flask import Flask, render_template, request, send_file
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'E:\\2024 SONG\\MachineLearning\\discriminator\\flask\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('E:\\2024 SONG\\MachineLearning\\discriminator\\flask\\model_ver2.0.keras')

def preprocess_image(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((100, 100))
    image_arrayed = np.array(image_resized) / 255.0
    preprocessed_image = Image.fromarray((image_arrayed * 255).astype(np.uint8))
    preprocessed_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_' + os.path.basename(image_path))
    preprocessed_image.save(preprocessed_filename)
    return preprocessed_filename

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image_preprocessed = np.expand_dims(image, axis=0)
    return image_preprocessed

def predict_gender(image_path):
    image_preprocessed = load_and_preprocess_image(image_path)
    prediction = model.predict(image_preprocessed)
    gender = '여성' if prediction[0][0] > 0.5 else '남성'
    return gender

def predict_probability(image_path):
    image_preprocessed = load_and_preprocess_image(image_path)
    prediction = model.predict(image_preprocessed)
    probability = prediction[0][0] * 100 if prediction[0][0] * 100 > 50 else 100 - prediction[0][0] * 100
    return probability

@app.route('/')
def index():
    return render_template('discriminator_ver4.1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return '파일을 업로드해주세요.', 400

    file = request.files['photo']
    if file.filename == '':
        return '파일을 선택해주세요.', 400

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        preprocess_image(filename)
        preprocessed_image_file = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_' + file.filename)
        gender = predict_gender(preprocessed_image_file)
        probability = predict_probability(preprocessed_image_file)
        return f'사진의 성별은 {probability}의 확률로 {gender}입니다.'
    
@app.route('/image/<filename>')
def show_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_' + filename), mimetype='image/jpg')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)