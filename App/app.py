from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)


def model_load():
    loaded_model = tf.keras.models.load_model('potato_disease_98.h5')
    return loaded_model

model = model_load()

def predict_img(image_path, loaded_model, class_names= ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    
    test_image = tf.keras.preprocessing.image.load_img(image_path, target_size= (256,256))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    pred_value = loaded_model.predict(test_image)
    return class_names[np.argmax(pred_value)]


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods = ['POST'])
def predict():

    if request.method == 'POST':

        file = request.files['file']
        print(file.filename)

        file.save(file.filename)

        data = predict_img(file.filename, model)

        if data == 'Potato___Early_blight':
            data = 'Has Early Blight Disease'
        elif data== 'Potato___Late_blight':
            data = 'Has Late Blight Disease'
        elif data == 'Potato___healthy':
            data = 'Is of a Healthy Potato plant'
        else:
            data = 'Leaf not recognized'

        os.remove(file.filename)
        return render_template('predict.html', value = data)
        

if __name__ == '__main__':
    app.run(debug=True)