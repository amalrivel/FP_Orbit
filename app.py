from flask import Flask, flash, request, redirect, url_for, render_template
import os, shutil
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import pastebin as pb
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_file_in_directory():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def machine_learning(filename):
    image_class_dict = pb.image_class_dict()
    image_class_clasification = pb.image_class_clasification()
    model = tf.keras.models.load_model('model/model.h5')
    img = tf.keras.utils.load_img('static/uploads/' + filename, target_size = (224,224))
    pred_prob = model.predict(tf.expand_dims(tf.keras.utils.img_to_array(img) , axis = 0))
    pred = pred_prob.max()
    result = image_class_dict[np.argmax(list(pred_prob))]
    result_class = image_class_clasification[np.argmax(list(pred_prob))]
    return_to = ""
    if pred < .5:
        return_to = f"Mungkin relief {result} kategori {result_class}, dengan prediksi: {pred:.2f} "
    else:
        return_to = f"Relief {result} kategori {result_class}, dengan prediksi: {pred:.2f}"
    return return_to
     
@app.route('/')
def home():
    delete_file_in_directory()
    return render_template('index.html')

@app.route('/feature')
def feature():
    delete_file_in_directory()
    return render_template('fitur.html')

@app.route('/about')
def about():
    delete_file_in_directory()
    return render_template('tentang.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html'), 404
 
@app.route('/feature', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = machine_learning(file.filename)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('fitur.html', filename=filename, result=result)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(port=5000, debug=True)