from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from keras.models import load_model
import cv2 as cv
import imutils
from imutils.contours import sort_contours
import numpy
import os

model = load_model('data_augmentation_model.h5')

app = Flask(__name__)
app.secret_key = os.urandom(24)

upload_folder = r'C:\Users\yuvim\OneDrive\Desktop\Flask_development\uploads'
app.config['UPLOAD_FOLDER'] = upload_folder


@app.route('/')
def success():
    return render_template('input.html')


def prediction(filename):
    image = cv.imread(os.path.join(upload_folder, filename))
    image = cv.resize(image, (800, 800))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edged = cv.Canny(gray, 30, 150)
    contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    for c in contours:
        (x, y, w, h) = cv.boundingRect(c)
        if 10 <= w <= 500 and 10 <= h <= 500:
            roi = gray[y:y + h, x:x + w]
            thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            (th, tw) = thresh.shape
            if tw > th:
                thresh = imutils.resize(thresh, width=32)
            if th > tw:
                thresh = imutils.resize(thresh, height=32)
            (th, tw) = thresh.shape
            dx = int(max(0, 32 - tw) / 2.0)
            dy = int(max(0, 32 - th) / 2.0)
            padded = cv.copyMakeBorder(thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv.BORDER_CONSTANT,
                                       value=(0, 0, 0))
            padded = cv.resize(padded, (32, 128))
            padded = numpy.array(padded)
            padded = numpy.expand_dims(padded, axis=0)
            pred = model.predict(padded) * 100
            max_value = max(pred[0])
            output_array = [1 if i >= max_value else 0 for i in pred[0]]
            z = output_array.index(1)
            label = labels[z]
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(image, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return image


@app.route('/submitted_image', methods=['POST', 'GET'])
def submitted_image():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            flash('No file selected')
            return redirect(url_for('success'))
        else:
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predicted_img = prediction(filename)
            predicted_img_name = os.path.splitext(filename)[0] + '_predicted_img.jpeg'
            cv.imwrite(os.path.join(upload_folder, predicted_img_name), predicted_img)
            return redirect(url_for('uploaded_file', final_image_path=predicted_img_name, initial_image_path=filename))


@app.route('/output/<final_image_path>/<initial_image_path>')
def uploaded_file(final_image_path, initial_image_path):
    return render_template('output.html', final_image_dir=final_image_path, initial_image_dir=initial_image_path)


@app.route('/upload/<filePath>')
def send_file(filePath):
    return send_from_directory(upload_folder, filePath)


@app.route('/download/<file_path>')
def download_file(file_path):
    return send_from_directory(upload_folder, file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
