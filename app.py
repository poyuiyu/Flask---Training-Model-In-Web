from flask import Flask, render_template, flash, request, redirect, send_file
from werkzeug.utils import secure_filename
import os 
import urllib.request
from random import random 
import io 
import base64 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns 
import numpy as np 
import pandas as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

UPLOAD_FOLDER = 'D:\DATA SEMESTER 7\Latihan Program Skripsiii\Train In Web Test\static'

app.secret_key = "Kaputpet34242432"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'xlsx'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


fig, ax = plt.subplots(figsize=(4,4))
ax = sns.set_style(style=('darkgrid'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('Tidak Ada File')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File Berhasil Diupload')
        return redirect('/')

@app.route('/visualisasi')
def visualisasi():
    import pandas as pd
    data = pd.read_csv('D:\DATA SEMESTER 7\Latihan Program Skripsiii\Train In Web Test\static\dataRegressi.csv')
    XX = data['X'].values
    X = XX.reshape(-1, 1)
    YY = data['Y'].values
    Y = YY.reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X,Y)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    plt.scatter(X_train, y_train, color='blue')
    plt.plot(X_train, model.predict(X_train), color='red')
    plt.title('Regressi Linear Grafik')
    plt.xlabel('Nilai X')
    plt.ylabel('Nilai y')
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')




if __name__ == '__main__':
    app.run(debug=True)