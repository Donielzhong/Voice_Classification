from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import os
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Set the Matplotlib backend to 'Agg' to avoid GUI issues
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PLOTS_FOLDER'] = 'static/plots/'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# Load your pre-trained model
model = load_model('model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Classify gender
            features = extract_feature(file_path, mel=True).reshape(1, -1)
            male_prob = model.predict(features)[0][0]
            female_prob = 1 - male_prob
            sex = "male" if male_prob > female_prob else "female"

            # Generate plots
            plot_filename_waveform = filename.rsplit('.', 1)[0] + '_waveform.png'
            plot_filepath_waveform = os.path.join(app.config['PLOTS_FOLDER'], plot_filename_waveform)
            plot_filename_spectrogram = filename.rsplit('.', 1)[0] + '_spectrogram.png'
            plot_filepath_spectrogram = os.path.join(app.config['PLOTS_FOLDER'], plot_filename_spectrogram)
            generate_waveform_plot(file_path, plot_filepath_waveform)
            generate_spectrogram_plot(file_path, plot_filepath_spectrogram)

            return render_template('result.html', filename=filename, 
                                   audio_path=url_for('uploaded_file', filename=filename), 
                                   plot_path_waveform=url_for('plot_file', filename=plot_filename_waveform), 
                                   plot_path_spectrogram=url_for('plot_file', filename=plot_filepath_spectrogram),
                                   sex=sex, male_prob=male_prob, female_prob=female_prob)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/<filename>')
def plot_file(filename):
    return send_from_directory(app.config['PLOTS_FOLDER'], filename)

def generate_waveform_plot(audio_file, plot_filepath):
    y, sr = librosa.load(audio_file, sr=None)
    
    plt.figure(figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(plot_filepath)
    plt.close()

def generate_spectrogram_plot(audio_file, plot_filepath):
    y, sr = librosa.load(audio_file, sr=None)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Hz)')
    plt.savefig(plot_filepath)
    plt.close()

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mel = np.mean(mel_spectrogram.T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    return result

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
    app.run(debug=True)
