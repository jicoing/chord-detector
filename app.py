import os
from flask import Flask, render_template, request, redirect, url_for
import librosa
import numpy as np
from scipy.signal import medfilt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}  # Add other allowed audio formats if needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload directory exists
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Utility Functions (same as before)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_chord_templates():
    """Generate chord templates for major and minor triads"""
    templates = []
    chord_names = []
    for root in range(12):
        # Major triad template (root, major third, perfect fifth)
        major = np.zeros(12)
        major[root] = 1
        major[(root + 4) % 12] = 1
        major[(root + 7) % 12] = 1
        major /= np.linalg.norm(major)
        templates.append(major)
        chord_names.append(f"{librosa.midi_to_note(root + 60)[:-1]}:maj")
        
        # Minor triad template (root, minor third, perfect fifth)
        minor = np.zeros(12)
        minor[root] = 1
        minor[(root + 3) % 12] = 1
        minor[(root + 7) % 12] = 1
        minor /= np.linalg.norm(minor)
        templates.append(minor)
        chord_names.append(f"{librosa.midi_to_note(root + 60)[:-1]}:min")
    
    return np.array(templates), chord_names

def format_time(seconds):
    """Convert seconds to minutes and seconds format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def detect_chords(audio_path, hop_length=512, n_fft=2048, min_duration=0.5):
    """Detect chords in an audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, mono=True)
        
        # Apply Harmonic-Percussive Source Separation (HPSS)
        y_harmonic, _ = librosa.effects.hpss(y)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_stft(
            y=y_harmonic, sr=sr, norm=2,
            hop_length=hop_length, n_fft=n_fft
        )
        
        # Generate chord templates
        templates, chord_names = generate_chord_templates()
        
        # Find best matching template for each frame
        chord_ids = np.argmax(templates @ chroma, axis=0)
        
        # Apply median filtering to smooth rapid transitions
        chord_ids = medfilt(chord_ids, kernel_size=5)
        
        # Get frame times
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sr, hop_length=hop_length, n_fft=n_fft
        )
        
        # Merge consecutive identical chords with a minimum duration
        current_chord = None
        start_time = 0
        chords = []
        for time, chord_id in zip(times, chord_ids):
            chord = chord_names[chord_id]
            if chord != current_chord:
                if current_chord is not None and (time - start_time) >= min_duration:
                    chords.append((current_chord, format_time(start_time), format_time(time)))
                current_chord = chord
                start_time = time
        
        # Add last chord
        if current_chord is not None and (times[-1] - start_time) >= min_duration:
            chords.append((current_chord, format_time(start_time), format_time(times[-1])))
        
        return chords
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']

        # If the user does not select a file, submit an empty part without a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the audio file
            detected_chords = detect_chords(file_path)
            return render_template('results.html', chords=detected_chords, filename=filename)
        else:
            return render_template('index.html', error='Invalid file type. Allowed types: mp3, wav, flac')

    return render_template('index.html', error=None)

@app.route('/results/<filename>')
def results(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    detected_chords = detect_chords(file_path)
    return render_template('results.html', chords=detected_chords, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
