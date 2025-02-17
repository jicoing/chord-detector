import os
import librosa
import numpy as np
from scipy.signal import medfilt
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import tempfile
from pydub import AudioSegment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format='wav')

def generate_chord_templates():
    """Generate chord templates for major and minor triads"""
    templates = []
    chord_names = []
    for root in range(12):
        # Major triad
        major = np.zeros(12)
        major[root] = 1
        major[(root + 4) % 12] = 1
        major[(root + 7) % 12] = 1
        major /= np.linalg.norm(major)
        templates.append(major)
        chord_names.append(f"{librosa.midi_to_note(root + 60)[:-1]}:maj")
        
        # Minor triad
        minor = np.zeros(12)
        minor[root] = 1
        minor[(root + 3) % 12] = 1
        minor[(root + 7) % 12] = 1
        minor /= np.linalg.norm(minor)
        templates.append(minor)
        chord_names.append(f"{librosa.midi_to_note(root + 60)[:-1]}:min")
    
    return np.array(templates), chord_names

def format_time(seconds):
    """Convert seconds to minutes:seconds format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def detect_chords(audio_path):
    """Detect chords in an audio file"""
    try:
        y, sr = librosa.load(audio_path, mono=True)
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, norm=2)
        templates, chord_names = generate_chord_templates()
        chord_ids = np.argmax(templates @ chroma, axis=0)
        chord_ids = medfilt(chord_ids, kernel_size=5)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
        
        current_chord = None
        start_time = 0
        chords = []
        for time, chord_id in zip(times, chord_ids):
            chord = chord_names[chord_id]
            if chord != current_chord:
                if current_chord is not None and (time - start_time) >= 0.5:
                    chords.append((current_chord, format_time(start_time), format_time(time)))
                current_chord = chord
                start_time = time
        if current_chord is not None and (times[-1] - start_time) >= 0.5:
            chords.append((current_chord, format_time(start_time), format_time(times[-1])))
        return chords
    except Exception as e:
        raise RuntimeError(f"Error processing audio: {str(e)}")

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "File type not allowed", 400
    
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    try:
        # Convert to WAV if necessary
        if not filename.lower().endswith('.wav'):
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted.wav')
            convert_to_wav(temp_path, wav_path)
            os.remove(temp_path)
            temp_path = wav_path
        
        chords = detect_chords(temp_path)
        
        # Create result file
        output = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        for chord, start, end in chords:
            output.write(f"{start}-{end}: {chord}\n")
        output.close()
        
        return send_file(output.name, as_attachment=True, download_name='chords.txt')
    
    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)