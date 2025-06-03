import sys
import librosa
import numpy as np
import joblib

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

if len(sys.argv) != 2:
    print(r"Uso: python teste_audio_novo.py C:\Users\Guilherme\OneDrive - Fiap-Faculdade de Informática e Administração Paulista\Global Solution\Computer Organization and Structure\ESC-50-master\audio\1-19501-A-7.wav") # <-- Use o SEU caminho aqui
    sys.exit(1)

audio_path = sys.argv[1]

# Carregue o modelo salvo
model = joblib.load("random_forest_model.joblib")

# Extraia features do novo áudio
features = extract_features(audio_path)

# Predição
prediction = model.predict([features])
labels = ['rain', 'thunderstorm', 'wind']
print(f"Predição do áudio: {labels[prediction[0]]}")
