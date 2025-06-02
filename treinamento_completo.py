import os
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Função para extrair MFCCs
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def main():
    # Caminhos
    csv_path = r"C:\Users\Guilherme\OneDrive - Fiap-Faculdade de Informática e Administração Paulista\Global Solution\Computer Organization and Structure\ESC-50-master\meta\esc50.csv"
    audio_folder = r"C:\Users\Guilherme\OneDrive - Fiap-Faculdade de Informática e Administração Paulista\Global Solution\Computer Organization and Structure\ESC-50-master\audio"

    df = pd.read_csv(csv_path)
    print(f"Total de amostras: {len(df)}")

    features = []
    labels = []

    for i, row in df.iterrows():
        file_path = os.path.join(audio_folder, row['filename'])
        if os.path.exists(file_path):
            try:
                mfcc = extract_features(file_path)
                features.append(mfcc)
                labels.append(row['category'])
            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")
        else:
            print(f"Arquivo não encontrado: {file_path}")

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Salvar modelo
    joblib.dump(model, "random_forest_model.joblib")
    print("Modelo salvo em random_forest_model.joblib")

    # Avaliação
    y_pred = model.predict(X_test)
    print("=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred))

    # Matriz de Confusão - versão otimizada
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y),
                cmap='rocket', cbar=True, square=True, linewidths=0.4, linecolor='gray', ax=ax)

    ax.set_xlabel("Predito", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)

    ax.tick_params(axis='x', rotation=90, labelsize=6)
    ax.tick_params(axis='y', rotation=0, labelsize=6)

    # Título externo
    plt.suptitle("Matriz de Confusão - 50 Classes do ESC-50 (Random Forest)", fontsize=14, y=1.02)

    # Otimizar layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

if __name__ == "__main__":
    main()
# Este script treina um modelo de classificação de áudio usando o dataset ESC-50.
# Ele extrai características MFCC dos arquivos de áudio, treina um modelo Random Forest e avalia seu desempenho.