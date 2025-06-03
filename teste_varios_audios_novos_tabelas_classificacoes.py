import os
import sys
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Função para extrair MFCCs de um arquivo de áudio
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Função para plotar matriz de confusão de forma legível
def plot_matriz_confusao(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(18, 14))
    sns.set(font_scale=0.9)

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 7}
    )

    plt.title("Matriz de Confusão", fontsize=18)
    plt.xlabel("Previsto", fontsize=14)
    plt.ylabel("Real", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

def main():
    modelo_path = "random_forest_model.joblib"
    pasta_audio = r"C:\Users\Guilherme\OneDrive - Fiap-Faculdade de Informática e Administração Paulista\Global Solution\Computer Organization and Structure\ESC-50-master\audio" # <-- Use o SEU caminho aqui
    csv_path = r"C:\Users\Guilherme\OneDrive - Fiap-Faculdade de Informática e Administração Paulista\Global Solution\Computer Organization and Structure\ESC-50-master\meta\esc50.csv" # <-- Use o SEU caminho aqui

    # Carregar o modelo treinado
    model = joblib.load(modelo_path)

    # Ler CSV
    df_csv = pd.read_csv(csv_path)
    df_csv = df_csv[['filename', 'category']]

    resultados = []

    for index, row in df_csv.iterrows():
        file_path = os.path.join(pasta_audio, row['filename'])

        if os.path.exists(file_path):
            try:
                features = extract_features(file_path)
                prediction = model.predict([features])[0]

                resultados.append({
                    'arquivo': row['filename'],
                    'real': row['category'],
                    'predicao': prediction
                })

            except Exception as e:
                print(f"Erro ao processar {row['filename']}: {e}")
        else:
            print(f"Arquivo não encontrado: {file_path}")

    # Salvar resultados em CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados_predicoes.csv", index=False)
    print("Predições salvas em 'resultados_predicoes.csv'")

    # Plotar matriz de confusão
    if not df_resultados.empty:
        y_true = df_resultados['real']
        y_pred = df_resultados['predicao']
        labels_ordenadas = sorted(y_true.unique())
        plot_matriz_confusao(y_true, y_pred, labels_ordenadas)

if __name__ == "__main__":
    main()
