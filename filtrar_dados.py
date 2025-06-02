import pandas as pd  # Importa a biblioteca pandas para manipulação de dados

# Caminho para o arquivo CSV contendo os metadados do dataset ESC-50
csv_path = r"C:\Users\Guilherme\OneDrive - Fiap-Faculdade de Informática e Administração Paulista\Global Solution\Computer Organization and Structure\ESC-50-master\meta\esc50.csv"  # <-- Use o SEU caminho aqui

# Lê o arquivo CSV e armazena em um DataFrame do pandas
df = pd.read_csv(csv_path)
print(df.head())  # Exibe as primeiras linhas do DataFrame para inspeção

# Define as categorias de interesse para filtrar os dados
categories_of_interest = ['thunderstorm', 'wind', 'rain']

# Filtra o DataFrame para manter apenas as linhas com as categorias de interesse
filtered_df = df[df['category'].isin(categories_of_interest)]

# Exibe as colunas 'filename' e 'category' do DataFrame filtrado
print(filtered_df[['filename', 'category']])
