import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Ruta del archivo CSV
file_path = r"/Users/alvaroinfante/Downloads/cleaned_dataset.csv"

# Cargar los datos
data = pd.read_csv(file_path)

# Calcular la matriz de correlación
correlation_matrix = data.corr()

# Crear un grafo
G = nx.Graph()

# Añadir nodos y aristas basados en la correlación
for i, column in enumerate(correlation_matrix.columns):
    for j in range(i+1, len(correlation_matrix.columns)):
        weight = abs(correlation_matrix.iloc[i, j])
        if weight > 0.5:  # Umbral de correlación
            G.add_edge(column, correlation_matrix.columns[j], weight=weight)

# Dibujar la red
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue")
plt.title("Red de Correlación entre Variables")
plt.show()

# Analizar métricas de la red
degree_centrality = nx.degree_centrality(G)
print("Centralidad de Grado:", degree_centrality)
