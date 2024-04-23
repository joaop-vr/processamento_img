import numpy as np

def find_local_minima(hist):
    minima_indices = []
    
    for i in range(1, len(hist) - 1):
        if hist[i] < hist[i-1] and hist[i] < hist[i+1]:
            minima_indices.append(i)
            
    # Ordenar os índices dos mínimos locais
    minima_indices.sort(key=lambda x: hist[x])
            
    return minima_indices

# Exemplo de um histograma (lista)
histograma = [5, 3, 2, 4, 5, 2, 1, 3, 6, 4, 2]

# Encontrar os mínimos locais e ordená-los
indices_minimos_ordenados = find_local_minima(histograma)

print("Indices dos mínimos locais ordenados:", indices_minimos_ordenados)
