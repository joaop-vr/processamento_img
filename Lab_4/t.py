def encontrar_intervalo_otimo(variancias, somatorios):
    melhor_indice = None
    melhor_razao = float('-inf')  # começa com o menor valor possível

    for i in range(len(variancias)):
        if somatorios[i] != 0:  # evita divisão por zero
            razao = variancias[i] / somatorios[i]
            if razao > melhor_razao:
                melhor_razao = razao
                melhor_indice = i

    return melhor_indice

# Exemplo de uso:
variancias = [20, 30, 40, 40, 40, 30, 20, 10, 10, 10]
somatorios = [5, 5, 10, 15, 20, 25, 25, 10, 15, 20]

indice_otimo = encontrar_intervalo_otimo(variancias, somatorios)

print(f"Índice do intervalo ótimo: {indice_otimo}")
print(f"Variancia do intervalo ótimo: {variancias[indice_otimo]}")
print(f"Somatorio do intervalo ótimo: {somatorios[indice_otimo]}")
