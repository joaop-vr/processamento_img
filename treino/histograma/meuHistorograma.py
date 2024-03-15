import cv2
from matplotlib import pyplot as plt

# Carregar uma imagem em escala de cinza
img_gray = cv2.imread("Foto.jpeg", 0)

# Declaração de uma lista para o histograma
histograma = [0] * 256

# Obtém as dimensões da imagem
altura, largura = img_gray.shape

# Calcular o histograma da imagem em escala de cinza
for i in range(altura):
    for j in range(largura):
        valor_pixel = img_gray[i][j]
        histograma[valor_pixel] += 1

# Plotar o histograma da imagem em escala de cinza
plt.figure()
plt.plot(histograma, color='black')
plt.title("Histograma da Imagem em Escala de Cinza")
plt.xlabel("Intensidade do Pixel")
plt.ylabel("Número de Pixels")
plt.show()
