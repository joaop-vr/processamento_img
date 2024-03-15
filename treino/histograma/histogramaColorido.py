"""
Exercício para praticar o conceito de histograma. 
Obtem-se o histograma dos níveis de cinza e das cores de uma mesma imagem.
"""

import cv2
from matplotlib import pyplot as plt

# Carregar uma imagem em escala de cinza
img_gray = cv2.imread("Foto.jpeg", 0)

# Carregar uma imagem colorida
img_color = cv2.imread("Foto.jpeg")

# Calcular o histograma da imagem em escala de cinza
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0,256])

# Calcular o histograma de cada canal de cor da imagem colorida
hist_blue = cv2.calcHist([img_color], [0], None, [256], [0,256])
hist_green = cv2.calcHist([img_color], [1], None, [256], [0,256])
hist_red = cv2.calcHist([img_color], [2], None, [256], [0,256])

# Plotar o histograma da imagem em escala de cinza
plt.figure()
plt.plot(hist_gray, color='black')
plt.title("Histograma da Imagem em Escala de Cinza")
plt.xlabel("Nível de Cinza")
plt.ylabel("Frequência")

# Plotar os histogramas de cada canal de cor da imagem colorida
plt.figure()
plt.plot(hist_blue, color='blue', label='Canal Azul')
plt.plot(hist_green, color='green', label='Canal Verde')
plt.plot(hist_red, color='red', label='Canal Vermelho')
plt.title("Histograma de Cada Canal de Cor da Imagem Colorida")
plt.xlabel("Intensidade")
plt.ylabel("Frequência")
plt.legend()

plt.show()
