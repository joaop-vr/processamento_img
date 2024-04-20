import cv2
import numpy as np
import sys

# Verificar se o caminho da imagem foi fornecido como argumento
if len(sys.argv) < 2:
    print("Por favor, forneça o caminho da imagem como argumento.")
    sys.exit(1)

# Carregar a imagem
image_path = sys.argv[1]
image = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem.")
    sys.exit(1)

# Converter a imagem para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Definir um kernel retangular vertical
kernel = np.ones((5, 1), np.uint8)

# Aplicar a operação de dilatação para preencher o risco vertical
dilated = cv2.dilate(gray, kernel, iterations=1)

# Subtrair a imagem dilatada da imagem original para remover o risco vertical
removed_vertical_line = cv2.subtract(dilated, gray)

# Mostrar a imagem resultante
cv2.imwrite('Imagem_s_risco.png', removed_vertical_line)
