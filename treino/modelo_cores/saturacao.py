import cv2
import numpy as np
import sys

def adjust_saturation(image, saturation_factor):
    # Convertendo a imagem para o espaço de cores HSV e convertendo para float32
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")

    # Separando os canais de cores HSV
    h, s, v = cv2.split(hsv_image)

    # Ajustando a saturação multiplicando pelo fator de ajuste
    s *= saturation_factor

    # Aplicando um limite para os valores de saturação
    s = np.clip(s, 0, 255)

    # Mesclando novamente os canais HSV
    hsv_adjusted = cv2.merge([h, s, v])

    # Convertendo de volta para o espaço de cores BGR e para uint8
    result_image = cv2.cvtColor(hsv_adjusted.astype("uint8"), cv2.COLOR_HSV2BGR)

    return result_image

# Carregando a imagem
image_path = sys.argv[1]
original_image = cv2.imread(image_path)

# Fator de ajuste de saturação
saturation_factor = 1.5  # Ajuste conforme necessário

# Ajustando a saturação da imagem
result_image = adjust_saturation(original_image, saturation_factor)

# Exibindo a imagem original e a imagem com saturação ajustada
cv2.imshow("Original", original_image)
cv2.imshow("Saturação Ajustada", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
