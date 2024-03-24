import cv2
import numpy as np
import sys

# Carrega a imagem
img = cv2.imread(sys.argv[1])

# Converte a imagem para o espaço de cores LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Aumenta os valores positivos no canal 'b' para realçar os tons de verde
enhanced_lab = lab.copy()
enhanced_lab[:, :, 2] = np.clip(1.5 * enhanced_lab[:, :, 2], 0, 255)  # Multiplica os valores do canal 'b' por 1.5

# Converte de volta para BGR para exibir
enhanced_lab_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Exibe a imagem resultante com os tons de verde realçados
cv2.imshow("Enhanced Green (LAB)", enhanced_lab_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
