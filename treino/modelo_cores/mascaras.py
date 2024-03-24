import sys
import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Valores HSV para a cor verde
green = np.uint8([[[0, 255, 0]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
H = hsv_green[0][0][0]

# Definindo os limites inferior e superior para o componente de matiz (H)
lower_green = np.array([H-10, 50, 50])
upper_green = np.array([H+10, 255, 255])

# Criando a máscara para a cor verde
mask = cv2.inRange(hsv, lower_green, upper_green)

# Aplicando a máscara para realçar a cor verde
imask = mask > 0
green_highlighted = img.copy()
green_highlighted[imask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[imask]
green_highlighted = np.clip(green_highlighted, 0, 255)

# Aplicando a máscara complementar
no_green = img.copy()
no_green = cv2.bitwise_and(no_green, no_green, mask=np.bitwise_not(imask).astype(np.uint8))

# Criando uma imagem apenas com as bolas azuis
blue_balls_image = np.zeros_like(img)
blue_balls_image[imask] = green_highlighted[imask]

# Mostrando as imagens
cv2.imshow("img_realcada", green_highlighted)
cv2.imshow("img_complementar", no_green)
cv2.imshow("img_final", blue_balls_image)
cv2.waitKey()
