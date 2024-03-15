"""
Exercício para praticar: Escreva um programa em python que

    1 - Amplie a imagem em 2x. 
    2 - Reduza a imagem pela metade. Nesse caso use a média dos pixels.
    3 - Você não pode usar a função resize.
"""

import cv2 as cv
import numpy as np

def amplify_image(image):
    # Obter dimensões da imagem original
    height, width = image.shape[:2]

    # Criar uma nova imagem ampliada com o dobro da altura e largura
    amplified_image = np.zeros((height*2, width*2, 3), dtype=np.uint8)

    # Ampliar a imagem replicando os pixels
    for i in range(height):
        for j in range(width):
            amplified_image[i*2:i*2+2, j*2:j*2+2] = image[i, j]

    return amplified_image

def reduce_image(image):
    # Obter dimensões da imagem original
    height, width = image.shape[:2]

    # Criar uma nova imagem reduzida pela metade
    reduced_image = np.zeros((height//2, width//2, 3), dtype=np.uint8)

    # Reduzir a imagem calculando a média ou mediana dos pixels
    for i in range(0, height, 2):
        for j in range(0, width, 2):
                reduced_image[i//2, j//2] = np.mean(image[i:i+2, j:j+2], axis=(0, 1)).astype(np.uint8)

    return reduced_image

# Carregar a imagem
image = cv.imread('white.png')

# Ampliar a imagem em 2x
amplified_image = amplify_image(image)
cv.imwrite('white_amplified.png', amplified_image)

# Reduzir a imagem pela metade usando média dos pixels
reduced_image = reduce_image(image)
cv.imwrite('white_reduced.png', reduced_image)


cv.waitKey(0)
cv.destroyAllWindows()
