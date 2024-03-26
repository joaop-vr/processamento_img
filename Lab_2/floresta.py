import sys
import cv2
import numpy as np

def segmentacao(input_image_path, output_image_path):

    # Carrega a imagem de entrada
    img = cv2.imread(input_image_path)

    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define os limites inferior e superior para a cor verde no espaço de cores HSV
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])

    # Cria a máscara para a cor verde
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # kernel para usar na dilatacao
    kernel = np.ones((3,3), np.uint8)

    # Aplicação do método de dilatação
    mask_dilatada = cv2.dilate(mask,kernel,iterations=1)

    # Aplica a máscara para realçar a área de interesse (floresta)
    segmented_image = cv2.bitwise_and(img, img, mask=mask_dilatada)

    # Salva a imagem segmentada
    cv2.imwrite(output_image_path, segmented_image)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 floresta.py input_image_path output_image_path")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        segmentacao(input_image_path, output_image_path)
