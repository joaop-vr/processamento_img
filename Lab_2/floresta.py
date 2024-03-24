import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def calc_hist(imagem, versao):

    # Carrega a imagem
    img = cv2.imread(imagem)

    # Calcular o histograma de cada canal de cor da imagem colorida
    hist_blue = cv2.calcHist([img], [0], None, [256], [0,256])
    hist_green = cv2.calcHist([img], [1], None, [256], [0,256])
    hist_red = cv2.calcHist([img], [2], None, [256], [0,256])

    # Plotar os histogramas de cada canal de cor da imagem colorida
    plt.figure()
    plt.plot(hist_blue, color='blue', label='Canal Azul')
    plt.plot(hist_green, color='green', label='Canal Verde')
    plt.plot(hist_red, color='red', label='Canal Vermelho')
    plt.title("Imagem:" + versao)
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência")
    plt.legend()

    plt.show()

def segment_forest(input_image_path, output_image_path):
    # Carrega a imagem de entrada
    img = cv2.imread(input_image_path)

    # Teste
    calc_hist(input_image_path, "Original")

    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define os limites inferior e superior para a cor verde no espaço de cores HSV
    lower_green = np.array([36, 25, 25])  # Valores aproximados para o verde no espaço HSV
    upper_green = np.array([86, 255, 255]) # Valores aproximados para o verde no espaço HSV

    # Cria a máscara para a cor verde
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Aplica a máscara para realçar a área de interesse (floresta)
    segmented_image = cv2.bitwise_and(img, img, mask=mask)

    # Salva a imagem segmentada
    cv2.imwrite(output_image_path, segmented_image)

    calc_hist(output_image_path, "Final")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 floresta.py input_image_path output_image_path")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        segment_forest(input_image_path, output_image_path)
