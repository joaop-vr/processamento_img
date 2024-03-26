import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

"""
def calc_hist(inicial, final, save_path):

    # Carrega as imagens
    img = cv2.imread(inicial)
    imgf = cv2.imread(final)

    # Calcular o histograma de cada canal de cor da imagem colorida
    hist_blue = cv2.calcHist([img], [0], None, [256], [0,256])
    hist_green = cv2.calcHist([img], [1], None, [256], [0,256])
    hist_red = cv2.calcHist([img], [2], None, [256], [0,256])

    hist_blue_final = cv2.calcHist([imgf], [0], None, [256], [0,256])
    hist_green_final = cv2.calcHist([imgf], [1], None, [256], [0,256])
    hist_red_final = cv2.calcHist([imgf], [2], None, [256], [0,256])

    # Normalizar os histogramas
    hist_blue = cv2.normalize(hist_blue, hist_blue, 0, 1, cv2.NORM_MINMAX)
    hist_green = cv2.normalize(hist_green, hist_green, 0, 1, cv2.NORM_MINMAX)
    hist_red = cv2.normalize(hist_red, hist_red, 0, 1, cv2.NORM_MINMAX)

    hist_blue_final = cv2.normalize(hist_blue_final, hist_blue_final, 0, 1, cv2.NORM_MINMAX)
    hist_green_final = cv2.normalize(hist_green_final, hist_green_final, 0, 1, cv2.NORM_MINMAX)
    hist_red_final = cv2.normalize(hist_red_final, hist_red_final, 0, 1, cv2.NORM_MINMAX)

    # Plotar os histogramas de cada canal de cor da imagem colorida
    plt.figure()
    plt.plot(hist_blue, color='blue', label='Canal Azul Inicial')
    plt.plot(hist_green, color='green', label='Canal Verde Inicial')
    plt.plot(hist_red, color='red', label='Canal Vermelho Inicial')
    plt.plot(hist_blue_final, color='blue', linestyle='--', label='Canal Azul Final')
    plt.plot(hist_green_final, color='green', linestyle='--', label='Canal Verde Final')
    plt.plot(hist_red_final, color='red', linestyle='--', label='Canal Vermelho Final')
    plt.title("Comparação")
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência Normalizada")
    plt.legend()

    # Salva o gráfico como uma imagem
    plt.savefig(save_path)

    plt.show()
"""
def segment_forest(input_image_path, output_image_path):
    # Carrega a imagem de entrada
    img = cv2.imread(input_image_path)

    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define os limites inferior e superior para a cor verde no espaço de cores HSV
    lower_green = np.array([36, 25, 25])  # Valores aproximados para o verde no espaço HSV
    upper_green = np.array([86, 255, 255]) # Valores aproximados para o verde no espaço HSV

    # Cria a máscara para a cor verde
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # kernel para usar na erosao e dilatacao
    kernel = np.ones((3,3), np.uint8)

    # Combinação de abertura e fechamento
    mask_combinada = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_combinada = cv2.morphologyEx(mask_combinada, cv2.MORPH_CLOSE, kernel)


    # Aplica a máscara para realçar a área de interesse (floresta)
    segmented_image = cv2.bitwise_and(img, img, mask=mask_combinada)

    # Salva a imagem segmentada
    cv2.imwrite(output_image_path, segmented_image)
"""
    file_name = os.path.basename(output_image_path)
    name, ext = os.path.splitext(file_name)
    graph_save_path = name + "_graph" + ".png"

    calc_hist(input_image_path, output_image_path, graph_save_path)"""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 floresta.py input_image_path output_image_path")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        segment_forest(input_image_path, output_image_path)
