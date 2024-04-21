import sys
import os
import cv2
import numpy as np
from math import floor
import matplotlib.pyplot as plt


def define_rois():

    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_1 = (850, 760, 1560, 120)
    roi_11 = (850, 950, 1560, 110)
    roi_12 = (850, 1120, 1560, 120)
    roi_13 = (850, 1310, 1560, 120)
    roi_14 = (850, 1490, 1560, 120)
    roi_15 = (850, 1670, 1560, 90)
    roi_2 = (850, 1770, 570, 200)
    roi_3 = (70, 2070, 600, 250)
    roi_4 = (1330, 2070, 700, 220)
    roi_5 = (900, 2290, 1480, 220)

    # Adiciona os rois à lista global
    rois.append(roi_1)
    rois.append(roi_11)
    rois.append(roi_12)
    rois.append(roi_13)
    rois.append(roi_14)
    rois.append(roi_15)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)

    return rois

"""def invert_colors(img_path):

    # Carregar a imagem em escala de cinza
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print("Erro ao carregar a imagem.")
        sys.exit(1)

    # Inverter as cores
    inverted_image = cv2.bitwise_not(image)

    # Salvar imagem resultante
    parts = img_path.split('/')
    name = "inversa_"+parts[-1]
    print(f"name:{name}")
    cv2.imwrite(name, inverted_image)

    return name"""

def remove_vertical_lines(img_path):
    # Carregar a imagem em escala de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Criar um kernel vertical de espessura 1 e altura da imagem
    kernel = np.ones((img.shape[0] // 50, 1), np.uint8)  # Ajuste o valor conforme necessário
    
    # Aplicar erosão para remover linhas verticais (as linhas serão substituídas por branco)
    inverted_img = cv2.erode(img, kernel, iterations=1)
    
    # Salvar imagem invertida
    parts = img_path.split('/')
    name = "inverted_" + parts[-1]
    cv2.imwrite(name, inverted_img)
    
    return name

def segment_form(img_path, rois):

    # Carregar a imagem
    img = cv2.imread(img_path)

    print(f"arquivo atual::{img_path}")

    for i in range(len(rois)):

        # Variáveis delimitadoras da região de interesse (ROI)
        x, y, w, h = rois[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

        # Recortar região de interesse (ROI)
        roi = img[y:y+h, x:x+w]

        # Salvar o ROI
        filename = "roi"+str(i) +".png"
        cv2.imwrite(filename, roi)

        img_segmented = cv2.imread(filename)
        # Calcular o histograma ao longo do eixo X
        histogram = np.sum(img_segmented, axis=0)

        # Normalizar o histograma
        histogram = histogram / np.max(histogram)
        
        if 6 <= i <= 8:

            # Determina o indice que divide o hsitograma ao meio
            middle_index = np.floor(len(histogram) / 2).astype(int)

            # Extrai o campo central de cada tupla do histograma (índice 1)
            central_values = [tup[1] for tup in histogram[2:-2]]
            
            # Calcula os valores absolutos
            absolute_values = np.abs(central_values)
            
            # Encontra o índice do menor valor em magnitude
            max_magnitude_index = absolute_values.argmax()

            #print(f"middle: {middle_index} | min_index: {min_magnitude_index} | min_value: {central_values[min_magnitude_index]}")

            if max_magnitude_index <= middle_index:
                print(filename + " -> Yes\n")
            else:
                print(filename + " -> No\n")

        if i == 9:

            # Descartamos o início e o fim do histograma por serem "ruído"
            cropped_histogram = histogram[2:-2]

            # Divide o eixo X do histograma cortado em 10 intervalos iguais
            num_intervals = 10
            interval_size = len(cropped_histogram) // num_intervals

            # Calcula a variância para cada intervalo no eixo X
            variations = []
            for i in range(num_intervals):
                start_idx = i * interval_size
                end_idx = (i + 1) * interval_size
                sub_histogram = cropped_histogram[start_idx:end_idx]
                variations.append(np.var(sub_histogram))

            # Encontra o intervalo com a maior variância
            max_variation_idx = np.argmax(variations)

            print(f"Última resposta da imagem {img_path}: {max_variation_idx+1}\n")

        # Plotar
        plt.figure(figsize=(10, 5))
        plt.plot(histogram, color='black')
        title = 'Projeção do Histograma' + str(i) + 'ao Longo do Eixo X'
        plt.title(title)
        plt.xlabel('Posição no Eixo X')
        plt.ylabel('Intensidade Normalizada')
        plt.grid(True)

        # Salvar graficos
        graphic = "porj_" + filename
        plt.savefig(graphic)
        plt.close()

    saida = img_path[-11:-4] + "_saida.png"
    cv2.imwrite(saida, img)
    cv2.destroyAllWindows()




def main(rois, input_dir, output_dir=None):
    
    # Verifica se o diretório de saída existe e criar se não existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de resultados
    results = []

    # Processa os arqs no diretório de entrada
    for arquivo in os.listdir(input_dir):
        img_path = os.path.join(input_dir, arquivo)

        # Remove ruídos
        clean_image_path = remove_vertical_lines(img_path)

        # Segmentar os formulários
        #segment_form(clean_image_path, rois)

    
    print("fim")
    # Ver se tem <dir_saida>
        # Se tiver então geramos as imgs de saida e o results.txt
        # Caso contrário so gera o results.txt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    rois = define_rois()
    main(rois, input_dir, output_dir)
