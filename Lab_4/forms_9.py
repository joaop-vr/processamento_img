
import sys
import os
import cv2
import numpy as np
from math import floor
import matplotlib.pyplot as plt


def isForm_1(img_path):

    img = cv2.imread(img_path)

    # Estabelece as coordenadas da área de interesse
    # (retangulo na porção inicial e central da imagem)
    x, y, w, h = (900, 20, 625, 170)
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

    # Recortar região de interesse (ROI)
    roi = img[y:y+h, x:x+w]

    histogram = np.sum(roi, axis=0)

    count = np.sum(histogram)
    
    # O valor de limiar foi obtido após
    # analises de formulários do tipo 2
    if count < 70000000:
        return 1
    else:
        return 0


def define_rois(id):

    rois = []

    if id == 1:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 1
        roi_1 = (850, 760, 1560, 120)
        roi_2 = (850, 950, 1560, 110)
        roi_3 = (850, 1120, 1560, 120)
        roi_4 = (850, 1310, 1560, 120)
        roi_5 = (850, 1490, 1560, 120)
        roi_6 = (850, 1670, 1560, 90)
        roi_7 = (850, 1770, 570, 200)
        roi_8 = (70, 2160, 500, 150)
        roi_9 = (1330, 2170, 600, 120)
        roi_10 = (900, 2290, 1480, 220)
    else:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 2
        roi_1 = (850, 700, 1560, 150)
        roi_2 = (850, 870, 1560, 150)
        roi_3 = (850, 1050, 1560, 150)
        roi_4 = (850, 1250, 1560, 150)
        roi_5 = (850, 1425, 1560, 150)
        roi_6 = (850, 1600, 1560, 150)
        roi_7 = (850, 1780, 590, 200)
        roi_8 = (70, 2110, 600, 200)
        roi_9 = (1330, 2120, 700, 170)
        roi_10 = (900, 2290, 1490, 220)

    rois.append(roi_1)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)
    rois.append(roi_6)
    rois.append(roi_7)
    rois.append(roi_8)
    rois.append(roi_9)
    rois.append(roi_10)

    return rois


def remove_labels(img_path):

    # Carregar imagem
    img = cv2.imread(img_path)

    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_1 = (980, 700, 350, 1050)
    roi_2 = (1470, 700, 170, 1050)
    roi_3 = (1900, 700, 140, 1050)
    roi_4 = (2240, 700, 130, 1050)
    roi_5 = (990, 1820, 130, 120)
    roi_6 = (1350, 1820, 130, 120)
    roi_7 = (210, 2130, 100, 120)
    roi_8 = (560, 2130, 100, 120)
    roi_9 = (1480, 2130, 100, 120)
    roi_10 = (1830, 2130, 100, 120)

    # Adiciona os rois à lista global
    rois.append(roi_1)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)
    rois.append(roi_6)
    rois.append(roi_7)
    rois.append(roi_8)
    rois.append(roi_9)
    rois.append(roi_10)


    for i in range(len(rois)):

        # Variáveis delimitadoras da região de interesse (ROI)
        x, y, w, h = rois[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)

        # Recortar e recolore a região de interesse (ROI)
        roi = img[y:y+h, x:x+w]
        img[y-2:y+h+2, x-2:x+w+2] = 255


    output = img_path[-11:-4] + "_labels_deleted.png"
    cv2.imwrite(output, img)

    return output


def remove_noise(img_path):

    img = cv2.imread(img_path, 0)

    # Aplicar o filtro de mediana
    denoised = cv2.medianBlur(img, 5)

    output = "clean_" + img_path[-11:]
    cv2.imwrite(output, denoised)

    return output


def segment_form(img_path, rois):

    # Carregar imagem
    img = cv2.imread(img_path)

    print(f"arquivo atual::{img_path}")

    for i in range(len(rois)):

        if i <= 5:

            # Variáveis delimitadoras da região de interesse (ROI)
            x, y, w, h = rois[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

            # Recorta e salva região de interesse (ROI)
            roi = img[y:y+h, x:x+w]
            filename = "roi"+str(i) + img_path[:-4] + ".png"
            cv2.imwrite(filename, roi)

            # Realça detalhes perdidos pela mediana
            img_segmented_bolder = dilation(filename)

            # Calcula o histograma ao longo do eixo X
            histogram = np.sum(img_segmented_bolder, axis=0)

            # Normaliza o histograma
            histogram = histogram / np.max(histogram)


            # Defina os índices de início e fim do histograma
            inicio_histo = 2
            fim_histo = -2
            histograma_cortado = histogram[inicio_histo:fim_histo]

            # Divida o eixo X do histograma cortado em 10 intervalos iguais
            num_intervalos = 4
            intervalo_tamanho = len(histograma_cortado) // num_intervalos

            sums = calc_sums(histograma_cortado, num_intervalos, intervalo_tamanho)

            variations = calc_variation(histograma_cortado, num_intervalos, intervalo_tamanho)

            melhor_indice = None
            melhor_razao = float('-inf')  # começa com o menor valor possível

            for j in range(len(variations)):
                    if sums[j] != 0:  # evita divisão por zero
                        razao = variations[j] / sums[j]
                        if razao > melhor_razao:
                            melhor_razao = razao
                            melhor_indice = j

            print(f"Resposta de {i} da {img_path}: {melhor_indice+1}")
        else:
            # Variáveis delimitadoras da região de interesse (ROI)
            x, y, w, h = rois[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

            # Recorta e salva região de interesse (ROI)
            roi = img[y:y+h, x:x+w]
            filename = "roi"+str(i) + img_path[:-4] + ".png"
            cv2.imwrite(filename, roi)

            img_segmented = cv2.imread(filename)

            # Calcula o histograma ao longo do eixo X
            histogram = np.sum(img_segmented, axis=0)

            # Normaliza o histograma
            histogram = histogram / np.max(histogram)

            if 6 <= i <= 8:

                # Determina o indice que divide o hsitograma ao meio
                middle_index = np.floor(len(histogram) / 2).astype(int)

                # Extrai o campo central de cada tupla do histograma (índice 1)
                central_values = [tup[1] for tup in histogram[2:-2]]
                
                # Calcula os valores absolutos
                absolute_values = np.abs(central_values)
                
                # Encontra o índice do menor valor em magnitude
                min_magnitude_index = absolute_values.argmin()

                if min_magnitude_index <= middle_index:
                    print(filename + " -> Yes")
                else:
                    print(filename + " -> No")

            elif i == 9:
                # Defina os índices de início e fim do histograma
                inicio_histo = 2
                fim_histo = -2
                histograma_cortado = histogram[inicio_histo:fim_histo]

                # Divida o eixo X do histograma cortado em 10 intervalos iguais
                num_intervalos = 10
                intervalo_tamanho = len(histograma_cortado) // num_intervalos

                # Calcule a variância para cada intervalo no eixo X
                sums = []
                for j in range(num_intervalos):
                    count = 0

                    inicio = j * intervalo_tamanho
                    fim = (j + 1) * intervalo_tamanho

                    count = np.sum(histograma_cortado[inicio:fim])

                    sums.append(count)


                # Encontre o intervalo com a menor soma
                min_sum_idx = np.argmin(sums)

                #print(f"Variancias:{variâncias}\n")
                print(f"Última resposta da imagem {img_path}: {min_sum_idx+1}\n")

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

    saida = img_path[-10:-4] + "_saida.png"
    cv2.imwrite(saida, img)
    cv2.destroyAllWindows()


def dilation(img_path):

    # Carregar a imagem em tons de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Definir o kernel 
    kernel = np.ones((3,3), np.uint8)

    # Aplicar a erosão na imagem
    img_eroded = cv2.erode(img, kernel, iterations=2)

    output = "eroded_" + img_path[-11:]
    cv2.imwrite(output, img_eroded)

    return img_eroded


def calc_sums(histogram, num_intervals, interval_size):

    # Calcula a variância para cada intervalo no eixo X
    sums = []
    for j in range(num_intervals):
        count = 0

        start_interval = j * interval_size
        end_interval = (j + 1) * interval_size

        count = np.sum(histogram[start_interval:end_interval])

        sums.append(count)

    return sums


def calc_variation(histogram, num_intervals, interval_size):

    # Calcula a variância para cada intervalo no eixo X
    variations = []
    for i in range(num_intervals):
        start_interval = i * interval_size
        end_interval = (i + 1) * interval_size
        sub_histograma = histogram[start_interval:end_interval]
        aux = np.var(sub_histograma)
        variations.append(aux)

    return variations


def main(input_dir, output_dir=None):
    
    # Verifica se o diretório de saída existe e criar se não existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de resultados
    results = []

    # Processa os arqs do diretório de entrada
    for arquivo in os.listdir(input_dir):

        print("Oi")

        # Obtém o caminho relativo ao arquivo
        img_path = os.path.join(input_dir, arquivo)

        rois = []

        if (isForm_1(img_path)):
            print(f"O arquivo {arquivo} é Form 1")
            rois = define_rois(1)
        else:
            print(f"O arquivo {arquivo} é Form 2")
            rois = define_rois(2)

            # Remove os rótulos (Excellent, Good, etc)
            img_path = remove_labels(img_path)

        # Remove os ruídos (linhas pretas contínuas)
        clean_img = remove_noise(img_path)

        # Segmentar os formulários
        segment_form(clean_img, rois)

    

    # Ver se tem <dir_saida>
        # Se tiver então geramos as imgs de saida e o results.txt
        # Caso contrário so gera o results.txt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_dir, output_dir)
