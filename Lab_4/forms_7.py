
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


def remove_noise(img_path):

    image = cv2.imread(img_path, 0)

    # Aplicar o filtro de mediana
    denoised = cv2.medianBlur(image, 5)

    name = "clean_" + img_path[-11:]
    print(name)
    cv2.imwrite(name, denoised)

    return name


def dilation(img_path):

    # Carregar a imagem em tons de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Definir o kernel para dilatação
    kernel = np.ones((3,3), np.uint8)  # Kernel 5x5 de uns

    # Aplicar a dilatação na imagem
    img_dilated = cv2.erode(img, kernel, iterations=2)

    name = "dilated_" + img_path[-11:]
    print(name)
    cv2.imwrite(name, img_dilated)

    return img_dilated


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

            # Calcule a variância para cada intervalo no eixo X
            variâncias = []
            for i in range(num_intervalos):
                inicio = i * intervalo_tamanho
                fim = (i + 1) * intervalo_tamanho
                sub_histograma = histograma_cortado[inicio:fim]
                variância_atual = np.var(sub_histograma)
                variâncias.append(variância_atual)


            # Encontre o intervalo com a maior variância
            indice_max_variancia = np.argmax(variâncias)
            max_variancia = variâncias[indice_max_variancia]

            #print(f"Variancias:{variâncias}\n")

            print(f"Resposta de {i} da {img_path}: {indice_max_variancia+1}")
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
                variâncias = []
                for i in range(num_intervalos):
                    inicio = i * intervalo_tamanho
                    fim = (i + 1) * intervalo_tamanho
                    sub_histograma = histograma_cortado[inicio:fim]
                    variância_atual = np.var(sub_histograma)
                    variâncias.append(variância_atual)


                # Encontre o intervalo com a maior variância
                indice_max_variancia = np.argmax(variâncias)
                max_variancia = variâncias[indice_max_variancia]

                #print(f"Variancias:{variâncias}\n")

                print(f"Última resposta da imagem {img_path}: {indice_max_variancia+1}\n")

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





def main(rois, input_dir, output_dir=None):
    
    # Verifica se o diretório de saída existe e criar se não existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de resultados
    results = []

    # Processa os arqs do diretório de entrada
    for arquivo in os.listdir(input_dir):

        # Obtém o caminho relativo ao arquivo
        img_path = os.path.join(input_dir, arquivo)

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

    rois = define_rois()
    main(rois, input_dir, output_dir)
