
import sys
import os
import cv2
import imghdr
import numpy as np
from math import floor
import matplotlib.pyplot as plt


def is_image(filename):
    # Verifica se o arquivo existe
    if not os.path.exists(filename):
        print(f"aqui que dey erro{filename}")
        return False
    
    # Usa imghdr para verificar o tipo da imagem
    image_type = imghdr.what(filename)
    
    # Retorna True se for um tipo de imagem conhecido
    return image_type in ['jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff']


def isForm_1(img_path):

    # Carrega imagem
    img = cv2.imread(img_path)

    # Estabelece as coordenadas da área de interesse
    # (retangulo na porção inicial e central da imagem)
    x, y, w, h = (900, 20, 625, 170)
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

    # Recorta região de interesse (ROI)
    roi = img[y:y+h, x:x+w]

    # Soma os valores do histograma em questão
    histogram = np.sum(roi, axis=0)
    count = np.sum(histogram)
    
    # O valor de limiar foi obtido após
    # analises de formulários do tipo 2
    if count < 70000000:
        return 1    # FOrmulário do tipo 1
    else:
        return 0    # Formulário do tipo 2


def define_rois(id, img, img_path):

    # Lista de áreas de interesse
    rois = []

    # Calcula o histograma no eixo y
    histogram = np.sum(img, axis=1)

    # Normaliza o histograma
    histogram = histogram / np.max(histogram)

    # Deebug
    #cv2.imwrite("Deebug.png", img)
    # Salva o histograma
    #cv2.imwrite("Histograma_Y.png", histogram)

    # Plotar
    plt.figure(figsize=(10, 5))
    plt.plot(histogram, color='black')
    title = 'Projeção do Histograma Longo do Eixo Y'
    plt.title(title)
    plt.xlabel('Posição no Eixo Y')
    plt.ylabel('Intensidade Normalizada')
    plt.grid(True)

    # Salvar graficos
    graphic = "porjY_" + img_path[:-4].split('/')[-1]
    plt.savefig(graphic)
    plt.close()

    if id == 1:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 1
        roi_1 = (850, 760, 1560, 120)
        roi_2 = (850, 950, 1560, 110)
        roi_3 = (850, 1120, 1560, 120)
        roi_4 = (850, 1310, 1560, 120)
        roi_5 = (850, 1490, 1560, 120)
        roi_6 = (850, 1670, 1560, 90)
        roi_7 = (850, 1840, 590, 140)
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
        roi_8 = (70, 2100, 600, 200)
        roi_9 = (1330, 2100, 700, 170)
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


def remove_labels(img):

    # Lista das áreas de interesse (ROI)
    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_1 = (980, 700, 350, 1050)
    roi_2 = (1460, 700, 170, 1050)
    roi_3 = (1900, 700, 140, 1050)
    roi_4 = (2240, 700, 130, 1050)
    roi_5 = (990, 1820, 130, 120)
    roi_6 = (1350, 1820, 130, 120)
    roi_7 = (200, 2110, 100, 140)
    roi_8 = (550, 2110, 100, 140)
    roi_9 = (1480, 2130, 100, 120)
    roi_10 = (1830, 2130, 100, 120)

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

    # Itera sobre os rois e remove os rótulos adjacentes às check-box
    for i in range(len(rois)):

        # Variáveis delimitadoras da região de interesse (ROI)
        x, y, w, h = rois[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)

        # Recorta e recolore a região de interesse (ROI)
        roi = img[y:y+h, x:x+w]
        img[y-2:y+h+2, x-2:x+w+2] = 255

    return img


def remove_noise(img):

    # Aplica o filtro de mediana para remover ruídos
    denoised = cv2.medianBlur(img, 5)

    return denoised


def highlight_img(img):

    # Defini o kernel 
    kernel = np.ones((3,3), np.uint8)

    # Aplica a erosão na imagem para realçar os detalhes perdidos 
    # pela função de remover os ruídos
    img_eroded = cv2.erode(img, kernel, iterations=2)

    return img_eroded


def calc_sums(histogram, num_intervals, interval_size):

    # Calcula o somatório do histograma para cada intervalo no eixo X
    sums = []
    for j in range(num_intervals):
        count = 0
        # Determinação do intervalo
        start_interval = j * interval_size
        end_interval = (j + 1) * interval_size

        # Calculo da soma do histograma no intervalo
        count = np.sum(histogram[start_interval:end_interval])
        sums.append(count)

    return sums


def calc_variation(histogram, num_intervals, interval_size):

    # Calcula a variância do histograma para cada intervalo no eixo X
    variations = []
    for i in range(num_intervals):
        # Determinação do intervalo
        start_interval = i * interval_size
        end_interval = (i + 1) * interval_size

        # Recorte do histograma do intervalo
        sub_histograma = histogram[start_interval:end_interval]

        # Calculo da variância no intervalo
        aux = np.var(sub_histograma)
        variations.append(aux)

    return variations


def calc_histogram(img, roi):

    # Variáveis delimitadoras da região de interesse (ROI)
    x, y, w, h = roi
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

    # Recorta região de interesse (ROI)
    roi = img[y:y+h, x:x+w]

    # Realça detalhes perdidos pela mediana
    img_segmented_bolder = highlight_img(roi)

    # Calcula o histograma ao longo do eixo X
    histogram = np.sum(img_segmented_bolder, axis=0)

    # Normaliza o histograma
    histogram = histogram / np.max(histogram)

    # Descarta o início e fim do histograma pois são "ruídos"
    cropped_histogram = histogram[2:-2]

    return cropped_histogram


def multiple_choice_questions(img, roi):

    # Calcula o histograma
    histogram = calc_histogram(img, roi)

    # Divida o eixo X do histograma em 4 intervalos
    # um para cada pergunta
    num_intervals = 4
    interval_size = len(histogram) // num_intervals

    # Calcula os parâmetros necessários para estabelecer a resposta da pergunta
    sums = calc_sums(histogram, num_intervals, interval_size)
    variations = calc_variation(histogram, num_intervals, interval_size)

    result = None
    best_proportion = float('-inf')  # começa com o menor valor possível

    # Calcula a proporção de varição por somatória dos intervalos
    # e retorna o indice da melhor proporção, que é um
    # indicativo de corresponder a uma resposta do formulário
    for j in range(len(variations)):
        if sums[j] != 0:  # evita divisão por zero
            proportion = variations[j] / sums[j]
            if proportion > best_proportion:
                best_proportion = proportion
                result = j

    return result


def binary_questions(histogram):

    # Determina o indice que divide o hsitograma ao meio
    middle_index = np.floor(len(histogram) / 2).astype(int)

    # Extrai o campo central de cada tupla do histograma (índice 1)
    central_values = [tup[1] for tup in histogram[2:-2]]
    
    # Calcula os valores absolutos
    absolute_values = np.abs(central_values)
    
    # Encontra o índice do menor valor em magnitude
    min_magnitude_index = absolute_values.argmin()

    # Indica se a resposta foi Yes (return 0) ou No (return 1)
    if min_magnitude_index <= middle_index:
        return 0
    else:
        return 1


def scalar_question(histogram):

    # Divide o eixo X do histograma cortado em 10 intervalos iguais
    num_intervals = 10
    interval_size = len(histogram) // num_intervals

    # Calcula a variância para cada intervalo no eixo X
    sums = []
    for j in range(num_intervals):
        count = 0
        start_interval = j * interval_size
        end_interval = (j + 1) * interval_size
        count = np.sum(histogram[start_interval:end_interval])
        sums.append(count)

    # Encontra o intervalo com a menor soma
    min_sum_idx = np.argmin(sums)
    return min_sum_idx


def segment_form(img, rois):

    results = []

    # Itera sobre as áreas de interesse e obtém as respostas do formulário
    for i in range(len(rois)):
        if i <= 5:
            result = multiple_choice_questions(img, rois[i])
        else:
            # Calcula o histograma
            histogram = calc_histogram(img, rois[i])
            if 6 <= i <= 8:
                result = binary_questions(histogram)
            elif i == 9:
                result = scalar_question(histogram)
        results.append(result)

    return results


def generate_txt(results):

    # Estruturas de dados utilizadas
    multiple_answers = np.zeros((6, 4))
    binary_answers = np.zeros((3, 2))
    scalar_answer = 0

    # Obtenção dos dados
    for result in results:
        for i in range(6):
            multiple_answers[i][result[1][i]] += 1
        for i in range(3):
            binary_answers[i][result[1][i+6]] += 1
        scalar_answer += result[1][len(result[1])-1]+1

    # Calculo das porcentagens
    mult_percent = multiple_answers / len(results) * 100
    bi_percent = binary_answers / len(results) * 100
    scalar_average = scalar_answer / len(results) 
    
    # Formata os resultados
    formatted_mult_percent = '\n'.join([' '.join(map(str, row)) for row in mult_percent.astype(int)])
    formatted_bi_percent = '\n'.join([' '.join(map(str, row)) for row in bi_percent.astype(int)])
    formatted_scalar_average = str(scalar_average)

    # Cria o arquivo results.txt
    with open("results.txt", "w") as file:
        file.write(formatted_mult_percent + "\n")
        file.write(formatted_bi_percent + "\n")
        file.write(formatted_scalar_average)


def select_field(type_form, num_question, num_answer):

    if type_form == 1:
        if num_question <= 5:
            x = 940 + (num_answer * 460)
            y = 860 + (num_question * 180)
            w = 80
            h = 5
        elif num_question == 6:
            x = 920 + (num_answer * 340)
            y = 1940
            w = 80
            h = 5
        elif num_question == 7:
            x = 140 + (num_answer * 330)
            y = 2270
            w = 80
            h = 5
        elif num_question == 8:
            x = 1430 + (num_answer * 330)
            y = 2270
            w = 80
            h = 5
        else:
            x = 950 + (num_answer * 143)
            y = 2480
            w = 80
            h = 5
    else:
        if num_question <= 5:
            x = 920 + (num_answer * 440)
            y = 840 + (num_question * 180)
            w = 80
            h = 5
        elif num_question == 6:
            x = 920 + (num_answer * 340)
            y = 1920
            w = 80
            h = 5
        elif num_question == 7:
            x = 140 + (num_answer * 330)
            y = 2215
            w = 80
            h = 5
        elif num_question == 8:
            x = 1430 + (num_answer * 330)
            y = 2215
            w = 80
            h = 5
        else:
            x = 950 + (num_answer * 143)
            y = 2460
            w = 80
            h = 5

    return (x, y, w, h)


def generate_img_out(results, output_dir):

    # 'result' = (img_path, (respostas))
    for result in results:

        # Carrega imagem
        img = cv2.imread(result[0])

        # Faz as marcações nas questões de multipla escolha
        for i in range(6):
            x, y, w, h = select_field(isForm_1(result[0]), i, result[1][i])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

        # Faz as marcações nas questões binárias
        for i in range(6,9):
            x, y, w, h = select_field(isForm_1(result[0]), i, result[1][i])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
        
        # Faz a marcação na questão escalar
        x, y, w, h = select_field(isForm_1(result[0]), 9, result[1][9])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

        filename = result[0][:-4].split('/')[-1]
        filename = output_dir + "/" + filename + ".out.png"
        print(f"Gerando imagem {filename}")
        cv2.imwrite(filename, img)


def main(input_dir, output_dir=None):

    # Lista de resultados
    results = []

    # Processa os arqs do diretório de entrada
    for arquivo in os.listdir(input_dir):

        # Obtém o caminho relativo ao arquivo
        img_path = os.path.join(input_dir, arquivo)

        img = cv2.imread(img_path)
        original_name = img_path

        if is_image(img_path):

            #print(f"Processando imagem {arquivo}...")
            # Lista de áreas de interesse
            #rois = []

            #if (isForm_1(img_path)):
                #rois = define_rois(1, img, img_path)
            #else:
                #rois = define_rois(2, img, img_path)
                #img = remove_labels(img)

            # Remove os ruídos (linhas pretas contínuas)
            clean_img = remove_noise(img)

            # Segmentar os formulários
            #form_results = segment_form(clean_img, rois)

            # Armazena o registro dos metadados do formulário
            #register = (original_name, form_results)
            #results.append(register)
            #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #q = cv2.equalizeHist(img_gray)
            #filename = "EQ_"+ img_path[:-4].split('/')[-1] + ".png"
            #cv2.imwrite(filename, eq)

            print(f"\nFILE:{img_path}")

            # Calcula o histograma no eixo y
            histogram_Y = np.sum(clean_img, axis=1)

            # Normaliza o histograma
            histogram_Y = histogram_Y / np.max(histogram_Y)

            # Extrai o campo central de cada tupla do histograma (índice 1)
            central_values_Y = [tup[1] for tup in histogram_Y]
            
            # Calcula os valores absolutos
            absolute_values_Y = np.abs(central_values_Y)

            # Encontre o índice do valor mínimo na primeira metade de absolute_values_Y
            """
            min_idx_Y = np.argmin(absolute_values_Y[:len(absolute_values_Y)//2-1])
            print(f"len(absolute_values_Y)-1: {len(absolute_values_Y)//2-1}")
            print(f"Min no eixo Y: {min_idx_Y}")

            # Porção à esquerda do histograma
            idx_min_lefth = np.argmin(absolute_values_Y[:min_idx_Y])
            print(f"Min à esquerda: {idx_min_lefth}")

            # Porção à direita do histograma
            idx_min_right = min_idx_Y + 1 + np.argmin(absolute_values_Y[min_idx_Y+1:len(absolute_values_Y)//2-1])
            print(f"Min à direita: {idx_min_right}")

            max_idx = np.argmax(absolute_values_Y)

            dif_lefth = np.abs(idx_min_lefth - min_idx_Y)
            dif_right = np.abs(idx_min_right - min_idx_Y)

            if dif_lefth < dif_right:
                idx_iteration = min_idx_Y
                while idx_iteration > 0 and absolute_values_Y[idx_iteration] <= absolute_values_Y[max_idx]:
                    idx_iteration -= 1
                idx_middle_roi = idx_iteration
            else:
                idx_iteration = min_idx_Y
                while idx_iteration < len(absolute_values_Y)//2-1 and absolute_values_Y[idx_iteration] <= absolute_values_Y[max_idx]:
                    idx_iteration += 1
                idx_middle_roi = idx_iteration

            print(f"Max: {absolute_values_Y[max_idx]}")
            print(f"lefth: {idx_min_lefth}")
            print(f"idx_middle_roi: {idx_middle_roi}")
            print(f"right: {idx_min_right}")
            print(f"Confirmação: absolute_values_Y[idx_middle_roi]: {absolute_values_Y[idx_middle_roi]}")
            """

            minima_indices = []
    
            for i in range(0, len(absolute_values_Y)//2):
                if absolute_values_Y[i] < absolute_values_Y[i-1] and absolute_values_Y[i] < absolute_values_Y[i+1]:
                    minima_indices.append(i)
                    
            # Ordenar os índices dos mínimos locais
            minima_indices.sort(key=lambda x: absolute_values_Y[x])

            print(minima_indices[:2])

            idx_middle = (minima_indices[0]+minima_indices[1])//2
            print(f"idx_middle: {idx_middle}")
            print(f"absolute_values_Y[idx_middle]:{absolute_values_Y[idx_middle]}")

            x = 0
            y = idx_middle - 80
            w = img.shape[1]-1
            h = 160

            roi = clean_img[y:y+h, x:x+w]
            _, binarized_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
            
            # Variáveis para armazenar os índices do primeiro pixel preto
            first_black_i = None
            first_black_j = None

            # Percorra a ROI binarizada para encontrar o primeiro pixel preto
            for i in range(roi.shape[0]):  # Percorre as linhas
                for j in range(roi.shape[1]):  # Percorre as colunas
                    if (binarized_roi[i, j] == 0).any():  # Se o pixel é preto
                        first_black_i = i + y  # Atualiza o índice i na imagem original
                        first_black_j = j + x  # Atualiza o índice j na imagem original
                        break  # Sai do loop interno se encontrar o primeiro pixel preto
                if first_black_i is not None and first_black_j is not None:
                    break  # Sai do loop externo se encontrar o primeiro pixel preto

            # Define a cor do pixel encontrado como (0,255,0) (verde)
            if first_black_i is not None and first_black_j is not None:
                clean_img[first_black_i, first_black_j] = (0, 255, 0)
                print(f"x: {first_black_i} y: {first_black_j} <------------------")
            else:
                print("Não foi possível encontrar um pixel preto.")

            #print(f"x:{x} y:{y} w:{w} h:{h}")
            #print(f"altura:{img.shape[0]} largura:{img.shape[1]}")

            cv2.rectangle(clean_img, (x,y), (x+w, y+h), (255,0, 255), 2)
            img_final_filename = "Segmentada_" + img_path[:-4].split('/')[-1] + ".png"
            cv2.imwrite(img_final_filename, clean_img)

            img_final_filename = "Roi" + img_path[:-4].split('/')[-1] + ".png"
            cv2.imwrite(img_final_filename, roi)


            # Calcula o histograma no eixo x
            histogram_X = np.sum(clean_img, axis=0)

            # Normaliza o histograma
            histogram_X = histogram_X / np.max(histogram_X)                 

            # Extrai o campo central de cada tupla do histograma (índice 1)
            central_values_X = [tup[1] for tup in histogram_X]
            
            # Calcula os valores absolutos
            absolute_values_X = np.abs(central_values_X)  

            min_idx_X = np.argmin(absolute_values_X)
            #print(f"Min no eixo X: {min_idx_X}")

            """min_1st = absolute_values[0]
            min_1st_idx = 0
            min_2nd = absolute_values[0]
            min_2nd_idx = 0
            min_3rd = absolute_values[0]
            min_3rd_idx = 0

            for i in range(len(absolute_values)):
                if 0.5 < absolute_values[i] < min_1st and abs(absolute_values[i] - min_1st) > 50:
                    min_3rd = min_2nd
                    min_3rd_idx = min_2nd_idx
                    min_2nd = min_1st
                    min_2nd_idx = min_1st_idx
                    min_1st = absolute_values[i]
                    min_1st_idx = i
                elif 0.5 < absolute_values[i] < min_2nd and abs(absolute_values[i] - min_2nd) > 50:
                    min_3rd = min_2nd
                    min_3rd_idx = min_2nd_idx
                    min_2nd = absolute_values[i]
                    min_2rd_idx = i
                elif 0.5 < absolute_values[i] < min_3rd and abs(absolute_values[i] - min_3rd) > 50:
                    min_3rd = absolute_values[i]
                    min_3rd_idx = i

            print(f"1st:{min_1st_idx}")
            print(f"2nd:{min_2nd_idx}")
            print(f"3nd:{min_3rd_idx}")"""

            # Inicializando as variáveis para os índices dos três menores mínimos locais
            #min_indices = []

            # Percorrendo o histograma para encontrar os mínimos locais
            #for i in range(1, len(absolute_values) - 1):  # Ignoramos os extremos para verificar mínimos locais
                #if absolute_values[i] < absolute_values[i-1] and absolute_values[i] < absolute_values[i+1]:
                    #min_indices.append(i)

            # Ordenando os índices dos mínimos locais pelos valores do absolute_valuesa
            #min_indices.sort(key=lambda x: absolute_values[x])

            # Pegando os três primeiros índices (os três menores mínimos locais)
            #min_indices = min_indices[:2]
            #min_idx = min_indices[0]
            #max_idx = np.argmax(histogram)
            #print(min_indices)

            #vet_mins = []
            #for i in range(len(absolute_values)):

            #minnnn = np.argmin(absolute_values)
            #print(f"MINNNNNN:{minnnn}")

            #j = min_idx
            #x1 = absolute_values[j]
            #x2 = absolute_values[j+3]
            #x3 = absolute_values[j+6]
            #while (x1 != x2 and x1 != x3 and x2 != x3 and x1,x2,x3 < absolute_values[max_idx]):
            #    j += 1

            #imagem_recortada = img[j:, :]
            #cv2.imwrite("Recortada.png", imagem_recortada)
            # Encontre o índice do bin com a maior variância
            #indice_max_variancia = np.argmin(absolute_values[2:-2])
            #print(indice_max_variancia)

            # Deebug
            #cv2.imwrite("Deebug.png", img)
            # Salva o histograma
            #cv2.imwrite("Histograma_Y.png", histogram)

            # Plotar
            #plt.figure(figsize=(10, 5))
            #plt.plot(histogram, 'black')

            # Plotar
            
            plt.figure(figsize=(10, 5))
            plt.plot(histogram_X, 'red')

            title = 'Projeção do Histograma Longo do Eixo X'
            plt.title(title)
            plt.xlabel('Posição no Eixo X')
            plt.ylabel('Intensidade Normalizada')
            plt.grid(True)

            # Salvar gráfico
            graphic = "porjX_" + img_path[:-4].split('/')[-1]
            plt.savefig(graphic)
            plt.close()

            # Plotar
            plt.figure(figsize=(10, 5))
            plt.plot(histogram_Y, 'black')

            title = 'Projeção do Histograma Longo do Eixo Y'
            plt.title(title)
            plt.xlabel('Posição no Eixo Y')
            plt.ylabel('Intensidade Normalizada')
            plt.grid(True)

            # Salvar gráfico
            graphic = "porjY_" + img_path[:-4].split('/')[-1]
            plt.savefig(graphic)
            plt.close()

        else:
            print(f"{arquivo} não é uma imagem. Ignorando...")

    

    #if results:
        #print("Gerando results.txt ...")
        #generate_txt(results)

    #if output_dir:
        # Verifica se o diretório de saída existe e cria se necessário
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)
        
        #print("Gerando imagens out ...")
        #generate_img_out(results, output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_dir, output_dir)

