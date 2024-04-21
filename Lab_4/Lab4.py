
import sys
import os
import cv2
import numpy as np
from math import floor
import matplotlib.pyplot as plt


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
        return 1
    else:
        return 0


def define_rois(id):

    # Lista de áreas de interesse
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

    # Carrega imagem
    img = cv2.imread(img_path)

    # Lista das áreas de interesse (ROI)
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

    output = img_path[-11:-4] + "_labels_deleted.png"
    cv2.imwrite(output, img)

    return output


def remove_noise(img_path):

    img = cv2.imread(img_path, 0)

    # Aplica o filtro de mediana para remover ruídos
    denoised = cv2.medianBlur(img, 5)

    output = "clean_" + img_path[-11:]
    cv2.imwrite(output, denoised)

    return output


def highlight_img(img):

    # Defini o kernel 
    kernel = np.ones((3,3), np.uint8)

    # Aplica a erosão na imagem para realçar os detalhes perdidos 
    # pela função de remover os ruídos
    img_eroded = cv2.erode(img, kernel, iterations=2)

    output = "highlighted.png"
    cv2.imwrite(output, img_eroded)

    return img_eroded


def calc_sums(histogram, num_intervals, interval_size):

    # Calcula o somatório do histograma para cada intervalo no eixo X
    sums = []
    for j in range(num_intervals):
        count = 0
        start_interval = j * interval_size
        end_interval = (j + 1) * interval_size
        count = np.sum(histogram[start_interval:end_interval])
        sums.append(count)

    return sums


def calc_variation(histogram, num_intervals, interval_size):

    # Calcula a variância do histograma para cada intervalo no eixo X
    variations = []
    for i in range(num_intervals):
        start_interval = i * interval_size
        end_interval = (i + 1) * interval_size
        sub_histograma = histogram[start_interval:end_interval]
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
    return min_sum_idx+1


def segment_form(img_path, rois):

    # Carrega imagem
    img = cv2.imread(img_path)

    results = []

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

    saida = img_path[-10:-4] + "_saida.png"
    cv2.imwrite(saida, img)

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
        scalar_answer += result[1][len(result[1])-1]

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


def select_field(type_form, type_question, num_question, num_answer):

    if type_form == 1:
        if type_question == "multiple":
            x = 940 + (num_answer * 460)
            y = 860 + (num_question * 180)
            w = 80
            h = 5
        elif type_question == "binary":
        else:
    else:
        print("Ainda em desenvolvimento")

    return (x, y, w, h)


def generate_img_out(results):

    for result in results:

        # Carrega imagem
        img = cv2.imread(result[0])

        for i in range(6):
            x, y, w, h = select_field(isForm_1(result[0]), "multiple", i, result[1][i])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

        for i in range(3):
            x, y, w, h = select_field(isForm_1(result[0]), "binary", i, result[1][i])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
        
        filename = result[0][:-4] + ".out.png"
        cv2.imwrite(filename, img)



def main(input_dir, output_dir=None):
    
    # Verifica se o diretório de saída existe e criar se não existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de resultados
    results = []

    # Processa os arqs do diretório de entrada
    for arquivo in os.listdir(input_dir):

        # Obtém o caminho relativo ao arquivo
        img_path = os.path.join(input_dir, arquivo)
        original_name = img_path

        rois = []

        if (isForm_1(img_path)):
            rois = define_rois(1)
        else:
            rois = define_rois(2)
            img_path = remove_labels(img_path) # Remove os rótulos (Excellent, Good, etc)

        # Remove os ruídos (linhas pretas contínuas)
        clean_img = remove_noise(img_path)

        # Segmentar os formulários
        form_results = segment_form(clean_img, rois)

        register = (original_name, form_results)
        results.append(register)

    generate_txt(results)

    if output_dir:
        generate_img_out(results)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_dir, output_dir)

