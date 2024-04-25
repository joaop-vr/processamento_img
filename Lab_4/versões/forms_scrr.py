
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
        print(f"Não foi possível encontrar o arquivo: {filename}")
        return False
    
    # Usa imghdr para verificar o tipo da imagem
    image_type = imghdr.what(filename)
    
    # Retorna True se for um tipo de imagem conhecido
    return image_type in ['jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff']


def define_threshold(img):

    # Calcula o histograma no eixo Y
    histogram_Y = np.sum(img, axis=1)

    # Normaliza o histograma
    histogram_Y = histogram_Y / np.max(histogram_Y)

    # Extrai o campo central de cada tupla do histograma (índice 1)
    central_values_Y = [tup[1] for tup in histogram_Y]
    
    # Calcula os valores absolutos
    absolute_values_Y = np.abs(central_values_Y)

    # Lista dos indices dos mínimos locais
    idx_mins = []
    
    for i in range(0, len(absolute_values_Y)//2):
        if absolute_values_Y[i] < absolute_values_Y[i-1] and absolute_values_Y[i] < absolute_values_Y[i+1]:
            idx_mins.append(i)
            
    # Ordenar de forma decrescente os índices dos mínimos locais
    idx_mins.sort(key=lambda x: absolute_values_Y[x])

    # Calcula as coordenadas do roi referente ao marco zero 
    # (referencial para estabelecer as demais áreas de interesse)
    idx_middle = (idx_mins[0]+idx_mins[1])//2
    x = 0
    y = idx_middle - 80
    w = img.shape[1]-1
    h = 160

    roi = img[y:y+h, x:x+w]
    _, binarized_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

    # Variáveis para armazenar os índices do primeiro pixel preto
    first_black_y = None
    first_black_x = None

    # Percorra a ROI binarizada para encontrar o primeiro pixel preto
    for i in range(roi.shape[0]): 
        for j in range(roi.shape[1]):  
            if (binarized_roi[i, j] == 0).any():  # Se o pixel é preto
                first_black_y = i + y 
                first_black_x = j + x 
                break  # Sai do loop interno se encontrar o primeiro pixel preto
        if first_black_y is not None and first_black_x is not None:
            break  # Sai do loop externo se encontrar o primeiro pixel preto

    if first_black_y is None or first_black_x is  None:
        print("Não foi possível encontrar um pixel preto.")

    return first_black_x, first_black_y


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


def define_rois(id, x, y, img, img_path):

    # Lista de áreas de interesse
    rois = []

    if id == 1:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 1
        roi_1 = (x+722, y+229, 1560, 120)
        roi_2 = (x+722, y+419, 1560, 110)
        roi_3 = (x+722, y+589, 1560, 120)
        roi_4 = (x+722, y+779, 1560, 120)
        roi_5 = (x+722, y+959, 1560, 120)
        roi_6 = (x+722, y+1139, 1560, 90)
        roi_7 = (x+722, y+1309, 590, 140)
        roi_8 = (70, y+1629, 500, 150)
        roi_9 = (1330, y+1639, 600, 120)
        roi_10 = (900, y+1759, 1480, 220)
    else:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 2
        roi_1 = (x+750, y+170, 400, 1050)
        roi_2 = (x+1200, y+169, 400, 1050)
        roi_3 = (x+1650, y+169, 300, 1050)
        roi_4 = (x+2000, y+169, 300, 1050)
        roi_5 = (x+862, y+1289, 130, 120)
        roi_6 = (x+1222, y+1289, 130, 120)
        roi_7 = (x+72, y+1579, 100, 140)
        roi_8 = (x+422, y+1579, 100, 140)
        roi_9 = (x+1352, y+1599, 100, 120)
        roi_10 = (x+1702, y+1599, 100, 120)
        """
        ERA O USADO LA LOGICA DE DPS REMOVER LABELS
        roi_1 = (x+722, y+169, 1560, 150)
        roi_2 = (x+722, y+339, 1560, 150)
        roi_3 = (x+722, y+519, 1560, 150)
        roi_4 = (x+722, y+719, 1560, 150)
        roi_5 = (x+722, y+894, 1560, 150)
        roi_6 = (x+722, y+1069, 1560, 150)
        roi_7 = (x+722, y+1249, 590, 200)
        roi_8 = (70, y+1569, 600, 200)
        roi_9 = (1330, y+1569, 700, 170)
        roi_10 = (900, y+1759, 1490, 220)"""

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


def remove_labels(x, y, img, img_path):

    # Lista das áreas de interesse (ROI)
    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_1 = (x+750, y+170, 400, 1050)
    roi_2 = (x+1200, y+169, 400, 1050)
    roi_3 = (x+1650, y+169, 300, 1050)
    roi_4 = (x+2000, y+169, 300, 1050)
    roi_5 = (x+862, y+1289, 130, 120)
    roi_6 = (x+1222, y+1289, 130, 120)
    roi_7 = (x+72, y+1579, 100, 140)
    roi_8 = (x+422, y+1579, 100, 140)
    roi_9 = (x+1352, y+1599, 100, 120)
    roi_10 = (x+1702, y+1599, 100, 120)

    """
    x, y, w, h = roi_1
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

    x, y, w, h = roi_2
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_3
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_4
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_5
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_6
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_7
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_8
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_9
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_10
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

    filename = "A_saida_" + img_path[:-4].split('/')[-1] + ".png"
    cv2.imwrite(filename, img)
    """

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


def multiple_choice_form1(img, roi):

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


def multiple_choice_form2(img, roi):

    # Variáveis delimitadoras da região de interesse (ROI)
    x, y, w, h = roi
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

    # Recorta região de interesse (ROI)
    roi = img[y:y+h, x:x+w]

    # Realça detalhes perdidos pela mediana
    img_segmented_bolder = highlight_img(roi)

    # Calcula o histograma ao longo do eixo X
    histogram = np.sum(img_segmented_bolder, axis=1)

    # Normaliza o histograma
    histogram = histogram / np.max(histogram)

    # Descarta o início e fim do histograma pois são "ruídos"
    cropped_histogram = histogram[2:-2]

    # Divida o eixo X do histograma em 4 intervalos
    # um para cada pergunta
    num_intervals = 4
    interval_size = len(cropped_histogram) // num_intervals

    # Calcula os parâmetros necessários para estabelecer a resposta da pergunta
    sums = calc_sums(cropped_histogram, num_intervals, interval_size)
    variations = calc_variation(cropped_histogram, num_intervals, interval_size)

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


def segment_form(img, rois, id_form):

    results = []

    # Itera sobre as áreas de interesse e obtém as respostas do formulário
    for i in range(len(rois)):
        if i <= 5:
            if id_form == 0:
                result = multiple_choice_form1(img, rois[i])
            else:
                result = multiple_choice_form2(img, roi[i])
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

            print(f"Processando imagem {arquivo}...")

            # Lista de áreas de interesse
            rois = []

            # Remove os ruídos (linhas pretas contínuas)
            clean_img = remove_noise(img)

            # Define o limiar referencial para calculo das áreas de interesse
            x, y = define_threshold(clean_img)

            if (isForm_1(img_path)):
                id_form = 0
                rois = define_rois(1, x, y, clean_img, img_path)    #PRECISA TIRAR O 'clean_img' e 'img_path' DESSA FUNÇÃO
            else:
                id_form = 1
                rois = define_rois(2, x, y, clean_img, img_path)
                #clean_img = remove_labels(x, y, clean_img, img_path)

            # Segmentar os formulários
            form_results = segment_form(clean_img, rois, id_form)

            # Armazena o registro dos metadados do formulário
            register = (original_name, form_results)
            results.append(register)

        else:
            print(f"{arquivo} não é uma imagem. Ignorando...")

    if results:
        print("Gerando results.txt ...")
        generate_txt(results)

    if output_dir:
        # Verifica se o diretório de saída existe e cria se necessário
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Gerando imagens out ...")
        generate_img_out(results, output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_dir, output_dir)

