import sys
import glob
import cv2
import re
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class RegistroImagem:
    def __init__(self, classe, histogramas, relacao_comparacoes):
        self.classe = classe
        self.histogramas = histogramas
        self.relacao_comparacoes = relacao_comparacoes

def aplicar_knn(metodo_id, registros):

    N = len(registros)
    K = 3
    y_pred = []
    for registro in registros:
        tuplas_vetoriais = []
        for i in range(len(registro.relacao_comparacoes)):
            x = registro.relacao_comparacoes[i][1]
            y = registro.relacao_comparacoes[i][2]
            z = registro.relacao_comparacoes[i][3]
            x2 = x ** 2
            y2 = y ** 2
            z2 = z ** 2
            soma = x2 + y2 + z2
            norma = np.sqrt(soma)
            #print("i:", i)
            #print("x:", x)
            #print("y:", y)
            #print("z:", z)
            #print()
            tupla = (registro.relacao_comparacoes[i][0], norma)
            tuplas_vetoriais.append(tupla)

        #print("Arquivo:", registro.classe)
        #print("Tuplas vetoriais:")
        #print(tuplas_vetoriais)

        if metodo_id == 2 or metodo_id == 4:    # Métricas: correlação e interseção
            tuplas_vetoriais.sort(key=lambda x: x[1], reverse=True)
        else:                                   # Métricas: Chi Sqr, Bhattacharyya e Dist. Euclidiana
            tuplas_vetoriais.sort(key=lambda x: x[1])

        k_vizinhos = tuplas_vetoriais[0:K]
        
        #Teste
        print("Amostra:", registro.classe)
        print("K-vizinhos:", k_vizinhos)

        # Conta as classes dentre os k vizinhos
        contagem_classes = Counter(tupla[0] for tupla in k_vizinhos)

        # Enontra a classe mais frequente
        classe_frequente =  contagem_classes.most_common(1)[0][0]

        y_pred.append(classe_frequente)


        y_data = []
        for registro in registros:
            y_data.append(registro.classe)

    return y_data, y_pred

def aplicar_metodo(metodo, registros):
    
    n = len(registros)
    
    for i in range(n):
        hist_pivo_azul = registros[i].histogramas[0]
        hist_pivo_verde = registros[i].histogramas[1]
        hist_pivo_verm = registros[i].histogramas[2]
        for j in range(n):
            if i != j:
                hist_iterado_azul = registros[j].histogramas[0]
                hist_iterado_verde = registros[j].histogramas[1]
                hist_iterado_verm = registros[j].histogramas[2]

                if metodo == "dist_euclidiana":
                    soma_azul = np.sum((hist_pivo_azul - hist_iterado_azul) ** 2)
                    pontuacao_azul = np.sqrt(soma_azul)
                    soma_verde = np.sum((hist_pivo_verde - hist_iterado_verde) ** 2)
                    pontuacao_verde = np.sqrt(soma_verde)
                    soma_verm = np.sum((hist_pivo_verm - hist_iterado_verm) ** 2)
                    pontuacao_verm = np.sqrt(soma_verm)
                else:
                    pontuacao_azul = cv2.compareHist(hist_pivo_azul, hist_iterado_azul, metodo)
                    pontuacao_verde = cv2.compareHist(hist_pivo_verde, hist_iterado_verde, metodo)
                    pontuacao_verm = cv2.compareHist(hist_pivo_verm, hist_iterado_verm, metodo)

                relacao = (registros[j].classe, pontuacao_azul, pontuacao_verde, pontuacao_verm)
                registros[i].relacao_comparacoes.append(relacao)

        soma_azul = 0
        soma_verde = 0
        soma_azul = 0
        for i in range(len(registro[i].relacao_comparacoes)):
            soma_azul = np.sum(registro[i].relacao_comparacoes[i][1])
            soma_verde = np.sum(registro[i].relacao_comparacoes[i][2])
            soma_verm = np.sum(registro[i].relacao_comparacoes[i][3])
        
        if metodo_id == 2 or metodo_id == 4:    # Métricas: correlação e interseção
            classes = []
            pontuacoes = []
            if np.max(soma_azul, soma_verde, soma_verm) == soma_azul:
                for i in range(len(registro[i].relacao_comparacoes)):
                    classes.append(registro[i].relacao_comparacoes[i][0])
                    pontuacoes.append(registro[i].relacao_comparacoes[i][1])
                registro[i].relacao_comparacoes = (classes, pontuacoes)
            if np.max(soma_azul, soma_verde, soma_verm) == soma_verde:
                for i in range(len(registro[i].relacao_comparacoes)):
                    classes.append(registro[i].relacao_comparacoes[i][0])
                    pontuacoes.append(registro[i].relacao_comparacoes[i][2])
                registro[i].relacao_comparacoes = (classes, pontuacoes)
            else:
                for i in range(len(registro[i].relacao_comparacoes)):
                    classes.append(registro[i].relacao_comparacoes[i][0])
                    pontuacoes.append(registro[i].relacao_comparacoes[i][3])
                registro[i].relacao_comparacoes = (classes, pontuacoes)
        else:                                   # Métricas: Chi Sqr, Bhattacharyya e Dist. Euclidiana
            classes = []
            pontuacoes = []
            if np.min(soma_azul, soma_verde, soma_verm) == soma_azul:
                for i in range(len(registro[i].relacao_comparacoes)):
                    classes.append(registro[i].relacao_comparacoes[i][0])
                    pontuacoes.append(registro[i].relacao_comparacoes[i][1])
                registro[i].relacao_comparacoes = (classes, pontuacoes)
            if np.min(soma_azul, soma_verde, soma_verm) == soma_verde:
                for i in range(len(registro[i].relacao_comparacoes)):
                    classes.append(registro[i].relacao_comparacoes[i][0])
                    pontuacoes.append(registro[i].relacao_comparacoes[i][2])
                registro[i].relacao_comparacoes = (classes, pontuacoes)
            else:
                for i in range(len(registro[i].relacao_comparacoes)):
                    classes.append(registro[i].relacao_comparacoes[i][0])
                    pontuacoes.append(registro[i].relacao_comparacoes[i][3])
                registro[i].relacao_comparacoes = (classes, pontuacoes)

    print("relacao de comparacoes do primeiro registro::")
    print("classe:" + registros[0].relacao_comparacoes[0][0])
    print(registros[0].relacao_comparacoes[0][1])
    return registros


def metodos_comparacao(metodo_id):
    metodos = {
        1: "dist_euclidiana",
        2: cv2.HISTCMP_CORREL,
        3: cv2.HISTCMP_CHISQR,
        4: cv2.HISTCMP_INTERSECT,
        5: cv2.HISTCMP_BHATTACHARYYA
    }
    return metodos.get(metodo_id, None)
    


def criar_registros(dir):
    arquivos = glob.glob(dir + "/*.bmp")

    # Testar
    print("arquivos:")
    for arquivo in arquivos:
        print(arquivo)

    registros = []
    
    for arquivo in arquivos:

        # Manipulamos o nome da imagem para associá-la a sua respectiva classe
        partes_nome = arquivo.split("/")
        nome_com_extensao = partes_nome[-1]
        nome_sem_extensao = nome_com_extensao.split(".")[0]
        nome_apenas_caracteres = re.findall("[a-zA-Z]", nome_sem_extensao)
        classe = "".join(nome_apenas_caracteres)

        # Estabelecemos a imagem pivo e calculamos seus histogramas
        imagem = cv2.imread(arquivo)

        hist_azul = cv2.calcHist([imagem], [0], None, [256], [0,256])
        cv2.normalize(hist_azul, hist_azul, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_verde = cv2.calcHist([imagem], [1], None, [256], [0,256])
        cv2.normalize(hist_verde, hist_verde, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_verm = cv2.calcHist([imagem], [2], None, [256], [0,256])
        cv2.normalize(hist_verm, hist_verm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        histogramas = (hist_azul, hist_verde, hist_verm)

        # Istanciando o registro para posteriormente
        # adicioná-lo ao array "registros"
        vet_comparacoes = []
        registro_aux = RegistroImagem(classe, histogramas, vet_comparacoes)
        registros.append(registro_aux)

    return registros

def main():
    if len(sys.argv) < 3:
        print("Comando esperado: histograma.py <metodo_de_calculo> <diretorio_de_imagens>.")
        sys.exit(1)

    dir = sys.argv[2]
    metodo_id = int(sys.argv[1])

    # Armazenando os dados em estruturas de dados (RegistroImagem)
    registros = criar_registros(dir)
    metodo = metodos_comparacao(metodo_id)
    if metodo is None:
        print("Código inserido é inválido!")
        sys.exit(1)
    registros = aplicar_metodo(metodo, registros)

    
    print("Classe: "+registros[0].classe)
    print("HIstogramas::")
    print("Azul:")
    print(registros[0].histogramas[0])
    print("Verde:")
    print(registros[0].histogramas[1])
    print("Vermelho:")
    print(registros[0].histogramas[2])
    print("Relações::")
    print(registros[0].relacao_comparacoes)
    

    # Aplicar KNN
    y_data, y_pred = aplicar_knn(metodo_id, registros)

    print("y_data::")
    print(y_data)

    print("y_pred::")
    print(y_pred)

    # Calcular acurácia e matriz de confusão
    soma = 0
    for i in range(len(y_data)):
        if y_pred[i] == y_data[i]:
            soma = soma + 1
    acuracia = soma/ len(y_data)
    acuracia_formatada = "{:.4f}".format(acuracia)
    print("Acurácia:", acuracia_formatada)

    matriz_confusao = confusion_matrix(y_data, y_pred)
    print("Matriz de Confusão:")
    print(matriz_confusao)

if __name__ == "__main__":
    main()
