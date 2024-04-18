import sys
import numpy as np
import cv2 as cv
import random
import numpy as np

def stack_images(images):
    stacked_image = np.mean(images, axis=0)
    return stacked_image.astype(np.uint8)

def sp_noise(image,prob):
        
    '''
    Adiociona o ruído Sal & Pimenta à imagem
    prob: Pobabilidade de ruído
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


def filtragem(img, prob):

    noise_img = sp_noise(img, prob)

    for mascara in range(3, 16, 2):

        # Aplicar os filtros para remover ruído
        blur_img = cv.blur(noise_img, (mascara, mascara))  # Filtro cvBlur
        gauss_blur_img = cv.GaussianBlur(noise_img, (mascara, mascara), 0)  # Filtro cvGaussianBlur
        mediana_blur_img = cv.medianBlur(noise_img, mascara)  # Filtro cvMedianBlur

        with open('blur.txt', 'a') as arquivo:
            arquivo.write(f"\nKernel = {mascara},{mascara}")
            psnr = cv.PSNR(img, blur_img)
            registro = "\nPSNR Blur = " + str(psnr)
            arquivo.write(registro)
            arquivo.write("\n")
            
        with open('gauss.txt', 'a') as arquivo:
            arquivo.write(f"\nKernel = {mascara},{mascara}")
            psnr = cv.PSNR(img, gauss_blur_img)
            registro = "\nPSNR Gauss = " + str(psnr)
            arquivo.write(registro)
            arquivo.write("\n")

        with open('mediana.txt', 'a') as arquivo:
            arquivo.write(f"\nKernel = {mascara},{mascara}")
            psnr = cv.PSNR(img, mediana_blur_img)
            registro = "\nPSNR Mediana = " + str(psnr)
            arquivo.write(registro)
            arquivo.write("\n")



def empilhamento(img, prob):

    vet_ruidos = []

    qntd_imgs = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

    for i in range(len(qntd_imgs)):

        for j in range(qntd_imgs[i]):
            aux = sp_noise(img, prob)
            vet_ruidos.append(aux)

        img_empilhadas = stack_images(vet_ruidos)
        with open('empilhadas.txt', 'a') as arquivo:
                arquivo.write(f"\nQntd_imgs == {str(qntd_imgs[i])}")
                psnr = cv.PSNR(img, img_empilhadas)
                registro = "\nPSNR Empilhadas = " + str(psnr)
                arquivo.write(registro)
                arquivo.write("\n")



def gera_arquivos(prob):

    with open('blur.txt', 'a') as arquivo:
            arquivo.write("=========================================\n")
            rotulo = "Análise sob probabilidade de " + str(prob)+"\n"
            arquivo.write(rotulo)
            arquivo.write("\n")

    with open('gauss.txt', 'a') as arquivo:
        arquivo.write("=========================================\n")
        rotulo = "Análise sob probabilidade de " + str(prob)+"\n"
        arquivo.write(rotulo)
        arquivo.write("\n")

    with open('mediana.txt', 'a') as arquivo:
        arquivo.write("=========================================\n")
        rotulo = "Análise sob probabilidade de " + str(prob)+"\n"
        arquivo.write(rotulo)
        arquivo.write("\n")

    with open('empilhadas.txt', 'a') as arquivo:
        arquivo.write("=========================================\n")
        rotulo = "Análise sob probabilidade de " + str(prob)+"\n"
        arquivo.write(rotulo)
        arquivo.write("\n")



def main(argv):

    if (len(sys.argv)!= 2):
        sys.exit("Espera-se: filtragem.py <imageIn>")

    # ler a imagem
    img = cv.imread(argv[1], 0)

    probs = [0.01, 0.02, 0.05, 0.07, 0.1]

    for i in range(len(probs)):  # Iterando sobre os índices da lista probs

        # Criando os arquivos de armazenamento de dados
        gera_arquivos(probs[i])
        
        # Aplica os filtros
        filtragem(img, probs[i])

        # Aplica o empilhamento
        empilhamento(img, probs[i])


if __name__ == '__main__':
    main(sys.argv)


