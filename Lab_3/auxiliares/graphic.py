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

        psnr_blur = cv.PSNR(img, blur_img)
        psn_mediana = cv.PSNR(img, mediana_blur_img)
        psnr_gauss = cv.PSNR(img, gauss_blur_img)

        registro_media = f"{mascara}.{mascara} {psnr_blur}"
        registro_mediana = f"{mascara}.{mascara} {psn_mediana}"
        registro_gauss = f"{mascara}.{mascara} {psnr_gauss}"

        if (prob == 0.01):
            with open('prob_001.txt', 'a') as arq:
                arq.write(registro_media+"\n")
                arq.write(registro_mediana+"\n")
                arq.write(registro_gauss+"\n")
        elif (prob == 0.02):
            with open('prob_002.txt', 'a') as arq:
                arq.write(registro_media+"\n")
                arq.write(registro_mediana+"\n")
                arq.write(registro_gauss+"\n")
        elif (prob == 0.05):
            with open('prob_005.txt', 'a') as arq:
                arq.write(registro_media+"\n")
                arq.write(registro_mediana+"\n")
                arq.write(registro_gauss+"\n")
        elif (prob == 0.07):
            with open('prob_007.txt', 'a') as arq:
                arq.write(registro_media+"\n")
                arq.write(registro_mediana+"\n")
                arq.write(registro_gauss+"\n")
        elif (prob == 0.1):
            with open('prob_01.txt', 'a') as arq:
                arq.write(registro_media+"\n")
                arq.write(registro_mediana+"\n")
                arq.write(registro_gauss+"\n")



def empilhamento(img, prob):

    vet_ruidos = []

    qntd_imgs = [3, 5, 7, 9]

    for i in range(len(qntd_imgs)):

        for j in range(qntd_imgs[i]):
            aux = sp_noise(img, prob)
            vet_ruidos.append(aux)

        img_empilhadas = stack_images(vet_ruidos)

        for mascara in range(3, 16, 2):

            # Aplicar os filtros para remover ruído
            blur_img = cv.blur(img_empilhadas, (mascara, mascara))  # Filtro cvBlur
            gauss_blur_img = cv.GaussianBlur(img_empilhadas, (mascara, mascara), 0)  # Filtro cvGaussianBlur
            mediana_blur_img = cv.medianBlur(img_empilhadas, mascara)  # Filtro cvMedianBlur

            filename = "emp_blur"+str(qntd_imgs[i])+".txt"
            with open(filename, 'a') as arquivo:
                arquivo.write(f"\nKernel == {mascara},{mascara}")
                psnr = cv.PSNR(img, blur_img)
                registro = "\nPSNR Blur = " + str(psnr)
                arquivo.write(registro)
                arquivo.write("\n")
                
            filename = "emp_gauss"+str(qntd_imgs[i])+".txt"
            with open(filename, 'a') as arquivo:
                arquivo.write(f"\nKernel == {mascara},{mascara}")
                psnr = cv.PSNR(img, gauss_blur_img)
                registro = "\nPSNR Gauss = " + str(psnr)
                arquivo.write(registro)
                arquivo.write("\n")

            filename = "emp_mediana"+str(qntd_imgs[i])+".txt"
            with open(filename, 'a') as arquivo:
                arquivo.write(f"\nKernel == {mascara},{mascara}")
                psnr = cv.PSNR(img, mediana_blur_img)
                registro = "\nPSNR Mediana = " + str(psnr)
                arquivo.write(registro)
                arquivo.write("\n")



def empilhamento2(img, prob):
    vet_ruidos = []
    vet_psnr = []
    
    for i in range(100):  # Loop 100 vezes
        img_ruido = sp_noise(img, prob)
        vet_ruidos.append(img_ruido)

        img_empilhadas = stack_images(vet_ruidos)
        psnr = cv.PSNR(img, img_empilhadas)
        vet_psnr.append(psnr)

    for i in range(len(vet_psnr)):  # Iteração direta sobre os elementos de vet_psnr
        with open(f'empilhadas_dados_{prob}.txt', 'a') as arquivo:
            registro = f"{i+1} {vet_psnr[i]}\n"
            arquivo.write(registro)




def main(argv):

    if (len(sys.argv)!= 3):
        sys.exit("Use: stack <imageIn> <imageOut>>")

    # ler a imagem
    img = cv.imread(argv[1], 0)

    probs = [0.01, 0.02, 0.05, 0.07, 0.1]
    ruidos = []

    for i in range(len(probs)):  # Iterando sobre os índices da lista probs

        prob = probs[i]
        
        #filtragem(img, prob)

        empilhamento2(img, prob)


if __name__ == '__main__':
    main(sys.argv)


