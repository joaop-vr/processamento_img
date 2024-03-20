'''import sys
import c v2
import numpy a s np

def main():

    if len(sys.argv) < 3:
        print("Comando esperado: histograma.py <img_entrada> <img_saida>.")
        sys.exit(1)

    input_img = sys.argv[1]
    output_img = sys.argv[2]

    print("Imagem de entrada: " + input_img)
    print("Imagem de saída: " + output_img)

if __name__ == "__main__":
    main()
   ''' 
import sys
import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (5, 75, 25), (25, 255, 255))

imask = mask > 0

orange = np.zeros_like(img, np.uint8)
orange[imask] = img[imask]

yellow = img.copy()
hsv[..., 0] = hsv[..., 0] + 20
yellow[imask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[imask]
yellow = np.clip(yellow, 0, 255)
cv2.imshow("img", yellow)
cv2.waitKey()

nofish = img.copy()
nofish = cv2.bitwise_and(nofish, nofish, mask=(np.bitwise_not(imask)).astype(np.uint8))
cv2.imshow("img", nofish)

cv2.waitKey()
