import imghdr
import os
import sys

def is_image(filename):
    # Verifica se o arquivo existe
    if not os.path.exists(filename):
        return False
    
    # Usa imghdr para verificar o tipo da imagem
    image_type = imghdr.what(filename)
    
    # Retorna True se for um tipo de imagem conhecido
    return image_type in ['jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff']

# Teste a função
filename = sys.argv[1]
if is_image(filename):
    print(f'{filename} é uma imagem.')
else:
    print(f'{filename} não é uma imagem ou o tipo de imagem não é suportado.')
