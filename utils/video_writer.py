import cv2
import glob

# Nome do arquivo de saída do vídeo
output_video = 'output_video2.avi'

# Pegar lista de imagens no diretório
images = list(sorted(glob.glob('dataset\\MODD\\frames\\*L.jpg')))
height, width, _ = cv2.imread(images[0]).shape

# Definir codec e criar objeto VideoWriter
# cv.VideoWriter_fourcc(*'XVID') é o codec que define o formato de vídeo (pode ser alterado)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))

# Loop sobre as imagens e adicioná-las ao vídeo
for image in images:
    frame = cv2.imread(image)
    # Escrever o frame no vídeo
    video.write(frame)

# Libera o objeto VideoWriter
video.release()

print(f"Vídeo salvo como {output_video}")