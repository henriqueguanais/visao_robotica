import cv2
import numpy as np
import matplotlib.pyplot as plt

class DistanceMeter():
    """Classe que faz o mapa de disparidade de uma imagem estereo."""
    def __init__(self, focal_length, baseline) -> None:
        self.focal_length = focal_length
        self.baseline = baseline
    

    def load_stereo_params(self, stereo_method_file):
        '''Carrega os parâmetros do método estéreo'''
        cv_file = cv2.FileStorage(stereo_method_file, cv2.FILE_STORAGE_READ)
        if not cv_file.isOpened():
            print("Erro: Não foi possível abrir o arquivo de parâmetros estéreo.")
        else:
            # Extrair os parâmetros do método estéreo
            PreFilterType = int(cv_file.getNode("PreFilterType").real())
            PreFilterSize = int(cv_file.getNode("PreFilterSize").real())
            PreFilterCap = int(cv_file.getNode("PreFilterCap").real())
            SADWindowSize = int(cv_file.getNode("SADWindowSize").real())
            MinDisparity = int(cv_file.getNode("MinDisparity").real())
            NumDisparities = int(cv_file.getNode("NumDisparities").real())
            TextureThreshold = int(cv_file.getNode("TextureThreshold").real())
            UniquenessRatio = int(cv_file.getNode("UniquenessRatio").real())
            SpeckleWindowSize = int(cv_file.getNode("SpeckleWindowSize").real())
            SpeckleRange = int(cv_file.getNode("SpeckleRange").real())
            Disp12MaxDiff = int(cv_file.getNode("Disp12MaxDiff").real())
            cv_file.release()

        # Configurar StereoBM usando parâmetros carregados
        self.stereo = cv2.StereoBM_create(
            numDisparities=NumDisparities,
            blockSize=SADWindowSize
        )

        self.stereo.setPreFilterType(PreFilterType)
        self.stereo.setPreFilterSize(PreFilterSize)
        self.stereo.setPreFilterCap(PreFilterCap)
        self.stereo.setTextureThreshold(TextureThreshold)
        self.stereo.setUniquenessRatio(UniquenessRatio)
        self.stereo.setSpeckleWindowSize(SpeckleWindowSize)
        self.stereo.setSpeckleRange(SpeckleRange)
        self.stereo.setDisp12MaxDiff(Disp12MaxDiff)

        
    def disparity_compute(self, imgL, imgR, object_position, bb_size):
        '''Calcula a disparidade e a profundidade de um objeto na imagem'''
        self.x = object_position[0]
        self.y = object_position[1]
        self.w = bb_size[0]
        self.h = bb_size[1]
        self.imgL = imgL
        self.imgR = imgR

        # Calcular disparidade
        self.disparity_map = self.stereo.compute(self.imgL, self.imgR).astype(np.float32)/16
        disparity_normalize = cv2.normalize(src=self.disparity_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        self.disparity_normalize = cv2.applyColorMap(disparity_normalize, cv2.COLORMAP_JET)

        # Extrair o valor médio de disparidade da região
        region_disparity = self.disparity_map[self.y:self.y+self.h, self.x:self.x+self.w]
        mean_disparity = np.mean(region_disparity)

        # Calcular a profundidade usando a média da disparidade
        if mean_disparity > 0:  # Evitar divisão por zero
            self.depth = (self.focal_length * self.baseline) / mean_disparity
            print(f"Distância estimada na região ({self.x}:{self.w + self.x}, {self.y}:{self.h + self.y}) é: {self.depth:.2f} metros")
        else:
            print("Disparidade insuficiente para calcular a distância.")

        disparity_max = np.max(self.disparity_map)
        disparity_min = np.min(self.disparity_map[self.disparity_map > 0])  # ignora valores zero ou negativos'

        # Calcular profundidade mínima e máxima
        if disparity_min > 0:  # Verifica se há valores válidos de disparidade
            self.depth_min = (self.focal_length * self.baseline) / disparity_max
            self.depth_max = (self.focal_length * self.baseline) / disparity_min
            print(f"Profundidade mínima: {self.depth_min:.2f} m")
            print(f"Profundidade máxima: {self.depth_max:.2f} m")
        else:
            print("Erro: disparidade insuficiente para calcular profundidade.")

        self.min_disparity_mask = (self.disparity_map == disparity_min)
        self.max_disparity_mask = (self.disparity_map == disparity_max)


    def plot_results(self, plot_extreme_points=False):
        """Plota os resultados do depth map"""
        if plot_extreme_points:
            # plota os pontos de maximo e minimo disparidade
            for y in range(self.disparity_map.shape[0]):
                for x in range(self.disparity_map.shape[1]):
                    if self.min_disparity_mask[y, x]:
                        # pontos de minima disparidade (maxima distancia), com cor preta
                        cv2.circle(self.disparity_normalize, (x, y), radius=5, color=(0, 0, 0), thickness=-1)
            for y in range(self.disparity_map.shape[0]):
                for x in range(self.disparity_map.shape[1]):
                    if self.max_disparity_mask[y, x]:
                        # pontos de maxima disparidade (minima distancia), com cor branca
                        cv2.circle(self.disparity_normalize, (x, y), radius=5, color=(255, 255, 255), thickness=-1)

        cv2.rectangle(self.imgL, (self.x, self.y), (self.x+self.w, self.y+self.h), (0, 0, 255), 3)
        cv2.rectangle(self.imgR, (self.x, self.y), (self.x+self.w, self.y+self.h), (0, 0, 255), 3)
        cv2.rectangle(self.disparity_normalize, (self.x, self.y), (self.x+self.w, self.y+self.h), (255, 255, 255), 3)

        plt.figure(figsize=(16, 10))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(self.imgL, cv2.COLOR_BGR2RGB))
        plt.title("Imagem Esquerda (L)")
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(self.imgR, cmap='gray')
        plt.title("Imagem Direita (R)")
        plt.axis('off')

        plt.subplot(133)
        plt.imshow((cv2.cvtColor(self.disparity_normalize, cv2.COLOR_BGR2RGB)))
        plt.title("Mapa de Disparidade")
        plt.text(10, 1000, f"Distância estimada na região é: {self.depth:.2f} metros", fontsize=12, color='blue')
        plt.text(10, 1100, f"Profundidade mínima: {self.depth_min:.2f} m", fontsize=12, color='blue')
        plt.text(10, 1200, f"Profundidade máxima: {self.depth_max:.2f} m", fontsize=12, color='blue')
        plt.text(1, 1300, f"Centro do objeto: {int(self.x + (self.w)/2), int(self.y + (self.h)/2)}", fontsize=12, color='blue')
        plt.axis('off')
        plt.show()

        


