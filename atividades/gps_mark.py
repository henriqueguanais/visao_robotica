from pyproj import Proj, Transformer
import numpy as np

class GPSMarker:
    '''
    Classe para obter as coordenadas de um objeto que está sendo detectado, em relação ao barco
    '''
    def __init__(self, magnetic_declination:float) -> None:
        """
        Parâmetros:
        -------------------
        magnetic_declination: float
            Declinação magnética do local em que o barco está
        
        Exemplo de uso:

        >>> from gps_mark import GPSMarker
        >>> import cv2
        >>> imgL = cv2.imread('imgL.png')
        >>> magnetic_declination = 4.41
        >>> focal_length = 500
        >>> gps_path = 'gps.txt'
        >>> imu_path = 'imu.txt'
        >>> center_x = 500
        >>> distance = 15
        >>> gps_marker = GPSMarker(magnetic_declination)
        >>> gps_marker.angle_object(center_x, distance, imgL, focal_length)
        >>> obj_latitude, obj_longitude = gps_marker.gps_mark(gps_path, imu_path, utm_zone=33)
    
        """
        self.magnetic_declination = magnetic_declination


    def get_gps_coords(self, gps_txt_path:str) -> tuple:
        '''
        Obtém as coordenadas UTM do barco.

        Parâmetros:
        ------------------
        gps_txt_path: str
            Caminho do arquivo de texto com as coordenadas UTM do barco

        Retorna:
        ------------------
            Uma tupla com as coordenadas UTM do barco
        '''
        try:
            with open(gps_txt_path, 'r') as f:
                data = f.readlines()
                x = float(data[0][0:-2])
                y = float(data[1][0:-2])
            return (x, y)
        except (FileNotFoundError, ValueError, IndexError) as e:
            raise ValueError(f"Erro ao processar o arquivo {gps_txt_path}: {e}")
        
        
    def gps_transform(self, utm_coords:tuple, utm_zone:int, south_hemisphere:bool = False) -> list: 
        '''
        Transforma as coordenadas UTM para WGS84.

        Parâmetros:
        --------------
        utm_coords: tuple
            Uma tupla com as coordenadas UTM do barco
        utm_zone: int
            Zona UTM onde as coordenadas estão
        south_hemisphere: bool
            Se as coordenadas estão no hemisfério sul

        Retorna:
        ----------
        boat_coords: list
            Uma lista com as coordenadas WGS84 do barco
        '''
        utm_proj = Proj(proj="utm", zone=utm_zone, ellps="WGS84", south=south_hemisphere)
        wgs84_proj = Proj(proj="latlong", datum="WGS84")
        transformer = Transformer.from_proj(utm_proj, wgs84_proj)
        longitude, latitude = transformer.transform(utm_coords[0], utm_coords[1])

        self.boat_coords = [latitude, longitude]
    
        return self.boat_coords
    

    def get_imu_values(self, imu_txt_path:str) -> list:
        '''
        Obtém os valores do IMU.

        Parâmetros:
        ------------------
        imu_txt_path: str
            Caminho do arquivo de texto com os valores do IMU
        
        Retorna:
        ------------------
        imu: list
            Uma lista com os valores do IMU
        '''
        try:
            with open(imu_txt_path, 'r') as f:
                data = f.readlines()
                x = float(data[0][0:-2])
                y = float(data[1][0:-2])
                z = float(data[2][0:-2])
                self.imu = [x, y, z]
            return self.imu
        except (FileNotFoundError, ValueError, IndexError) as e:
            raise ValueError(f"Erro ao processar o arquivo {imu_txt_path}: {e}")
        

    def angle_object(self, center_ox:int, distance:float, img:np.ndarray, focal_length:float) -> None:
        '''
        Calcula o ângulo do objeto em relação ao centro da imagem (barco).

        Parâmetros:
        -------------------
        center_ox: int
            Coordenada x do centro do objeto
        distance: float
            Distância do objeto em relação ao barco
        img: np.ndarray
            Imagem onde o objeto foi detectado
        focal_length: float
            Distância focal da câmera
        
        Retorna:
        -------------------
        new_angle: float
            O ângulo do objeto em relação ao barco, em graus
        '''
        self.distance = distance
        width = img.shape[1]

        # distancia do centro x do objeto ao centro da imagem em pixels
        distance_xcenter2object = center_ox - width/2
        # calculo da distancia do centro x do objeto ao centro da imagem em metros
        distance_in_meters = distance_xcenter2object *self.distance/focal_length
        # angulo do centro do objeto ate o barco em radianos
        angle_o2c = np.arctan(distance_in_meters/self.distance)
        
        self.new_angle = np.degrees(angle_o2c)
        
        
    def gps_mark(self, gps_txt_path:str, imu_txt_path:str, utm_zone:int, south_hemisphere:bool = False) -> list:
        '''
        Calcula as coordenadas do objeto em relação ao barco.
        
        Parâmetros:
        -------------------
        gps_txt_path: str
            Caminho do arquivo de texto com as coordenadas UTM do barco
        imu_txt_path: str
            Caminho do arquivo de texto com os valores do IMU
        utm_zone: int
            Zona UTM onde as coordenadas estão
        south_hemisphere: bool
            Se as coordenadas estão no hemisfério sul
        
        Retorna:
        -------------------
        obj_coords: list
            Uma lista com as coordenadas do objeto em relação ao barco
        '''
        gps_coords = self.get_gps_coords(gps_txt_path)
        gps_coords = self.gps_transform(gps_coords, utm_zone, south_hemisphere)
        imu_values = self.get_imu_values(imu_txt_path)

        # calcula o heading do barco, com base nos valores do imu
        magnetic_heading = -np.arctan2(imu_values[1], imu_values[0])
        magnetic_heading = np.degrees(magnetic_heading)
        
        # ângulo do barco, considerando o heading magnético, declinação magnética e o ângulo do objeto
        true_heading = magnetic_heading + self.magnetic_declination + self.new_angle
        
        print(f"Angulo em relacao ao norte: {true_heading}º")
        theta = np.deg2rad(true_heading)
        # calcula o deslocamento do objeto em relação ao barco, em metros
        delta_x = self.distance * np.sin(theta)
        delta_y = self.distance * np.cos(theta)

        # conversao de metros para graus de latitude, 1 grau de latitude corresponde a 111111 metros
        delta_latitude = delta_y / 111111
        # conversao de metros para graus de longitude, 1 grau de longitude corresponde a 111111 metros * cos(latitude)
        delta_longitude = delta_x / (111111 * np.cos(np.deg2rad(gps_coords[0])))
        
        self.obj_longitude = gps_coords[1] + delta_longitude
        self.obj_latitude = gps_coords[0] + delta_latitude
        self.obj_coords = [self.obj_latitude, self.obj_longitude]
      
        return self.obj_coords

