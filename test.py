from geopy.distance import distance
from geopy.point import Point
import math

# Coordenadas iniciais (latitude, longitude) em graus
latitude = 45.54761966523571  # Latitude do ponto inicial
longitude = -13.705387316093407  # Longitude do ponto inicial

# Distância e heading
dist = 15  # Distância em metros
heading = 300  # Heading em graus (0 = Norte, 90 = Leste, etc.)

earth_radius = 6371000  # raio da Terra em metros

# Cálculo de deslocamento
lat_rad = math.radians(latitude)
lon_rad = math.radians(longitude)
bearing_rad = math.radians(heading)

delta_lat = dist * math.cos(bearing_rad) / earth_radius
delta_lon = dist * math.sin(bearing_rad) / (earth_radius * math.cos(lat_rad))

new_lat = latitude + math.degrees(delta_lat)
new_lon = longitude + math.degrees(delta_lon)

print(f"Nova coordenada Fórmula Direta: ({new_lat}, {new_lon})")
