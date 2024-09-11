import cv2 as cv
import numpy as np
import argparse
from math import sqrt
from os import listdir
from utils.showresults import showKeypoints
from matcher import matchWithRatioTest
import networkx as nx

"""
Clase Imagen

Almacena el nombre de la imagen, keypoints, descriptores y un diccionario con los emparejamientos que tiene.
"""


class Image:
    def __init__(self, img_name, image, keypoints, descriptors):
        self.img_name = img_name
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.matches = {}

    def __hash__(self):
        return hash(self.img_name)

    def __str__(self):
        return f"Imagen: {self.img_name} \n # Keypoints: {len(self.keypoints)} \n # Descriptores: {self.descriptors.shape} \n Matches: {list(self.matches.keys())} \n"


"""
Clase ImageFeature
Almacena la referencia a una imagen (nombre del archivo) junto con un índice del keypoint específico a esa imagen
"""


class ImageFeature:
    def __init__(self, img_name, kpt_idx):
        self.img_name = img_name
        self.kpt_idx = kpt_idx

    def __hash__(self):
        return hash(self.img_name + str(self.kpt_idx))

    def __str__(self):
        return f"{self.img_name}, keypoint: {self.kpt_idx}"


PAIR_MATCH_SURVIVAL_RATE = 0.5

parser = argparse.ArgumentParser(
    description="Codigo para realizar SfM (Structure from Motion)"
)
parser.add_argument(
    "--images_folder",
    help="Path de la carpeta de imágenes",
    default="./images",
)
args = parser.parse_args()

# Diccionario de imágenes
images = {}

# Carpeta de las imagenes
images_folder = args.images_folder

# Leer todos los archivos dentro de la carpeta
images_files = listdir(images_folder)

# Inicializar descriptor AKAZE
akaze = cv.AKAZE_create()
# Almacenamos todos los keypoints y descriptors que encontramos
keypoints = []
descriptors = []
for img in images_files:
    img_g = cv.imread(images_folder + "/" + img, cv.IMREAD_GRAYSCALE)
    if img_g is None:
        print("No se pudo abrir la imagen " + img)
        exit(0)

    # Inicializar el descriptor AKAZE y detectar keypoints y extraer los descriptores de la imagen.
    # kpts -> keypoints
    # descs -> descriptores
    kpts, descs = akaze.detectAndCompute(img_g, None)
    print("keypoints: {}, descriptors: {}".format(len(kpts), descs.shape))

    # Crear la clase imagen con sus keypoints y descriptors
    # Se incluye en el diccionario de imágenes
    images[img] = Image(img, img_g, kpts, descs)

    # Mostrar resultados de la extracción
    # showKeypoints(img_g, kpts)

# Fase de Matching

# Crear matches para todas las imágenes
# Utilizamos distancia de Hamming porque AKAZE es un descriptor binario
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

for i, img_i in enumerate(list(images.values())):
    for j, img_j in enumerate(list(images.values())[i + 1 : :]):
        # Encontrar los match
        matches = matchWithRatioTest(
            matcher, img_i.descriptors, img_j.descriptors
        )

        # Test de reciprocidad
        matchRcp = matchWithRatioTest(
            matcher, img_j.descriptors, img_i.descriptors
        )

        merged_matches = []

        for mR_m in matchRcp:
            found = False
            for m_m in matches:
                # Solo se acepta el match si es recíproco
                if (
                    mR_m.queryIdx == m_m.trainIdx
                    and mR_m.trainIdx == m_m.queryIdx
                ):
                    merged_matches.append(m_m)
                    found = True
                    break
                if found:
                    continue

        # Restricción Epipolar. Calculamos la matriz fundamental utilizando RANSAC
        img_i_pts = []
        img_j_pts = []

        for m in merged_matches:
            img_i_pts.append(img_i.keypoints[m.queryIdx].pt)
            img_j_pts.append(img_j.keypoints[m.trainIdx].pt)

        F, inliersMask = cv.findFundamentalMat(
            np.array([img_i_pts]), np.array([img_j_pts]), cv.FM_RANSAC
        )

        final = []

        for mask, match in zip(inliersMask, merged_matches):
            if mask:
                final.append(match)

        if float(len(final)) / float(len(matches)) < PAIR_MATCH_SURVIVAL_RATE:
            print(
                "El match final tiene menos de la mitad de matches que el original. Ignorando imagen."
            )
            continue

        images[img_i.img_name].matches[
            (img_i.img_name, img_j.img_name)
        ] = final

# Crear grafo de características para rastrearlas

vertexByImageFeature = {}

G = nx.Graph()

# Agregamos nodos con las imágenes
for img_tpl in images.values():
    for i in range(len(img_tpl.keypoints)):
        node = ImageFeature(img_tpl.img_name, i)
        G.add_node(node)
        vertexByImageFeature[(img_tpl.img_name, i)] = node

# Agregamos las aristas (matches)
for img_tpl in images.values():
    for matches in img_tpl.matches:
        for m in img_tpl.matches[matches]:
            v1 = vertexByImageFeature[(matches[0], m.queryIdx)]
            v2 = vertexByImageFeature[(matches[1], m.trainIdx)]
            G.add_edge(v1, v2)

# Eliminar nodos sin aristas
G.remove_nodes_from(list(nx.isolates(G)))

# Obtener los componentes conectados (retorna una lista de sets con nodos (componentes))
components = nx.connected_components(G)
goodComponents = list()
# Filtramos los componentes (más de 1 característica por imagen)
for c in components:
    isGoodComponent = True
    imagesInComponent = set()
    for img in c:
        img_name = img.img_name
        if img_name in imagesInComponent:
            # La imagen ya se encuentra en el componente
            isGoodComponent = False
            break
        else:
            imagesInComponent.add(img_name)
    if isGoodComponent:
        goodComponents.append(c)

print(f"Total de componentes: {nx.number_connected_components(G)}")
print(f"Componentes buenos: {len(goodComponents)}")
