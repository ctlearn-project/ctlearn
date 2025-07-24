import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas(img):
    """Detecta líneas rectas usando Canny + Hough."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=120, maxLineGap=20)
    return lines

def dibujar_lineas(img, lines):
    img_lines = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines, (x1, y1), (x2, y2), (0,255,0), 2)
    return img_lines

def calcular_puntos_de_fuga(lines):
    """Calcula puntos de fuga aproximados por agrupación de dirección."""
    puntos = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        puntos.append(((x1, y1), (x2, y2)))
    
    # Calcular las intersecciones de todas las líneas
    def intersection(line1, line2):
        (x1,y1), (x2,y2) = line1
        (x3,y3), (x4,y4) = line2
        denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if denom == 0:
            return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/denom
        return [px, py]
    
    intersections = []
    for i in range(len(puntos)):
        for j in range(i+1, len(puntos)):
            pt = intersection(puntos[i], puntos[j])
            if pt is not None and all(0 <= c <= 1500 for c in pt):  # dentro de rango razonable
                intersections.append(pt)
    return np.array(intersections)

def estimar_centro_optico_y_focal(intersections, img_shape):
    """Estima el centro óptico y focal a partir de puntos de fuga."""
    if len(intersections) < 2:
        return (img_shape[1]//2, img_shape[0]//2), 900

    # Promedia las intersecciones para estimar el centro óptico
    intersections = np.array(intersections)
    cx = np.median(intersections[:,0])
    cy = np.median(intersections[:,1])
    
    # Estimación simple de la focal como distancia al centro de la imagen
    f_est = np.mean(np.linalg.norm(intersections - np.array([[cx, cy]]), axis=1))
    return (cx, cy), f_est

# --- MAIN ---
img_path = './calibration/image_2.png'  # Usa el nombre de tu imagen
img = cv2.imread(img_path)

lines = detectar_lineas(img)
img_with_lines = dibujar_lineas(img, lines)

plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Líneas detectadas')
plt.show()

intersections = calcular_puntos_de_fuga(lines)
(cx, cy), f_est = estimar_centro_optico_y_focal(intersections, img.shape)

print("Centro óptico estimado: (%.1f, %.1f)" % (cx, cy))
print("Distancia focal estimada (en píxeles): %.1f" % f_est)

# Matriz intrínseca
K = np.array([
    [f_est, 0, cx],
    [0, f_est, cy],
    [0, 0, 1]
])
print("Matriz intrínseca estimada:\n", K)
