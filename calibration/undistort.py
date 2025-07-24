import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import minimize

# 1. Selección interactiva de puntos con matplotlib
def select_lines_points(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Haz click para seleccionar puntos de una línea recta.\nPresiona ENTER para terminar la línea. ESC para terminar todas.")
    lines_pts = []
    curr_pts = []

    def onclick(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            curr_pts.append([x, y])
            plt.plot(x, y, 'ro')
            plt.draw()

    def onkey(event):
        if event.key == 'enter':
            if len(curr_pts) >= 2:
                lines_pts.append(np.array(curr_pts))
                plt.plot(np.array(curr_pts)[:,0], np.array(curr_pts)[:,1], 'g-')
                plt.draw()
                curr_pts.clear()
        elif event.key == 'escape':
            plt.close()

    fig = plt.gcf()
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    return lines_pts

# 2. Funciones para ajuste de distorsión
def undistort_points(points, k, cx, cy):
    undistorted = []
    for x, y in points:
        xd = x - cx
        yd = y - cy
        r2 = xd**2 + yd**2
        factor = 1 + k[0]*r2 + k[1]*r2**2 + k[2]*r2**3 + k[3]*r2**4
        xu = cx + xd * factor
        yu = cy + yd * factor
        undistorted.append([xu, yu])
    return np.array(undistorted)

def line_straightness_error(k, lines_pts, cx, cy):
    total_error = 0
    for pts in lines_pts:
        pts_ud = undistort_points(pts, k, cx, cy)
        # Ajuste de recta a los puntos no distorsionados
        [vx, vy, x0, y0] = cv2.fitLine(pts_ud.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        # Distancia de cada punto a la recta
        dists = np.abs(vy*(pts_ud[:,0]-x0) - vx*(pts_ud[:,1]-y0))
        total_error += np.sum(dists**2)
    return total_error

# --- MAIN ---
# Cambia esto por la ruta de tu imagen
img_path = './calibration/image_2.png'
img = cv2.imread(img_path)
img_h, img_w = img.shape[:2]
cx, cy = img_w / 2, img_h / 2  # Centro óptico estimado

# Paso 1: Selección interactiva de puntos
lines_pts = select_lines_points(img)
if len(lines_pts) < 1:
    print("¡Debes seleccionar al menos una línea!")
    exit()

# Paso 2: Ajuste de coeficientes de distorsión radial
k_init = np.array([-0.01, 0, 0, 0])

res = minimize(line_straightness_error, k_init, args=(lines_pts, cx, cy), method='Powell')
k_opt = res.x

print("Coeficientes de distorsión radial estimados:")
print("k1=%.6f, k2=%.6f, k3=%.6f, k4=%.6f" % tuple(k_opt))

# Paso 3: Visualiza el resultado
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
colors = ['r', 'g', 'b', 'c', 'm']
for i, pts in enumerate(lines_pts):
    pts_ud = undistort_points(pts, k_opt, cx, cy)
    plt.plot(pts[:,0], pts[:,1], colors[i%len(colors)]+'o-', label=f'Línea {i+1} original')
    plt.plot(pts_ud[:,0], pts_ud[:,1], colors[i%len(colors)]+'x--', label=f'Línea {i+1} corregida')
plt.title('Puntos originales y corregidos')
plt.legend()
plt.show()
