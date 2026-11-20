"""Manos sinteticas en 3D (para probar todo sin camara).

mano(pose, x, y, lado) -> (norm(21,2), world(21,3), etiqueta, score), igual que MediaPipe.
Poses: 'abierta', 'puno', 'ok', 'relajada' y 'ok_real' (OK como lo hace la gente:
el pulgar toca la ultima falange del indice, los otros dedos algo doblados y la
profundidad de MediaPipe equivocada por 3.5 cm).
"""

import math

import numpy as np

# nudillos (MCP) y largo de las 3 falanges de indice, medio, anular, menique
MCP = {5: (0.030, -0.085), 9: (0.008, -0.090), 13: (-0.012, -0.085), 17: (-0.030, -0.075)}
FALANGES = {5: (0.040, 0.025, 0.022), 9: (0.045, 0.028, 0.024),
            13: (0.042, 0.026, 0.022), 17: (0.032, 0.020, 0.020)}
CURVA = {"abierta": (0, 0, 0, 0), "puno": (1, 1, 1, 1), "ok": (0.55, 0, 0, 0), "relajada": (0.35,) * 4,
         "ok_real": (0.6, 0.3, 0.35, 0.4)}
ESCALA_IMAGEN = 1.4          # metros -> coordenadas 0..1 (mano de ~13 % del alto)


def mano_world(pose):
    w = np.zeros((21, 3))
    for (base, (x, y)), c in zip(MCP.items(), CURVA[pose]):
        p = np.array([x, y, 0.0])
        w[base] = p
        a = 0.0
        for k, (largo, inc) in enumerate(zip(FALANGES[base], (75, 95, 65))):
            a += math.radians(inc) * c
            p = p + largo * np.array([0.0, -math.cos(a), -math.sin(a)])
            w[base + 1 + k] = p
    w[1], w[2], w[3] = (0.025, -0.020, 0), (0.045, -0.040, 0), (0.060, -0.055, 0)
    if pose == "abierta":
        w[4] = (0.070, -0.070, 0.0)
    elif pose == "puno":
        w[4] = (0.000, -0.060, -0.030)
    elif pose == "ok":
        w[4] = w[8] + np.array([0.004, 0.0, 0.0])
    elif pose == "ok_real":
        w[4] = 0.5 * (w[7] + w[8]) + np.array([0.006, 0.0, 0.035])
    else:
        w[4] = (0.050, -0.065, -0.010)
    return w


def mano(pose, x, y, lado="Right", score=0.95, ruido=0.0, rng=None):
    world = mano_world(pose)
    palma = world[[0, 5, 9, 13, 17], :2].mean(axis=0)
    norm = np.array([x, y]) + (world[:, :2] - palma) * ESCALA_IMAGEN
    if lado == "Left":                       # espejo de la geometria (no afecta las medidas)
        norm[:, 0] = 2 * x - norm[:, 0]
    if ruido and rng is not None:
        norm = norm + rng.normal(0, ruido, norm.shape)
    return norm, world, lado, score
