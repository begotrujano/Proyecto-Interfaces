"""Saturno: la esfera (como cualquier astro, ver tierra.py) mas sus anillos, dibujados por trazado de rayos.

Los anillos son un disco plano en el plano ecuatorial de Saturno (perpendicular a su polo norte). Para cada
pixel de la vista (proyeccion ortografica, como la esfera) se busca el punto del plano que se ve ahi: su
distancia al centro da el brillo y la opacidad (perfil radial: anillos D, C, B, division de Cassini, A,
hueco de Encke y F), y su posicion da:
  - si lo tapa el planeta (anillo detras del disco),
  - si cae en la sombra del planeta (cuña oscura del lado opuesto al Sol),
  - que cara ve la camara: la iluminada (brillante) o la oscura (casi negra, salvo donde es poco opaca).
La sombra de los anillos sobre el planeta se calcula igual, lanzando un rayo del planeta hacia el Sol.
Todo depende solo del polo y del Sol vistos desde la camara (no del giro: los anillos son simetricos), asi
que se cachea y solo se rehace cuando cambian (o el tamano o la posicion). Solo numpy y OpenCV.
"""

import math

import cv2
import numpy as np

from universo import config as C
from universo.tierra import Tierra

RADIO_KM = 58232.0                  # radio medio de Saturno (el mismo de sistema.FISICOS)
R_IN, R_OUT = 66900.0, 140600.0     # km desde el centro: borde interior del anillo D y exterior del anillo F
N_PERFIL = 2048
FACTOR_EXTERIOR = R_OUT / RADIO_KM  # ~2.41 radios: lo que se extienden los anillos (para acomodarlos en pantalla)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def perfil_anillos(n=N_PERFIL, semilla=3):
    """Perfil radial de los anillos: (opacidad (n,), color BGR 0..1 (n, 3)) de R_IN a R_OUT."""
    r = np.linspace(R_IN, R_OUT, n)
    rng = np.random.default_rng(semilla)
    paso = (R_OUT - R_IN) / n

    def ruido(km):                                    # ondulaciones (anillitos) de ese tamano, en ~[-1, 1]
        z = cv2.GaussianBlur(rng.normal(size=(1, n)).astype(np.float32), (0, 0), max(km / paso, 0.5))[0]
        return z / (np.abs(z).max() + 1e-9)

    fino, medio = ruido(120.0), ruido(700.0)
    caja = lambda a, b, w: _sig((r - a) / w) - _sig((r - b) / w)
    rel = lambda a, b: np.clip((r - a) / (b - a), 0.0, 1.0)

    d = caja(R_IN, 74658.0, 250.0) * 0.06
    c = caja(74658.0, 92000.0, 250.0) * (0.27 + 0.08 * medio + 0.04 * fino)
    b_ = caja(92000.0, 117580.0, 250.0) * (0.50 + 0.45 * rel(92000.0, 102000.0)) * (0.88 + 0.10 * medio + 0.06 * fino)
    b_ = b_ * (1.0 - 0.35 * np.exp(-((r - 104000.0) / 900.0) ** 2))      # un surco oscuro dentro del B
    cas = caja(117580.0, 122170.0, 200.0) * 0.07 + 0.10 * np.exp(-((r - 119900.0) / 350.0) ** 2) \
        + 0.08 * np.exp(-((r - 121000.0) / 250.0) ** 2)
    a = caja(122170.0, 136775.0, 200.0) * (0.62 + 0.07 * medio + 0.04 * fino)
    a = a * (1.0 - 0.95 * np.exp(-((r - 133589.0) / 170.0) ** 2))         # hueco de Encke
    a = a * (1.0 - 0.60 * np.exp(-((r - 136505.0) / 60.0) ** 2))          # hueco de Keeler
    f = 0.30 * np.exp(-((r - 140180.0) / 280.0) ** 2)                     # anillo F: delgadito
    alfa = np.clip(d + c + b_ + cas + a + f, 0.0, 0.97).astype(np.float32)

    bgr = lambda rr, gg, bb: np.array([bb, gg, rr], np.float32) / 255.0
    col = np.zeros((n, 3), np.float32)
    peso = np.zeros(n, np.float32)
    for w, cc in ((d, bgr(170, 160, 150)), (c, bgr(168, 152, 132)), (b_, bgr(236, 214, 182)),
                  (cas, bgr(165, 150, 135)), (a, bgr(212, 194, 165)), (f, bgr(205, 190, 170))):
        col += w[:, None].astype(np.float32) * cc
        peso += w.astype(np.float32)
    col = col / np.maximum(peso, 1e-6)[:, None]
    col = col * (0.93 + 0.07 * fino[:, None])
    return alfa, np.clip(col, 0.0, 1.0).astype(np.float32)


def polo_en_vista(inclinacion, rodar):
    """Polo norte del astro en el marco de la vista (x derecha, y abajo, z hacia ti), con la misma
    convencion de Tierra.dibujar(inclinacion, rodar): inclinacion > 0 = se ve el polo sur."""
    ci, si, cr, sr = math.cos(inclinacion), math.sin(inclinacion), math.cos(rodar), math.sin(rodar)
    return np.array([sr * ci, -cr * ci, -si])


def _suave(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Anillos:
    def __init__(self):
        alfa, col = perfil_anillos()
        self._lut = np.ascontiguousarray(np.concatenate([alfa[:, None], col], 1)[None, :, :], np.float32)
        self._anillo = None                           # (clave, x0, y0, x1, y1, inverso 3 canales, color premultiplicado)
        self._sombra = None                           # (clave, x0, y0, x1, y1, factor 3 canales)

    def _buscar(self, rho_km):
        """(opacidad, B, G, R) de cada punto del plano segun su distancia al centro (km)."""
        idx = ((rho_km - R_IN) * ((N_PERFIL - 1) / (R_OUT - R_IN))).astype(np.float32)
        return cv2.remap(self._lut, idx, np.zeros_like(idx), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    @staticmethod
    def _rejilla(x0, y0, x1, y1, cx, cy, radio, max_px):
        """Coordenadas (en radios del planeta) de una rejilla que cubre la region, a resolucion reducida
        si es muy grande: -> (X (ww,), Y (hh,))."""
        esc = min(1.0, math.sqrt(max_px / max((x1 - x0) * (y1 - y0), 1)))
        ww, hh = max(int(round((x1 - x0) * esc)), 2), max(int(round((y1 - y0) * esc)), 2)
        X = ((x0 + (np.arange(ww) + 0.5) * ((x1 - x0) / ww)) - cx) / radio
        Y = ((y0 + (np.arange(hh) + 0.5) * ((y1 - y0) / hh)) - cy) / radio
        return X.astype(np.float32), Y.astype(np.float32)

    # ------------------------------------------------------------------
    def _construir(self, clave, x0, y0, x1, y1, cx, cy, radio, polo, s):
        X, Y = self._rejilla(x0, y0, x1, y1, cx, cy, radio, C.ANILLOS_PX_MAX)
        nx, ny, nz = polo
        nzc = nz if abs(nz) > 0.012 else math.copysign(0.012, nz if nz != 0 else 1.0)   # de canto: una linea fina
        a, b = -nx / nzc, -ny / nzc
        xs, ys = X[None, :], Y[:, None]
        z = a * xs + b * ys                                       # punto del plano de los anillos bajo cada pixel
        r2 = xs * xs + ys * ys
        rho2 = r2 + z * z
        pix = self._buscar(np.sqrt(rho2) * RADIO_KM)
        alfa, col = pix[..., 0], pix[..., 1:4]
        oculto = (z < 0) & (r2 < 1.0)                             # detras del disco del planeta
        alfa = np.where(oculto, 0.0, alfa).astype(np.float32)

        sx, sy, sz = s
        nd = nx * sx + ny * sy + nz * sz                          # n.sol: >0 = el Sol esta del lado norte
        # sombra del planeta sobre los anillos: punto "detras" del planeta respecto al Sol y dentro de su cilindro
        ps = (sx + a * sz) * xs + (sy + b * sz) * ys
        perp = np.sqrt(np.maximum(rho2 - ps * ps, 0.0))
        sombra = np.where(ps < 0, _suave(0.97, 1.03, perp), 1.0).astype(np.float32)
        if nd * nz >= 0:                                          # la camara ve la cara iluminada
            luz = 0.22 + 0.78 * float(_suave(0.0, 0.30, abs(nd)))
            fac = (luz * 1.12) * sombra
        else:                                                     # cara oscura: solo brilla lo poco opaco
            fac = (0.03 + 0.50 * (1.0 - alfa) ** 2.5) * sombra
        cpre = col * (fac * alfa)[..., None] * 255.0
        a8 = np.clip(alfa * 255.0, 0, 255).astype(np.uint8)
        c8 = np.clip(cpre, 0, 255).astype(np.uint8)
        w, h = x1 - x0, y1 - y0
        if a8.shape != (h, w):
            a8 = cv2.resize(a8, (w, h), interpolation=cv2.INTER_LINEAR)
            c8 = cv2.resize(c8, (w, h), interpolation=cv2.INTER_LINEAR)
        inv = cv2.merge([255 - a8] * 3)
        self._anillo = (clave, x0, y0, x1, y1, inv, np.ascontiguousarray(c8))

    def dibujar(self, buf, cx, cy, radio, polo, sol):
        """Compone los anillos sobre buf (despues del planeta)."""
        alto, ancho = buf.shape[:2]
        E = FACTOR_EXTERIOR * radio + 2.0
        x0, y0, x1, y1 = max(int(cx - E), 0), max(int(cy - E), 0), min(int(cx + E) + 1, ancho), min(int(cy + E) + 1, alto)
        if x1 <= x0 or y1 <= y0:
            return
        s = np.asarray(sol, np.float64)
        s = s / (np.linalg.norm(s) + 1e-12)
        clave = (round(radio * 20.0), x0, y0, x1, y1, round(cx * 2.0), round(cy * 2.0),
                 tuple(np.round(polo / 0.003).astype(int)), tuple(np.round(s / 0.003).astype(int)))
        if self._anillo is None or self._anillo[0] != clave:
            self._construir(clave, x0, y0, x1, y1, cx, cy, radio, polo, s)
        _, x0, y0, x1, y1, inv, c8 = self._anillo
        roi = buf[y0:y1, x0:x1]
        roi[:] = cv2.add(cv2.multiply(roi, inv, scale=1.0 / 255.0), c8)

    def sombrear_planeta(self, buf, cx, cy, radio, polo, sol):
        """Oscurece el planeta donde le cae la sombra de los anillos (rayo hacia el Sol que cruza el plano)."""
        alto, ancho = buf.shape[:2]
        x0, y0, x1, y1 = max(int(cx - radio) - 1, 0), max(int(cy - radio) - 1, 0), \
            min(int(cx + radio) + 2, ancho), min(int(cy + radio) + 2, alto)
        if x1 <= x0 or y1 <= y0:
            return
        s = np.asarray(sol, np.float64)
        s = s / (np.linalg.norm(s) + 1e-12)
        nd = float(polo @ s)
        if abs(nd) < 0.02:                                        # el Sol casi en el plano: sombra de una linea
            return
        clave = (round(radio * 20.0), x0, y0, x1, y1, round(cx * 2.0), round(cy * 2.0),
                 tuple(np.round(polo / 0.003).astype(int)), tuple(np.round(s / 0.003).astype(int)))
        if self._sombra is None or self._sombra[0] != clave:
            X, Y = self._rejilla(x0, y0, x1, y1, cx, cy, radio, 300000)
            xs, ys = X[None, :], Y[:, None]
            r2 = xs * xs + ys * ys
            disco = r2 < 1.0
            zs = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
            qn = polo[0] * xs + polo[1] * ys + polo[2] * zs       # altura del punto sobre el plano de los anillos
            t = -qn / nd                                          # hasta cruzar el plano, camino al Sol
            qs = s[0] * xs + s[1] * ys + s[2] * zs
            rho = np.sqrt(np.maximum(1.0 + 2.0 * t * qs + t * t, 0.0))
            alfa = self._buscar(rho * RADIO_KM)[..., 0]
            f = 1.0 - 0.9 * np.where(disco & (t > 0), alfa, 0.0)
            f = cv2.GaussianBlur(f.astype(np.float32), (0, 0), 0.8)
            f8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
            w, h = x1 - x0, y1 - y0
            if f8.shape != (h, w):
                f8 = cv2.resize(f8, (w, h), interpolation=cv2.INTER_LINEAR)
            self._sombra = (clave, x0, y0, x1, y1, cv2.merge([f8] * 3))
        _, x0, y0, x1, y1, f3 = self._sombra
        roi = buf[y0:y1, x0:x1]
        roi[:] = cv2.multiply(roi, f3, scale=1.0 / 255.0)


class Saturno(Tierra):
    """Saturno: la esfera de Tierra y encima sus anillos."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._anillos = Anillos()

    def dibujar(self, buf, cx, cy, radio, giro, inclinacion=0.0, sol=(-0.62, -0.38, 0.69), sol_tierra=None,
                rodar=0.0, luz_vista=False):
        super().dibujar(buf, cx, cy, radio, giro, inclinacion, sol, sol_tierra, rodar, luz_vista)
        if radio < 3.0:
            return
        radio = round(radio * 20.0) / 20.0
        polo = polo_en_vista(inclinacion, rodar)
        self._anillos.sombrear_planeta(buf, cx, cy, radio, polo, sol)
        self._anillos.dibujar(buf, cx, cy, radio, polo, sol)
