"""Tierra 3D que gira sobre su eje (como globo terraqueo), dibujada solo con OpenCV.

Como se hace rapido:
  - Una "tabla" precalculada dice, para cada punto del disco, que latitud/longitud
    se ve ahi. Girar la Tierra = sumar un desplazamiento a la longitud.
  - Cada cuadro: 1 warpAffine de la tabla (escala/posicion) + 1 cv2.transform
    (aplica el giro) + 1 cv2.remap de la textura. Todo en C, nada en Python puro.
  - La iluminacion (dia/noche, brillo del mar, atmosfera) tambien esta precalculada.
  - Piramides (mipmaps) de textura y tablas: sin parpadeo al alejar y el costo es
    proporcional al tamano en pantalla.
"""

import math
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
ATM = np.array([1.0, 0.66, 0.36], np.float32)      # BGR, azul atmosfera
# tabla de color: brillo (0..255) -> azul de la atmosfera (mucho mas rapido que multiplicar)
LUT_ATM = np.clip(np.arange(256)[:, None] * ATM, 0, 255).astype(np.uint8).reshape(1, 256, 3)
ESCALA_MUL = 200.0                                    # la luz se guarda x200 para permitir brillo > 1


def _suave(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------
# Textura
# ----------------------------------------------------------------------
def _ruido_esfera(n, rng, octavas, f0, ondas):
    total = np.zeros(len(n), np.float32)
    amp, f, var = 1.0, f0, 0.0
    for _ in range(octavas):
        for _ in range(ondas):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            total += amp * np.sin(f * (n @ d).astype(np.float32) + rng.uniform(0, 2 * math.pi))
            var += 0.5 * amp * amp
        f *= 2.0
        amp *= 0.5
    return total / math.sqrt(var)


def textura_procedural(ancho=1024, semilla=7):
    """Planeta inventado (solo si falta la textura real)."""
    alto = ancho // 2
    rng = np.random.default_rng(semilla)
    lon = (np.arange(ancho) + 0.5) / ancho * 2 * math.pi - math.pi
    lat = math.pi / 2 - (np.arange(alto) + 0.5) / alto * math.pi
    LON, LAT = np.meshgrid(lon, lat)
    n = np.stack([np.cos(LAT) * np.sin(LON), -np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    n = n.reshape(-1, 3).astype(np.float32)
    alt = _ruido_esfera(n, rng, 5, 3.0, 12)
    tierra = _suave(0.25, 0.35, alt)[:, None]
    oceano = np.array([0.40, 0.17, 0.04]) * (1 - _suave(-0.3, 0.3, alt)[:, None]) + \
        np.array([0.60, 0.34, 0.08]) * _suave(-0.3, 0.3, alt)[:, None]
    suelo = np.array([0.18, 0.40, 0.20]) * (1 - _suave(0.6, 1.2, alt)[:, None]) + \
        np.array([0.30, 0.45, 0.55]) * _suave(0.6, 1.2, alt)[:, None]
    col = oceano * (1 - tierra) + suelo * tierra
    hielo = _suave(0.86, 0.95, np.abs(np.sin(LAT)).reshape(-1) + 0.03 * alt)[:, None]
    col = col * (1 - hielo) + 0.95 * hielo
    return (np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(alto, ancho, 3)


def cargar_textura(ruta):
    """Equirectangular BGR (ancho = 2 x alto). Si no existe, procedural."""
    if ruta:
        p = Path(ruta)
        p = p if p.is_absolute() else AQUI / p
        if p.is_file():
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)  # rutas con acentos
            if img is not None:
                return img, p.name
    return textura_procedural(), "procedural"


def mascara_oceano(bgr):
    """Donde hay mar (para el reflejo del sol). Mar = azul dominante y oscuro."""
    f = bgr.astype(np.float32)
    dif = f[..., 0] - np.maximum(f[..., 1], f[..., 2])
    lum = f.sum(-1)
    m = _suave(2.0, 12.0, dif) * (1.0 - _suave(330.0, 480.0, lum))
    sigma = max(bgr.shape[1] / 1000.0, 1.0)              # suave: el mar oscuro de la foto tiene bloques JPEG
    return (cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) * 255).astype(np.uint8)


def piramide_textura(bgr, ancho_min=128):
    """Niveles BGRA (A = mar), reduciendo a la mitad con envoltura horizontal."""
    if bgr.shape[1] != 2 * bgr.shape[0]:
        bgr = cv2.resize(bgr, (2 * bgr.shape[0], bgr.shape[0]), interpolation=cv2.INTER_AREA)
    nivel = np.ascontiguousarray(np.dstack([bgr, mascara_oceano(bgr)]))
    niveles = [nivel]
    while nivel.shape[1] // 2 >= ancho_min:
        pad = 4
        ext = np.concatenate([nivel[:, -pad:], nivel, nivel[:, :pad]], axis=1)  # sin costura en 180 grados
        red = cv2.pyrDown(ext)
        nivel = np.ascontiguousarray(red[:, pad // 2: pad // 2 + nivel.shape[1] // 2])
        niveles.append(nivel)
    return niveles                                        # de mayor a menor


# ----------------------------------------------------------------------
# Tablas de la esfera e iluminacion (en el plano de la imagen)
# ----------------------------------------------------------------------
def tablas_esfera(R=600, margen=1.18):
    S = 2 * int(math.ceil(margen * R)) + 2
    c = (S - 1) / 2.0
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    u, v = (xs - c) / R, (ys - c) / R
    r = np.sqrt(u * u + v * v)

    # --- Coordenadas de textura (fuera del disco se prolonga el borde) ---
    rc = np.minimum(r, 0.9995)
    k = np.where(r > 1e-9, rc / np.maximum(r, 1e-9), 1.0)
    uc, vc = u * k, v * k
    zc = np.sqrt(np.clip(1.0 - uc * uc - vc * vc, 0.0, 1.0))
    lon = np.arctan2(uc, zc)                              # este a la derecha
    lat = np.arcsin(np.clip(-vc, -1.0, 1.0))              # norte arriba
    tabla = np.dstack([lon / (2 * math.pi) + 0.5,
                       np.clip(0.5 - lat / math.pi, 0.0005, 0.9995)]).astype(np.float32)

    # --- Iluminacion: sol arriba a la izquierda, un poco de frente ---
    luz = np.array([-0.62, -0.38, 0.69], np.float32)
    luz /= np.linalg.norm(luz)
    z = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    ndl = u * luz[0] + v * luz[1] + z * luz[2]
    dia = _suave(-0.10, 0.45, ndl)                         # terminador suave
    mul = 0.07 + 1.12 * dia
    h = luz + np.array([0, 0, 1], np.float32)
    h /= np.linalg.norm(h)
    spec = np.clip(u * h[0] + v * h[1] + z * h[2], 0, 1) ** 40 * 0.30 * dia
    borde = (1.0 - z) ** 2.4
    brillo_borde = borde * (0.15 + 0.85 * dia) * 0.95

    a_disco = np.clip((1.0 - r) * R + 0.5, 0.0, 1.0)
    fuera = r > 1.0
    halo = np.exp(-(r - 1.0) / 0.03) * fuera
    dir_luz = np.clip(((u * luz[0] + v * luz[1]) / (r + 1e-6)) * 0.5 + 0.5, 0.0, 1.0)
    halo = halo * (0.15 + 0.85 * dir_luz ** 1.5) * 0.85
    halo *= np.clip((margen - 0.02 - r) / 0.06, 0.0, 1.0)
    halo *= 1.0 - a_disco
    alfa = np.clip(a_disco + halo, 0.0, 1.0)

    # 4 canales: luz (x200), reflejo del sol, alfa, brillo de atmosfera (se colorea al dibujar)
    luz = np.dstack([np.clip(mul * a_disco * ESCALA_MUL, 0, 255),
                     np.clip(spec * a_disco * 255, 0, 255),
                     alfa * 255,
                     np.clip((a_disco * brillo_borde + halo) * 255, 0, 255)])
    return tabla, (luz + 0.5).astype(np.uint8), R


def piramide_tablas(tabla, luz, R, paso=0.7071, radio_min=6.0):
    S = tabla.shape[0]
    niveles, f = [], 1.0
    while R * f >= radio_min:
        lado = max(int(round(S * f)), 8)
        if lado == S:
            t, l = tabla, luz
        else:
            t = cv2.resize(tabla, (lado, lado), interpolation=cv2.INTER_AREA)
            l = cv2.resize(luz, (lado, lado), interpolation=cv2.INTER_AREA)
        niveles.append((np.ascontiguousarray(t), np.ascontiguousarray(l), R * lado / S))
        f *= paso
    return niveles


# ----------------------------------------------------------------------
class Tierra:
    def __init__(self, ruta_textura=None, R=600):
        bgr, self.origen = cargar_textura(ruta_textura)
        self.texturas = piramide_textura(bgr)
        self.niveles = piramide_tablas(*tablas_esfera(R))
        self._cache = None

    def dibujar(self, buf, cx, cy, radio, giro):
        """Compone la Tierra sobre buf (BGR uint8) centrada en (cx, cy).
        giro en radianes: positivo = la superficie de enfrente se mueve a la derecha."""
        if radio < 1.0:
            return
        alto, ancho = buf.shape[:2]

        radio = round(radio * 20.0) / 20.0               # permite reutilizar la luz entre cuadros
        clave = (radio, round(cx * 20.0), round(cy * 20.0), ancho, alto)
        if self._cache is None or self._cache[0] != clave:
            self._cache = (clave,) + self._preparar(radio, cx, cy, ancho, alto)
        _, region, t, mul3, spec, inv3, atm3 = self._cache
        if region is None:
            return
        x0, y0, x1, y1 = region

        # textura: ~1 texel por pixel en el ecuador
        tex = self.texturas[0]
        for cand in self.texturas:
            if cand.shape[1] >= 2 * math.pi * radio:
                tex = cand
            else:
                break
        Ht, Wt = tex.shape[:2]
        desplaz = -(giro % (2 * math.pi)) / (2 * math.pi) * Wt
        mapas = cv2.transform(t, np.array([[Wt, 0, desplaz - 0.5], [0, Ht, -0.5]], np.float32))
        muestra = cv2.remap(tex, mapas, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        color = cv2.multiply(cv2.cvtColor(muestra, cv2.COLOR_BGRA2BGR), mul3, scale=1.0 / ESCALA_MUL)
        reflejo = cv2.multiply(spec, cv2.extractChannel(muestra, 3), scale=1.0 / 255.0)
        color = cv2.add(cv2.add(color, cv2.cvtColor(reflejo, cv2.COLOR_GRAY2BGR)), atm3)
        roi = buf[y0:y1, x0:x1]
        buf[y0:y1, x0:x1] = cv2.add(cv2.multiply(roi, inv3, scale=1.0 / 255.0), color)

    def _preparar(self, radio, cx, cy, ancho, alto):
        """Lo que solo depende del tamano y la posicion (no del giro): se calcula
        una vez y se reutiliza mientras la Tierra solo gira."""
        tabla, luz, Rl = self.niveles[0]
        for niv in self.niveles:                          # el mas chico que aun alcanza
            if niv[2] >= radio:
                tabla, luz, Rl = niv
            else:
                break
        s = radio / Rl
        lado = tabla.shape[0]
        c = (lado - 1) / 2.0
        mitad = s * lado / 2.0 + 2.0
        x0, x1 = max(int(cx - mitad), 0), min(int(cx + mitad) + 1, ancho)
        y0, y1 = max(int(cy - mitad), 0), min(int(cy + mitad) + 1, alto)
        if x1 <= x0 or y1 <= y0:
            return (None,) * 6
        w, h = x1 - x0, y1 - y0
        M = np.array([[s, 0, cx - s * c - x0], [0, s, cy - s * c - y0]], np.float32)
        t = cv2.warpAffine(tabla, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        l = cv2.warpAffine(luz, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        mul, spec, alfa, brillo_atm = cv2.split(l)
        return ((x0, y0, x1, y1), t, cv2.cvtColor(mul, cv2.COLOR_GRAY2BGR), spec,
                cv2.cvtColor(255 - alfa, cv2.COLOR_GRAY2BGR),
                cv2.LUT(cv2.cvtColor(brillo_atm, cv2.COLOR_GRAY2BGR), LUT_ATM))
