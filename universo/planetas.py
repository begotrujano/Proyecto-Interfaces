"""Texturas de respaldo de Mercurio, Venus, Marte y el Sol, generadas con numpy/OpenCV.

Sirven mientras no pongas las imagenes reales de la NASA en assets/ (mira
herramientas/descargar_texturas.py). Marte lleva sus rasgos reales en su sitio (Syrtis Major,
Valles Marineris, Olympus Mons, Hellas, Solis Lacus, los casquetes polares...); Mercurio, sus
cuencas (Caloris, Rembrandt, Beethoven...) y sus crateres con rayos (Hokusai, Kuiper...); Venus,
sus nubes en bandas con la "Y" oscura y los collares polares; el Sol, granulacion y manchas. Son equirectangulares (longitud 0 al centro, este a la derecha), igual que las de la NASA.

Generar los archivos (una sola vez):   python -m universo.planetas   (o solo uno: ... planetas mercurio)
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

_TABLA = None


def _tabla(n=128, semilla=11):
    global _TABLA
    if _TABLA is None:
        _TABLA = np.random.default_rng(semilla).uniform(-1.0, 1.0, (n, n, n)).astype(np.float32)
    return _TABLA


def _ruido3d(p):
    """Ruido de valor 3D suave (trilineal) en [-1, 1] para puntos p (N, 3)."""
    t = _tabla()
    m = t.shape[0] - 1
    i = np.floor(p).astype(np.int32)
    f = (p - i).astype(np.float32)
    u = f * f * (3.0 - 2.0 * f)
    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]

    def g(dx, dy, dz):
        return t[(ix + dx) & m, (iy + dy) & m, (iz + dz) & m]

    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    x00 = g(0, 0, 0) + (g(1, 0, 0) - g(0, 0, 0)) * ux
    x10 = g(0, 1, 0) + (g(1, 1, 0) - g(0, 1, 0)) * ux
    x01 = g(0, 0, 1) + (g(1, 0, 1) - g(0, 0, 1)) * ux
    x11 = g(0, 1, 1) + (g(1, 1, 1) - g(0, 1, 1)) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return y0 + (y1 - y0) * uz


def _fbm(n, f0, octavas, aspereza=0.5, desplaza=0.0):
    """Ruido fractal en la esfera (n: vectores unitarios (N, 3)). Devuelve ~[-1, 1]."""
    total, amp, f, norma = np.zeros(len(n), np.float32), 1.0, f0, 0.0
    for k in range(octavas):
        total += amp * _ruido3d(n * f + (17.3 * (k + 1) + desplaza))
        norma += amp
        amp *= aspereza
        f *= 2.03
    return total / norma


def _normales(ancho, alto, fila0, fila1):
    lon = ((np.arange(ancho) + 0.5) / ancho * 2 * math.pi - math.pi).astype(np.float32)
    lat = (math.pi / 2 - (np.arange(fila0, fila1) + 0.5) / alto * math.pi).astype(np.float32)
    LON, LAT = np.meshgrid(lon, lat)
    n = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    return n.reshape(-1, 3), np.degrees(LAT), np.degrees(LON)


def _por_filas(ancho, alto, funcion, bloque=128):
    """Aplica funcion(n, lat, lon) -> (k,) por bloques de filas y arma el mapa (alto, ancho)."""
    salida = np.zeros((alto, ancho), np.float32)
    for f0 in range(0, alto, bloque):
        f1 = min(f0 + bloque, alto)
        n, lat, lon = _normales(ancho, alto, f0, f1)
        salida[f0:f1] = funcion(n, lat, lon).reshape(f1 - f0, ancho)
    return salida


def _suave(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist_lon(lon, lon0):
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _mancha(lat, lon, lat0, lon0, rlat, rlon, alabeo=None):
    """Mancha elíptica (en grados) con borde suave; alabeo la deforma como una nube."""
    dx = _dist_lon(lon, lon0) * max(math.cos(math.radians(lat0)), 0.2) / rlon
    dy = (lat - lat0) / rlat
    q = dx * dx + dy * dy
    if alabeo is not None:
        q = q * (1.0 + 0.55 * alabeo)
    return np.exp(-q)


# ----------------------------------------------------------------------
# Marte
# ----------------------------------------------------------------------
# (nombre, latitud, longitud este, radio_lat, radio_lon, fuerza): + oscurece, - aclara
_MARTE_ALBEDO = [
    ("Syrtis Major", 9, 69, 13, 8, 0.62), ("Sinus Meridiani", -2, -2, 6, 14, 0.40),
    ("Sinus Sabaeus", -8, 18, 5, 20, 0.45), ("Margaritifer", -12, -25, 7, 10, 0.36),
    ("Mare Erythraeum", -25, -40, 9, 12, 0.42), ("Aurorae Sinus", -12, -50, 5, 8, 0.40),
    ("Solis Lacus", -26, -88, 6, 10, 0.52), ("Mare Sirenum", -35, -150, 7, 24, 0.46),
    ("Mare Cimmerium", -25, 145, 8, 26, 0.46), ("Tyrrhena Terra", -15, 100, 8, 15, 0.30),
    ("Acidalia Planitia", 47, -25, 13, 30, 0.52), ("Utopia Planitia", 46, 110, 12, 30, 0.36),
    ("Vastitas Borealis", 68, -70, 8, 70, 0.30), ("Chryse Planitia", 25, -40, 9, 15, 0.16),
    ("Arabia Terra", 20, 5, 14, 26, -0.20), ("Elysium", 25, 147, 9, 15, -0.18),
    ("Tharsis", 5, -105, 20, 32, -0.16), ("Hellas", -42, 70, 10, 15, -0.62),
    ("Argyre", -50, -43, 7, 12, -0.55), ("Isidis", 13, 88, 6, 12, -0.25),
    ("Olympus Mons", 18.6, -133.8, 5, 6, -0.40), ("Arsia", -9, -120, 3, 3.6, -0.30),
    ("Pavonis", 0.8, -112.8, 3, 3.6, -0.30), ("Ascraeus", 11.8, -104.5, 3, 3.6, -0.30),
]
_VALLES = [(-6.5, -103), (-8, -92), (-11, -80), (-12.5, -68), (-12.5, -58), (-10, -48)]
_MARTE_BRILLANTE = np.array([88, 138, 205], np.float32)        # BGR: ocre anaranjado
_MARTE_OSCURO = np.array([58, 66, 84], np.float32)             # BGR: basalto pardo
_MARTE_ROJIZO = np.array([62, 98, 178], np.float32)            # BGR: polvo rojizo
_HIELO = np.array([245, 245, 238], np.float32)


def textura_marte(ancho=4096, semilla=21, verbose=False):
    """Marte (BGR uint8, ancho x ancho/2) con sus rasgos reales: ver _MARTE_ALBEDO."""
    alto = ancho // 2

    def albedo(n, lat, lon):
        ruido = _fbm(n, 2.2, 5).reshape(lat.shape)
        alabeo = _fbm(n, 4.0, 3, desplaza=40.0).reshape(lat.shape)
        a = 0.58 + 0.10 * ruido + 0.05 * _fbm(n, 14.0, 3, desplaza=7.0).reshape(lat.shape)
        for _, la, lo, rla, rlo, k in _MARTE_ALBEDO:
            a = a - 1.45 * k * _mancha(lat, lon, la, lo, rla, rlo, alabeo)
        # las tierras altas del sur son mas claras y moteadas; el norte (llanuras) mas oscuro y liso
        a += 0.05 * np.clip(-lat / 60.0, -1, 1) * (0.6 + 0.8 * _fbm(n, 6.0, 3, desplaza=3.0).reshape(lat.shape))
        return a

    if verbose:
        print("  albedo...", flush=True)
    alb = _por_filas(ancho, alto, albedo)
    if verbose:
        print("  relieve...", flush=True)
    relieve = _por_filas(ancho, alto, lambda n, lat, lon: _fbm(n, 5.0, 6, 0.55, desplaza=90.0).reshape(lat.shape))

    # Valles Marineris y crateres (dibujados en el mapa, con la distorsion de la proyeccion)
    rng = np.random.default_rng(semilla)
    px = lambda lat, lon: (int(round((lon + 180.0) / 360.0 * ancho)), int(round((90.0 - lat) / 180.0 * alto)))
    capa = np.zeros((alto, ancho), np.float32)
    for (la0, lo0), (la1, lo1) in zip(_VALLES[:-1], _VALLES[1:]):
        cv2.line(capa, px(la0, lo0), px(la1, lo1), 1.0, max(3, ancho // 330), cv2.LINE_AA)
    valles = cv2.GaussianBlur(capa, (0, 0), ancho / 1500.0)
    valles = valles / max(valles.max(), 1e-6)
    alb -= 0.38 * valles
    crat = np.zeros((alto, ancho), np.float32)                       # + borde claro, - piso oscuro
    for _ in range(int(ancho * 0.55)):
        lat = math.degrees(math.asin(rng.uniform(-1, 1)))
        lon = rng.uniform(-180, 180)
        if lat > 25 and rng.random() < 0.8:                           # el norte casi no tiene
            continue
        r = ancho * 0.00055 * (rng.pareto(1.8) + 0.7)
        r = min(r, ancho * 0.006)
        cx, cy = px(lat, lon)
        rx = max(r / max(math.cos(math.radians(lat)), 0.12), 1.0)
        cv2.ellipse(crat, (cx, cy), (int(rx), int(r) + 1), 0, 0, 360, -0.5, -1, cv2.LINE_AA)
        cv2.ellipse(crat, (cx, cy), (int(rx) + 1, int(r) + 2), 0, 0, 360, 0.5, 1, cv2.LINE_AA)
        if cx - rx < 0 or cx + rx >= ancho:                            # que no se corte en el borde de la imagen
            continue
    crat = cv2.GaussianBlur(crat, (0, 0), 0.8)
    alb += 0.045 * crat

    # sombreado de relieve (luz desde el oeste): da volumen a lo que no es plano
    h = relieve + 0.7 * valles * -1.0 + 0.35 * crat
    gx = cv2.Sobel(cv2.GaussianBlur(h, (0, 0), 1.2), cv2.CV_32F, 1, 0, ksize=3)
    lat = np.radians(90.0 - (np.arange(alto) + 0.5) / alto * 180.0)[:, None].astype(np.float32)
    sombra = 1.0 + np.clip(gx * (ancho / 1500.0) * np.cos(lat), -0.28, 0.28)

    # color: de basalto oscuro a ocre claro, con un tinte rojizo de polvo en las zonas claras
    t = np.clip(alb, 0.0, 1.0)[..., None]
    polvo = np.clip(0.5 + 0.9 * _suave(0.5, 0.9, alb)[..., None] * (0.7 + 0.3 * relieve[..., None]), 0, 1)
    color = _MARTE_OSCURO * (1 - t) + (_MARTE_BRILLANTE * (1 - 0.5 * polvo) + _MARTE_ROJIZO * 0.5 * polvo) * t
    color = color * sombra[..., None]

    # casquetes polares (irregulares)
    ruido_polar = cv2.GaussianBlur(rng.random((alto, ancho)).astype(np.float32), (0, 0), ancho / 120.0)
    ruido_polar = (ruido_polar - ruido_polar.mean()) / (ruido_polar.std() + 1e-6)
    latg = np.degrees(lat)
    norte = _suave(0.0, 1.0, (latg - (83.0 - 0.7 * ruido_polar)) * 1.2)
    sur = _suave(0.0, 1.0, (-latg - (86.5 - 0.6 * ruido_polar)) * 1.2)
    hielo = np.clip(norte + sur, 0.0, 1.0)[..., None]
    color = color * (1 - hielo) + _HIELO * (0.9 + 0.1 * sombra[..., None]) * hielo
    return np.clip(color, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Crateres (los comparten Mercurio y Marte-like): borde claro, piso oscuro, en el mapa
# ----------------------------------------------------------------------
def _crateres(ancho, alto, rng, n, r_min, r_max, aspereza=1.8, filtro=None):
    """-> mapa (alto, ancho) con + en el borde y - en el piso de cada crater (para albedo y relieve)."""
    px = lambda lat, lon: (int(round((lon + 180.0) / 360.0 * ancho)), int(round((90.0 - lat) / 180.0 * alto)))
    crat = np.zeros((alto, ancho), np.float32)
    for _ in range(n):
        lat = math.degrees(math.asin(rng.uniform(-1, 1)))
        lon = rng.uniform(-180, 180)
        if filtro is not None and not filtro(lat, lon, rng):
            continue
        r = min(ancho * r_min * (rng.pareto(aspereza) + 0.7), ancho * r_max)
        cx, cy = px(lat, lon)
        rx = max(r / max(math.cos(math.radians(lat)), 0.12), 1.0)
        for dx in (0, -ancho, ancho):                    # que no se corte en el borde de la imagen
            if dx and not (cx + dx - rx < ancho and cx + dx + rx >= 0):
                continue
            c = (cx + dx, cy)
            cv2.ellipse(crat, c, (int(rx), int(r) + 1), 0, 0, 360, -0.5, -1, cv2.LINE_AA)
            cv2.ellipse(crat, c, (int(rx) + 1, int(r) + 2), 0, 0, 360, 0.5, max(1, int(r * 0.18)), cv2.LINE_AA)
    return crat


def _sombra_relieve(h, ancho, alto, fuerza=1.0, tope=0.3):
    """Sombreado de relieve con luz desde el oeste: da volumen a lo que no es plano."""
    gx = cv2.Sobel(cv2.GaussianBlur(h, (0, 0), 1.2), cv2.CV_32F, 1, 0, ksize=3)
    lat = np.radians(90.0 - (np.arange(alto) + 0.5) / alto * 180.0)[:, None].astype(np.float32)
    return 1.0 + np.clip(gx * (ancho / 1500.0) * fuerza * np.cos(lat), -tope, tope)


# ----------------------------------------------------------------------
# Mercurio
# ----------------------------------------------------------------------
# Cuencas de impacto: (nombre, latitud, longitud este, radio en grados, albedo): + aclara (llanuras lisas)
_MERCURIO_CUENCAS = [
    ("Caloris", 30.5, -170.2, 18.0, 0.10), ("Rembrandt", -32.9, 87.9, 8.4, 0.05),
    ("Beethoven", -20.8, -123.6, 7.4, 0.04), ("Tolstoj", -16.3, -164.7, 4.7, -0.10),
    ("Rachmaninoff", 27.6, 57.6, 3.4, 0.06), ("Raditladi", 27.3, 119.1, 3.1, 0.05),
]
# Crateres jovenes con rayos brillantes: (nombre, latitud, longitud este, radio del crater en grados, largo de rayos)
_MERCURIO_RAYOS = [
    ("Hokusai", 57.8, 16.8, 1.6, 34.0), ("Kuiper", -11.35, -31.5, 1.45, 16.0),
    ("Debussy", -33.9, 12.5, 1.9, 26.0), ("Degas", 37.1, -127.0, 1.2, 12.0),
    ("Bartok", -29.6, -134.6, 1.4, 9.0), ("Snorri", -8.7, -83.4, 0.5, 8.0),
]
_MERCURIO_OSCURO = np.array([62, 67, 74], np.float32)          # BGR: material de baja reflectancia (azulado)
_MERCURIO_CLARO = np.array([160, 168, 178], np.float32)        # BGR: gris calido
_MERCURIO_LLANURA = np.array([150, 170, 190], np.float32)      # BGR: llanuras lisas (un poco mas pardas)


def textura_mercurio(ancho=4096, semilla=31, verbose=False):
    """Mercurio (BGR uint8, ancho x ancho/2): crateres por todas partes, cuencas y rayos reales."""
    alto = ancho // 2

    def albedo(n, lat, lon):
        a = 0.52 + 0.09 * _fbm(n, 2.6, 5).reshape(lat.shape) + 0.04 * _fbm(n, 12.0, 3, desplaza=5.0).reshape(lat.shape)
        # material oscuro de baja reflectancia en manchas grandes
        a -= 0.10 * _suave(0.15, 0.6, _fbm(n, 1.8, 3, desplaza=61.0).reshape(lat.shape))
        for _, la, lo, r, k in _MERCURIO_CUENCAS:
            a = a + k * _mancha(lat, lon, la, lo, r, r)
        return a + 0.04 * _suave(52.0, 68.0, lat)                # llanuras lisas del norte (Borealis)

    def lisura(n, lat, lon):
        m = _suave(52.0, 68.0, lat).astype(np.float32)
        for _, la, lo, r, _k in _MERCURIO_CUENCAS:
            m = np.maximum(m, _mancha(lat, lon, la, lo, r * 0.9, r * 0.9))
        return m

    if verbose:
        print("  albedo...", flush=True)
    alb = _por_filas(ancho, alto, albedo)
    liso = _por_filas(ancho, alto, lisura)
    relieve = _por_filas(ancho, alto, lambda n, lat, lon: _fbm(n, 6.0, 6, 0.55, desplaza=120.0).reshape(lat.shape))

    if verbose:
        print("  crateres...", flush=True)
    rng = np.random.default_rng(semilla)
    # menos crateres en las llanuras lisas (son mas jovenes)
    en_liso = lambda lat, lon, g: not (lat > 60 and g.random() < 0.6)
    crat = _crateres(ancho, alto, rng, int(ancho * 1.4), 0.00045, 0.008, 1.6, en_liso)
    crat = cv2.GaussianBlur(crat, (0, 0), 0.8)
    # anillos de las cuencas (borde montanoso)
    px = lambda lat, lon: (int(round((lon + 180.0) / 360.0 * ancho)), int(round((90.0 - lat) / 180.0 * alto)))
    anillos = np.zeros((alto, ancho), np.float32)
    for _, la, lo, r, _k in _MERCURIO_CUENCAS:
        cx, cy = px(la, lo)
        ry = r / 180.0 * alto
        rx = ry / max(math.cos(math.radians(la)), 0.15)
        for dx in (0, -ancho, ancho):                    # Caloris cruza el borde de la imagen (lon -180)
            cv2.ellipse(anillos, (cx + dx, cy), (int(rx), int(ry)), 0, 0, 360, 1.0, max(2, int(ry * 0.12)), cv2.LINE_AA)
    anillos = cv2.GaussianBlur(anillos, (0, 0), ancho / 1200.0)

    # rayos de los crateres jovenes: lineas finas que salen del centro (y un halo claro)
    rayos = np.zeros((alto, ancho), np.float32)
    for _, la, lo, r, largo in _MERCURIO_RAYOS:
        for _k in range(int(18 + largo)):
            ang = rng.uniform(0, 2 * math.pi)
            L = largo * rng.uniform(0.35, 1.0)
            pasos = 12
            pts = []
            for j in range(pasos + 1):                           # recta en la esfera, aprox. en grados locales
                d = L * j / pasos
                lat_j = la + d * math.sin(ang)
                lon_j = lo + d * math.cos(ang) / max(math.cos(math.radians(lat_j)), 0.15)
                pts.append(px(max(min(lat_j, 89.5), -89.5), lon_j))
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                if abs(x1 - x0) < ancho / 2:
                    cv2.line(rayos, (x0, y0), (x1, y1), rng.uniform(0.4, 1.0), max(1, ancho // 2048), cv2.LINE_AA)
        halo = _por_filas(ancho, alto, lambda n, lat, lon, la=la, lo=lo, r=r: _mancha(lat, lon, la, lo, r * 2.5, r * 2.5))
        rayos = np.maximum(rayos, 0.9 * halo)
    rayos = cv2.GaussianBlur(rayos, (0, 0), ancho / 3000.0)

    alb = alb + 0.05 * crat * (1.0 - 0.7 * liso) + 0.28 * rayos
    h = relieve * (1.0 - 0.6 * liso) + 0.5 * crat * (1.0 - 0.6 * liso) + 0.6 * anillos
    sombra = _sombra_relieve(h, ancho, alto, 1.0, 0.32)

    t = np.clip(alb, 0.0, 1.0)[..., None]
    base = _MERCURIO_OSCURO * (1 - t) + _MERCURIO_CLARO * t
    color = base * (1 - 0.35 * liso[..., None]) + (_MERCURIO_LLANURA * (0.6 + 0.6 * t)) * 0.35 * liso[..., None]
    color = color * sombra[..., None]
    return np.clip(color, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Venus (las nubes: lo unico que se ve desde el espacio)
# ----------------------------------------------------------------------
_VENUS_CLARO = np.array([196, 226, 242], np.float32)           # BGR: crema amarillenta
_VENUS_OSCURO = np.array([118, 166, 200], np.float32)          # BGR: ocre palido
_VENUS_Y = [  # la "Y" oscura: tallo en el ecuador y brazos que se abren hacia el oeste
    [(0, 70), (0, 35), (0, 5)],
    [(0, 5), (12, -25), (28, -60), (38, -95)],
    [(0, 5), (-12, -25), (-28, -60), (-38, -95)],
]


def textura_venus(ancho=2048, semilla=41, verbose=False):
    """Nubes de Venus (BGR uint8): bandas zonales, rayas en "V" por el viento, la "Y" y collares polares."""
    alto = ancho // 2

    def nubes(n, lat, lon):
        # el viento (superrotacion, ~100 m/s) estira las nubes en longitud y las tuerce en "V"
        la = np.radians(lat)
        lo = np.radians(lon + 38.0 * np.abs(lat) / 90.0 * (1.0 + 0.3 * np.sin(np.radians(lon * 2))))
        q = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) * 3.2], -1).reshape(-1, 3)
        rayas = _fbm(q.astype(np.float32), 3.0, 5, 0.55).reshape(lat.shape)
        grumos = _fbm(n, 9.0, 3, 0.5, desplaza=12.0).reshape(lat.shape)
        bandas = 0.5 + 0.5 * np.cos(np.radians(lat) * 7.0 + 0.6 * rayas)
        b = 0.60 + 0.19 * rayas + 0.06 * grumos + 0.08 * (bandas - 0.5)
        # collares polares (brillantes, ~60-70 grados) y vortice algo mas oscuro en cada polo
        alat = np.abs(lat)
        b += 0.12 * np.exp(-((alat - 64.0) / 7.0) ** 2)
        b -= 0.10 * _suave(76.0, 90.0, alat)
        return b

    if verbose:
        print("  nubes...", flush=True)
    b = _por_filas(ancho, alto, nubes)
    px = lambda lat, lon: (int(round((lon + 180.0) / 360.0 * ancho)), int(round((90.0 - lat) / 180.0 * alto)))
    capa = np.zeros((alto, ancho), np.float32)
    for tramo in _VENUS_Y:
        for (la0, lo0), (la1, lo1) in zip(tramo[:-1], tramo[1:]):
            cv2.line(capa, px(la0, lo0), px(la1, lo1), 1.0, max(3, ancho // 60), cv2.LINE_AA)
    capa = cv2.GaussianBlur(capa, (0, 0), ancho / 120.0)
    b -= 0.13 * capa / max(capa.max(), 1e-6)
    b = cv2.GaussianBlur(b, (0, 0), ancho / 2048.0)              # las nubes no tienen bordes duros
    t = np.clip(b, 0.0, 1.0)[..., None]
    return np.clip(_VENUS_OSCURO * (1 - t) + _VENUS_CLARO * t, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Sol
# ----------------------------------------------------------------------
_MANCHAS_SOL = [(14, -40, 3.2), (-11, 15, 2.4), (18, 80, 2.0), (-16, 130, 2.8), (9, -125, 1.6),
                (-8, -75, 1.3), (22, 25, 1.2)]
_SOL_COLOR = np.array([0.30, 0.70, 1.0], np.float32)             # BGR de la fotosfera (blanco-amarillo)


def textura_sol(ancho=2048, semilla=5, verbose=False):
    """Fotosfera del Sol (BGR uint8): granulacion, manchas con penumbra y fáculas."""
    alto = ancho // 2

    def brillo(n, lat, lon):
        gran = _fbm(n, 34.0, 3, 0.55).reshape(lat.shape)            # celulas de ~1 000 km
        super_gran = _fbm(n, 16.0, 3, 0.6, desplaza=33.0).reshape(lat.shape)
        celdas = 1.0 - np.abs(gran) * 1.6                            # carriles oscuros entre granulos
        b = 0.72 + 0.28 * celdas + 0.10 * super_gran
        for la, lo, r in _MANCHAS_SOL:
            for lo2, rr in ((lo, r), ):                              # mancha + compañera
                d = np.degrees(np.arccos(np.clip(
                    np.sin(np.radians(lat)) * math.sin(math.radians(la))
                    + np.cos(np.radians(lat)) * math.cos(math.radians(la)) * np.cos(np.radians(lon - lo2)), -1, 1)))
                penumbra = np.exp(-(d / (rr * 2.2)) ** 3)
                umbra = np.exp(-(d / (rr * 0.8)) ** 3)
                faculas = np.exp(-(d / (rr * 4.0)) ** 2) * (1.0 - penumbra) * 0.12 * (0.6 + 0.4 * gran)
                b = b - 0.30 * penumbra - 0.38 * umbra + faculas
        return b

    return np.clip(_por_filas(ancho, alto, brillo)[..., None] * _SOL_COLOR * 255.0, 0, 255).astype(np.uint8)


def generar(carpeta, verbose=True, solo=None):
    """solo: nombres de astros a generar (p. ej. {"venus"}); None = todos."""
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)
    for nombre, f, kw in (("mercurio_procedural.jpg", textura_mercurio, dict(ancho=4096)),
                          ("venus_procedural.jpg", textura_venus, dict(ancho=2048)),
                          ("marte_procedural.jpg", textura_marte, dict(ancho=4096)),
                          ("sol_procedural.jpg", textura_sol, dict(ancho=2048))):
        if solo and nombre.split("_")[0] not in solo:
            continue
        print("Generando", nombre, "...", flush=True)
        img = f(verbose=verbose, **kw)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        (carpeta / nombre).write_bytes(buf.tobytes())
        print("  listo:", nombre, img.shape, len(buf) // 1024, "KB", flush=True)


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    generar(raiz / "assets", solo=set(sys.argv[1:]) or None)       # p. ej.: python -m universo.planetas venus
