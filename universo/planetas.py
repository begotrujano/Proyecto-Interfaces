"""Texturas de respaldo de Mercurio, Venus, Marte, Jupiter, Saturno y el Sol, generadas con numpy/OpenCV.

Sirven mientras no pongas las imagenes reales de la NASA en assets/ (mira
herramientas/descargar_texturas.py). Marte lleva sus rasgos reales en su sitio (Syrtis Major,
Valles Marineris, Olympus Mons, Hellas, Solis Lacus, los casquetes polares...); Mercurio, sus
cuencas (Caloris, Rembrandt, Beethoven...) y sus crateres con rayos (Hokusai, Kuiper...); Venus,
sus nubes en bandas con la "Y" oscura y los collares polares; Jupiter, sus cinturones y zonas con la Gran Mancha Roja; Saturno, sus bandas doradas suaves y el hexagono del polo norte; el Sol, granulacion y manchas. Son equirectangulares (longitud 0 al centro, este a la derecha), igual que las de la NASA.

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
# Jupiter (las nubes: cinturones oscuros y zonas claras, la Gran Mancha Roja y sus ovalos)
# ----------------------------------------------------------------------
_JUP_ZONA = np.array([204, 228, 238], np.float32)             # BGR: crema
_JUP_CINTURON = np.array([72, 112, 172], np.float32)          # BGR: marron anaranjado
_JUP_POLO = np.array([150, 158, 168], np.float32)             # BGR: gris pardo de los polos
_JUP_ROJO = np.array([55, 85, 190], np.float32)               # BGR: rojo ladrillo (la Gran Mancha)
# (latitud, oscuridad 0..1): 0 = zona clara, 1 = cinturon oscuro (NTB, NEB, SEB, STB... con su zona entre ellos)
_JUP_PERFIL = [(-90, 0.50), (-70, 0.60), (-60, 0.42), (-50, 0.25), (-44, 0.55), (-39, 0.20), (-34, 0.60),
               (-29, 0.18), (-24, 0.35), (-19, 0.85), (-13, 0.95), (-8, 0.62), (-4, 0.12), (0, 0.04),
               (4, 0.10), (8, 0.55), (13, 0.95), (17, 0.88), (21, 0.30), (25, 0.65), (29, 0.20),
               (36, 0.50), (42, 0.18), (50, 0.40), (60, 0.50), (72, 0.62), (90, 0.52)]
_JUP_PERFIL = np.array(_JUP_PERFIL, np.float32)
_JUP_OVALOS = [(-33.0, -58.0, 1.0), (-33.0, 95.0, 0.9), (-33.0, 150.0, 0.8), (-41.0, 40.0, 0.7), (-41.0, -130.0, 0.7)]


def textura_jupiter(ancho=2048, semilla=23, verbose=False):
    """Nubes de Jupiter (BGR uint8): cinturones y zonas con bordes ondulados y vientos que las estiran,
    festones en el ecuador, la Gran Mancha Roja (22 grados sur) y ovalos blancos."""
    alto = ancho // 2

    def nubes(n, lat, lon):
        la = np.radians(lat)
        ondas = _fbm(n, 2.4, 4, 0.55, desplaza=31.0).reshape(lat.shape)
        # remolinos: la longitud se tuerce mas donde hay cortante entre cinturon y zona (cerca de los bordes)
        cortante = np.abs(np.gradient(np.interp(lat, _JUP_PERFIL[:, 0], _JUP_PERFIL[:, 1]), axis=0)) * 60.0
        lo = np.radians(lon) + 0.9 * ondas * np.clip(0.25 + cortante, 0.0, 1.0)
        # corrientes en chorro: el ruido se estira mucho en longitud (los vientos cambian con la latitud)
        q = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) * 6.0], -1).reshape(-1, 3)
        rayas = _fbm(q.astype(np.float32), 3.2, 6, 0.62).reshape(lat.shape)
        q2 = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) * 2.5], -1).reshape(-1, 3)
        grumos = _fbm(q2.astype(np.float32) * 5.0, 3.0, 4, 0.6, desplaza=9.0).reshape(lat.shape)
        # el borde de cada cinturon serpentea: se busca la oscuridad en una latitud desplazada
        borde = 1.0 + 2.2 * np.exp(-((np.abs(lat) - 9.0) / 5.0) ** 2)      # mas fuerte en los bordes del ecuador
        lat_w = lat + (2.2 * ondas + 1.2 * rayas) * borde
        osc = np.interp(lat_w, _JUP_PERFIL[:, 0], _JUP_PERFIL[:, 1])
        osc = osc + 0.20 * rayas + 0.22 * grumos * (0.4 + osc)
        # festones: lengüetas oscuras que salen del cinturon ecuatorial norte hacia la zona clara
        feston = np.exp(-((lat - 6.5) / 2.2) ** 2) * np.clip(np.sin(np.radians(lon) * 11.0 + 3.0 * ondas), 0, 1)
        osc = osc + 0.35 * feston
        return osc

    if verbose:
        print("  nubes...", flush=True)
    osc = np.clip(_por_filas(ancho, alto, nubes), 0.0, 1.0) ** 1.15

    lat = (90.0 - (np.arange(alto) + 0.5) / alto * 180.0).astype(np.float32)[:, None] * np.ones((1, ancho), np.float32)
    lon = ((np.arange(ancho) + 0.5) / ancho * 360.0 - 180.0).astype(np.float32)[None, :] * np.ones((alto, 1), np.float32)
    ala = np.zeros((alto, ancho), np.float32)
    # Gran Mancha Roja: nucleo rojo, halo claro alrededor y estela turbulenta hacia el oeste
    nucleo = _mancha(lat, lon, -22.0, 20.0, 5.2, 11.0)
    halo = np.clip(_mancha(lat, lon, -22.0, 20.0, 8.5, 17.5) - nucleo, 0.0, 1.0)
    estela = _mancha(lat, lon, -22.0, 6.0, 4.5, 22.0)
    osc = osc * (1.0 - 0.75 * halo) + 0.25 * estela * (0.5 + 0.5 * np.sin(np.radians(lon) * 17.0))
    osc = np.clip(osc, 0.0, 1.0)
    rojo = nucleo ** 0.7
    ovalos = np.zeros((alto, ancho), np.float32)
    for la0, lo0, k in _JUP_OVALOS:
        ovalos = np.maximum(ovalos, k * _mancha(lat, lon, la0, lo0, 2.2 * k, 3.8 * k))

    polo = _suave(55.0, 80.0, np.abs(lat))
    t = osc[..., None]
    color = _JUP_ZONA * (1 - t) + _JUP_CINTURON * t
    color = color * (1 - 0.55 * polo[..., None]) + _JUP_POLO * 0.55 * polo[..., None] * (0.75 + 0.25 * (1 - t))
    color = color * (1 - rojo[..., None]) + _JUP_ROJO * rojo[..., None]
    color = color * (1 - 0.8 * ovalos[..., None]) + _JUP_ZONA * 1.04 * 0.8 * ovalos[..., None]
    color = cv2.GaussianBlur(color, (0, 0), ancho / 4000.0)     # las nubes no tienen bordes duros
    return np.clip(color, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Saturno (bandas doradas muy suaves, tormentas pequenas y el hexagono del polo norte; los anillos: anillos.py)
# ----------------------------------------------------------------------
_SAT_ZONA = np.array([150, 205, 232], np.float32)             # BGR: crema dorado
_SAT_BANDA = np.array([108, 158, 198], np.float32)            # BGR: ocre
_SAT_POLO = np.array([128, 150, 150], np.float32)             # BGR: gris verdoso (polos)
_SAT_PERFIL = np.array([(-90, 0.60), (-75, 0.55), (-62, 0.45), (-52, 0.22), (-42, 0.40), (-34, 0.12), (-24, 0.42),
                        (-16, 0.18), (-7, 0.38), (0, 0.08), (7, 0.30), (15, 0.16), (24, 0.45), (34, 0.14),
                        (44, 0.40), (56, 0.26), (66, 0.46), (78, 0.55), (90, 0.60)], np.float32)


def textura_saturno(ancho=2048, semilla=29, verbose=False):
    """Nubes de Saturno (BGR uint8): bandas doradas de poco contraste, estrias, tormentas y el hexagono polar."""
    alto = ancho // 2

    def nubes(n, lat, lon):
        la = np.radians(lat)
        ondas = _fbm(n, 2.2, 4, 0.5, desplaza=13.0).reshape(lat.shape)
        lo = np.radians(lon) + 0.5 * ondas
        q = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) * 8.0], -1).reshape(-1, 3)
        rayas = _fbm(q.astype(np.float32), 3.0, 5, 0.6).reshape(lat.shape)
        lat_w = lat + 1.2 * ondas + 0.6 * rayas
        osc = np.interp(lat_w, _SAT_PERFIL[:, 0], _SAT_PERFIL[:, 1]) + 0.10 * rayas
        # tormentas: manchas claras alargadas en el hemisferio norte (como la "Gran Mancha Blanca")
        for la0, lo0, r in ((32.0, 40.0, 1.0), (40.0, -120.0, 0.6), (-38.0, 150.0, 0.5)):
            osc = osc - 0.45 * r * _mancha(lat, lon, la0, lo0, 2.4 * r, 7.0 * r, alabeo=ondas)
        # el hexagono del polo norte: la corriente en chorro (lat ~78) forma 6 lados
        hexa = 78.5 + 1.6 * np.cos(np.radians(lon) * 6.0)
        osc = osc + 0.28 * np.exp(-((lat - hexa) / 1.4) ** 2) - 0.10 * _suave(hexa, 90.0, lat)
        return osc

    if verbose:
        print("  nubes...", flush=True)
    osc = np.clip(_por_filas(ancho, alto, nubes), 0.0, 1.0)
    lat = (90.0 - (np.arange(alto) + 0.5) / alto * 180.0).astype(np.float32)[:, None]
    polo = _suave(62.0, 88.0, np.abs(lat)) * np.ones((1, ancho), np.float32)
    t = osc[..., None]
    color = _SAT_ZONA * (1 - t) + _SAT_BANDA * t
    color = color * (1 - 0.50 * polo[..., None]) + _SAT_POLO * 0.50 * polo[..., None]
    color = cv2.GaussianBlur(color, (0, 0), ancho / 4000.0)
    return np.clip(color, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Urano (casi liso: cian palido con bandas muy tenues y casquete polar mas claro)
# ----------------------------------------------------------------------
_URA_CLARO = np.array([196, 192, 138], np.float32)            # BGR: cian palido
_URA_OSCURO = np.array([172, 170, 118], np.float32)           # BGR: cian algo mas oscuro


def textura_urano(ancho=2048, semilla=37, verbose=False):
    """Atmosfera de Urano (BGR uint8): bandas casi invisibles y el casquete polar (sur) mas brillante."""
    alto = ancho // 2

    def nubes(n, lat, lon):
        la = np.radians(lat)
        ondas = _fbm(n, 2.0, 3, 0.5, desplaza=7.0).reshape(lat.shape)
        lat_w = lat + 1.5 * ondas
        bandas = 0.5 + 0.5 * np.sin(np.radians(lat_w) * 7.0 + 0.8)
        q = np.stack([np.cos(la) * np.cos(np.radians(lon)), np.cos(la) * np.sin(np.radians(lon)), np.sin(la) * 6.0], -1).reshape(-1, 3)
        rayas = _fbm(q.astype(np.float32), 4.0, 4, 0.55).reshape(lat.shape)
        return 0.55 * bandas + 0.35 * rayas + 0.10 * np.clip(lat / 90.0, -1, 1)

    if verbose:
        print("  nubes...", flush=True)
    t = np.clip(_por_filas(ancho, alto, nubes), 0.0, 1.0)[..., None]
    lat = (90.0 - (np.arange(alto) + 0.5) / alto * 180.0).astype(np.float32)[:, None]
    capa = _suave(-45.0, -75.0, lat) * np.ones((1, ancho), np.float32)       # casquete polar sur, mas claro
    color = _URA_CLARO * (1 - 0.55 * t) + _URA_OSCURO * 0.55 * t
    color = color * (1 - 0.35 * capa[..., None]) + np.array([208, 204, 160], np.float32) * 0.35 * capa[..., None]
    color = cv2.GaussianBlur(color, (0, 0), ancho / 3000.0)
    return np.clip(color, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Neptuno (azul profundo, bandas suaves, Gran Mancha Oscura y nubes de cirro blancas)
# ----------------------------------------------------------------------
_NEP_CLARO = np.array([215, 125, 80], np.float32)             # BGR: azul medio
_NEP_OSCURO = np.array([160, 70, 40], np.float32)             # BGR: azul profundo


def textura_neptuno(ancho=2048, semilla=41, verbose=False):
    """Atmosfera de Neptuno (BGR uint8): bandas azules, Gran Mancha Oscura, cirros blancos alargados."""
    alto = ancho // 2

    def nubes(n, lat, lon):
        la = np.radians(lat)
        ondas = _fbm(n, 2.4, 4, 0.5, desplaza=9.0).reshape(lat.shape)
        lat_w = lat + 2.0 * ondas
        bandas = 0.5 + 0.5 * np.sin(np.radians(lat_w) * 5.0 + 0.3)
        q = np.stack([np.cos(la) * np.cos(np.radians(lon)), np.cos(la) * np.sin(np.radians(lon)), np.sin(la) * 7.0], -1).reshape(-1, 3)
        rayas = _fbm(q.astype(np.float32), 3.5, 5, 0.55).reshape(lat.shape)
        osc = 0.45 * bandas + 0.35 * rayas
        osc = osc + 0.55 * _mancha(lat, lon, -22.0, 20.0, 6.0, 12.0, alabeo=ondas)             # Gran Mancha Oscura
        return osc

    def cirros(n, lat, lon):
        la = np.radians(lat)
        ondas = _fbm(n, 2.0, 3, 0.5, desplaza=5.0).reshape(lat.shape)
        q = np.stack([np.cos(la) * np.cos(np.radians(lon)), np.cos(la) * np.sin(np.radians(lon)), np.sin(la) * 16.0], -1).reshape(-1, 3)
        f = _fbm(q.astype(np.float32), 4.0, 4, 0.6).reshape(lat.shape)
        franja = np.exp(-((np.abs(lat + 30.0) - 6.0) / 5.0) ** 2) + 0.7 * np.exp(-((lat - 40.0) / 6.0) ** 2)
        return np.clip((f - 0.55) * 4.0, 0.0, 1.0) * franja + 0.9 * _mancha(lat, lon, -42.0, 30.0, 1.5, 6.0, alabeo=ondas)

    if verbose:
        print("  nubes...", flush=True)
    t = np.clip(_por_filas(ancho, alto, nubes), 0.0, 1.0)[..., None]
    c = np.clip(_por_filas(ancho, alto, cirros), 0.0, 1.0)[..., None]
    color = _NEP_CLARO * (1 - t) + _NEP_OSCURO * t
    color = color * (1 - 0.7 * c) + np.array([245, 235, 225], np.float32) * 0.7 * c
    color = cv2.GaussianBlur(color, (0, 0), ancho / 3500.0)
    return np.clip(color, 0, 255).astype(np.uint8)


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
                          ("jupiter_procedural.jpg", textura_jupiter, dict(ancho=2048)),
                          ("saturno_procedural.jpg", textura_saturno, dict(ancho=2048)),
                          ("urano_procedural.jpg", textura_urano, dict(ancho=2048)),
                          ("neptuno_procedural.jpg", textura_neptuno, dict(ancho=2048)),
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
