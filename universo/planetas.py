"""Texturas de respaldo de Marte y del Sol, generadas con numpy/OpenCV.

Sirven mientras no pongas las imagenes reales de la NASA en assets/ (mira
herramientas/descargar_texturas.py). Marte lleva sus rasgos reales en su sitio (Syrtis Major,
Valles Marineris, Olympus Mons, Hellas, Solis Lacus, los casquetes polares...); el Sol, granulacion
y manchas. Son equirectangulares (longitud 0 al centro, este a la derecha), igual que las de la NASA.

Generar los archivos (una sola vez):   python -m universo.planetas
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


def generar(carpeta, verbose=True):
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)
    for nombre, f, kw in (("marte_procedural.jpg", textura_marte, dict(ancho=4096)),
                          ("sol_procedural.jpg", textura_sol, dict(ancho=2048))):
        print("Generando", nombre, "...", flush=True)
        img = f(verbose=verbose, **kw)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        (carpeta / nombre).write_bytes(buf.tobytes())
        print("  listo:", nombre, img.shape, len(buf) // 1024, "KB", flush=True)


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    generar(raiz / "assets")
