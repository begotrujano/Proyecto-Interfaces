"""Astronomia: donde estan de verdad el Sol, la Luna y las estrellas, y la camara.

Solo depende de numpy (y math/datetime). Todo se expresa en el marco de la Tierra
("ECEF"): x hacia lon 0 / lat 0 (Greenwich, ecuador), y hacia lon 90 E, z hacia el
polo norte. Las distancias van en radios terrestres salvo que se diga otra cosa.

Precision (formulas de baja precision del Astronomical Almanac):
  - Sol: ~0.01 grados en direccion, ~0.0001 UA en distancia.
  - Luna: ~0.3 grados en direccion y ~0.2 % en distancia (unos 800 km de 384 400).
Suficiente para que el dia/noche, las fases y los eclipses salgan donde deben.
"""

import math
from datetime import datetime, timezone

import numpy as np

RADIO_TIERRA_KM = 6378.137
RADIO_LUNA_KM = 1737.4
RADIO_SOL_KM = 695_700.0
UA_KM = 149_597_870.7

_RAD = math.pi / 180.0


def dia_juliano(fecha):
    """datetime (con zona; si no tiene, se asume UTC) -> dia juliano."""
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    fecha = fecha.astimezone(timezone.utc)
    j2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    return 2451545.0 + (fecha - j2000).total_seconds() / 86400.0


def tiempo_sidereo(jd):
    """Angulo de Greenwich (GMST) en radianes: cuanto ha girado la Tierra respecto a las estrellas."""
    d = jd - 2451545.0
    t = d / 36525.0
    g = 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t
    return (g % 360.0) * _RAD


def _oblicuidad(jd):
    return (23.439291 - 0.0130042 * (jd - 2451545.0) / 36525.0) * _RAD


def _ecliptica_a_ecuatorial(lon, lat, eps):
    cl, sl, cb, sb = math.cos(lon), math.sin(lon), math.cos(lat), math.sin(lat)
    ce, se = math.cos(eps), math.sin(eps)
    return np.array([cb * cl, ce * cb * sl - se * sb, se * cb * sl + ce * sb])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ecuatorial_a_tierra(jd):
    """Matriz que pasa vectores del marco ecuatorial (estrellas) al marco de la Tierra."""
    return rot_z(-tiempo_sidereo(jd))


def sol(jd):
    """-> (direccion unitaria en el marco de la Tierra, distancia en km)."""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) * _RAD
    g = (357.528 + 0.9856003 * n) * _RAD
    lam = L + (1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) * _RAD
    r_ua = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
    v = ecuatorial_a_tierra(jd) @ _ecliptica_a_ecuatorial(lam, 0.0, _oblicuidad(jd))
    return v / np.linalg.norm(v), r_ua * UA_KM


def sol_posicion(jd):
    """Centro del Sol en el marco de la Tierra, en radios terrestres."""
    d, km = sol(jd)
    return d * (km / RADIO_TIERRA_KM)


def luna(jd):
    """-> vector Tierra->Luna en el marco de la Tierra, en radios terrestres."""
    t = (jd - 2451545.0) / 36525.0
    s = lambda a, b: math.sin((a + b * t) * _RAD)
    c = lambda a, b: math.cos((a + b * t) * _RAD)
    lam = (218.32 + 481267.881 * t
           + 6.29 * s(135.0, 477198.87) - 1.27 * s(259.3, -413335.36)
           + 0.66 * s(235.7, 890534.22) + 0.21 * s(269.9, 954397.74)
           - 0.19 * s(357.5, 35999.05) - 0.11 * s(186.5, 966404.03))
    bet = (5.13 * s(93.3, 483202.02) + 0.28 * s(228.2, 960400.89)
           - 0.28 * s(318.3, 6003.15) - 0.17 * s(217.6, -407332.21))
    par = (0.9508 + 0.0518 * c(135.0, 477198.87) + 0.0095 * c(259.3, -413335.36)
           + 0.0078 * c(235.7, 890534.22) + 0.0028 * c(269.9, 954397.74))
    dist = 1.0 / math.sin(par * _RAD)                       # radios terrestres
    v = _ecliptica_a_ecuatorial(lam * _RAD, bet * _RAD, _oblicuidad(jd))
    return ecuatorial_a_tierra(jd) @ v * dist


def punto_superficie(lat_g, lon_g):
    """Latitud/longitud (grados) -> vector unitario en el marco de la Tierra."""
    la, lo = lat_g * _RAD, lon_g * _RAD
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


# ----------------------------------------------------------------------
# Camara que orbita la Tierra
# ----------------------------------------------------------------------
class Camara:
    """Camara en orbita mirando al centro de la Tierra.

    giro/inclinacion son los mismos numeros que maneja control.py (no cambia como se
    sienten los gestos): la camara esta sobre lon = -giro, lat = -inclinacion.
    ejes: filas de M = derecha, abajo, atras (hacia la camara). Un vector v del marco
    de la Tierra se ve en pantalla como M @ v (x derecha, y abajo, z hacia ti)."""

    def __init__(self, giro, inclinacion, distancia, focal_px, cx, cy, centro=None):
        lon, lat = -giro, -inclinacion
        cl, sl, cb, sb = math.cos(lon), math.sin(lon), math.cos(lat), math.sin(lat)
        atras = np.array([cb * cl, cb * sl, sb])
        derecha = np.array([-sl, cl, 0.0])
        arriba = np.array([-sb * cl, -sb * sl, cb])
        self.M = np.stack([derecha, -arriba, atras])
        centro = np.zeros(3) if centro is None else np.asarray(centro, float)
        self.pos = centro + atras * distancia
        self.distancia = distancia
        self.f, self.cx, self.cy = focal_px, cx, cy

    @classmethod
    def en_astro(cls, giro, inclinacion, distancia, focal_px, cx, cy, centro, rot):
        """Camara en orbita de un astro cualquiera: giro/inclinacion se miden en el marco del
        astro (rot: marco de la Tierra -> marco del astro) y su polo norte queda ARRIBA en la
        pantalla. Para la Tierra (rot = identidad) es igual que Camara(...)."""
        cam = cls(giro, inclinacion, distancia, focal_px, cx, cy)
        atras_astro = cam.pos / distancia                       # (el centro de cam era el origen)
        cam.M = cam.M @ rot                                     # ejes de la camara, ahora en marco de la Tierra
        cam.pos = np.asarray(centro, float) + (rot.T @ atras_astro) * distancia
        return cam

    @classmethod
    def mirando(cls, pos, objetivo, focal_px, cx, cy, arriba=(0.0, 0.0, 1.0)):
        """Camara en pos que mira a objetivo con `arriba` hacia arriba (por omision, el norte
        de la Tierra). Se usa en el viaje."""
        cam = cls(0.0, 0.0, 1.0, focal_px, cx, cy)
        adelante = np.asarray(objetivo, float) - np.asarray(pos, float)
        adelante /= np.linalg.norm(adelante)
        derecha = np.cross(adelante, arriba)
        if np.linalg.norm(derecha) < 1e-6:
            derecha = np.array([0.0, 1.0, 0.0])
        derecha /= np.linalg.norm(derecha)
        abajo = np.cross(adelante, derecha)
        cam.M = np.stack([derecha, abajo, -adelante])
        cam.pos = np.asarray(pos, float)
        cam.distancia = float(np.linalg.norm(np.asarray(objetivo, float) - cam.pos))
        return cam

    def ejes_hacia(self, centro):
        """Ejes de pantalla (derecha, abajo, atras) de la camara apuntada a un astro concreto: la
        esfera se ve desde la direccion astro -> camara, no desde el eje optico (importa en las orillas)."""
        atras = self.pos - np.asarray(centro, float)
        atras = atras / np.linalg.norm(atras)
        derecha = self.M[0] - (self.M[0] @ atras) * atras
        n = np.linalg.norm(derecha)
        derecha = derecha / n if n > 1e-9 else np.cross(self.M[1], atras)
        return np.stack([derecha, np.cross(derecha, atras), atras])

    def a_camara(self, v):
        """Vector(es) del marco de la Tierra -> marco de la camara."""
        return np.asarray(v) @ self.M.T

    def proyectar_direccion(self, d):
        """Direccion(es) lejanas (estrellas, Sol) -> (x, y, delante)."""
        c = self.a_camara(d)
        z = -c[..., 2]
        zz = np.where(z > 1e-9, z, 1e-9)
        return self.cx + self.f * c[..., 0] / zz, self.cy + self.f * c[..., 1] / zz, z > 1e-6

    def proyectar_punto(self, p):
        """Punto (radios terrestres) -> (x, y, profundidad). profundidad <= 0: detras de la camara."""
        c = self.a_camara(np.asarray(p) - self.pos)
        z = -c[2]
        if z <= 1e-9:
            return 0.0, 0.0, z
        return self.cx + self.f * c[0] / z, self.cy + self.f * c[1] / z, z


def giro_inclinacion(direccion):
    """Direccion desde el centro de un astro hacia la camara -> (giro, inclinacion) de control.py."""
    d = np.asarray(direccion, float) / np.linalg.norm(direccion)
    return -math.atan2(d[1], d[0]), -math.asin(float(np.clip(d[2], -1, 1)))


def orientacion_vista(M, rot):
    """Como se ve un astro desde una camara cualquiera, en los 3 numeros que usa su dibujante:
    (giro, inclinacion, rodar). M: ejes de la camara en el marco de la Tierra (filas = derecha,
    abajo, atras); rot: marco de la Tierra -> marco del astro. `rodar` es cuanto esta girada la
    camara alrededor de su eje de vista respecto a la que tiene el norte del astro arriba."""
    Mb = M @ rot.T                                   # ejes de la camara en el marco del astro
    atras = Mb[2] / np.linalg.norm(Mb[2])
    lon, lat = math.atan2(atras[1], atras[0]), math.asin(float(np.clip(atras[2], -1, 1)))
    cl, sl, cb, sb = math.cos(lon), math.sin(lon), math.cos(lat), math.sin(lat)
    derecha_c = np.array([-sl, cl, 0.0])             # la "derecha" de la camara sin rodar...
    arriba_c = np.array([-sb * cl, -sb * sl, cb])    # ...y su "arriba"
    rodar = math.atan2(float(Mb[0] @ arriba_c), float(Mb[0] @ derecha_c))
    return -lon, -lat, rodar


def distancia_para_radio(radio_px, focal_px):
    """Distancia al centro (radios terrestres) a la que la Tierra se ve con ese radio en pantalla."""
    return 1.0 / math.sin(math.atan(max(radio_px, 1e-6) / focal_px))


def radio_aparente_px(radio, distancia, focal_px):
    return focal_px * math.tan(math.asin(min(radio / max(distancia, 1e-9), 1.0)))


# ----------------------------------------------------------------------
# Estrellas: las mas brillantes son reales (para reconocer Orion, la Osa Mayor,
# Casiopea, la Cruz del Sur...) y el resto se rellena con estadistica real
# (cuantas hay por magnitud y que se juntan en la Via Lactea).
# (ascension recta en horas, declinacion en grados, magnitud, color: a azul, b blanca, n naranja, r roja)
# ----------------------------------------------------------------------
BRILLANTES = [
    (6.752, -16.72, -1.46, "a"), (6.399, -52.70, -0.74, "b"), (14.660, -60.84, -0.27, "n"),
    (14.261, 19.18, -0.05, "n"), (18.616, 38.78, 0.03, "a"), (5.278, 46.00, 0.08, "n"),
    (5.242, -8.20, 0.13, "a"), (7.655, 5.22, 0.34, "b"), (5.919, 7.41, 0.50, "r"),
    (1.629, -57.24, 0.46, "a"), (14.064, -60.37, 0.61, "a"), (19.846, 8.87, 0.77, "b"),
    (12.443, -63.10, 0.77, "a"), (4.599, 16.51, 0.86, "r"), (16.490, -26.43, 0.96, "r"),
    (13.420, -11.16, 0.97, "a"), (7.755, 28.03, 1.14, "n"), (22.961, -29.62, 1.16, "b"),
    (20.690, 45.28, 1.25, "b"), (12.795, -59.69, 1.25, "a"), (10.140, 11.97, 1.35, "a"),
    (6.977, -28.97, 1.50, "a"), (7.577, 31.89, 1.58, "b"), (17.560, -37.10, 1.62, "a"),
    (12.519, -57.11, 1.63, "r"), (5.419, 6.35, 1.64, "a"), (5.438, 28.61, 1.65, "a"),
    (9.220, -69.72, 1.67, "b"), (5.604, -1.20, 1.69, "a"), (5.679, -1.94, 1.77, "a"),
    (5.533, -0.30, 2.23, "a"), (12.900, 55.96, 1.77, "b"), (11.062, 61.75, 1.79, "n"),
    (3.405, 49.86, 1.79, "b"), (7.140, -26.39, 1.83, "b"), (13.792, 49.31, 1.86, "a"),
    (17.622, -43.00, 1.86, "b"), (8.375, -59.51, 1.86, "n"), (5.992, 44.95, 1.90, "b"),
    (16.811, -69.03, 1.91, "n"), (6.629, 16.40, 1.92, "b"), (20.427, -56.74, 1.94, "a"),
    (2.530, 89.26, 1.98, "b"), (6.378, -17.96, 1.98, "a"), (9.460, -8.66, 1.98, "n"),
    (2.120, 23.46, 2.00, "n"), (0.727, -17.99, 2.04, "n"), (18.921, -26.30, 2.05, "a"),
    (14.111, -36.37, 2.06, "n"), (0.140, 29.09, 2.06, "a"), (1.162, 35.62, 2.07, "r"),
    (5.796, -9.67, 2.09, "a"), (14.845, 74.16, 2.08, "n"), (17.582, 12.56, 2.08, "b"),
    (3.136, 40.96, 2.10, "a"), (2.065, 42.33, 2.10, "n"), (11.818, 14.57, 2.13, "b"),
    (11.031, 56.38, 2.37, "b"), (11.897, 53.69, 2.44, "b"), (12.257, 57.03, 3.31, "b"),
    (13.399, 54.93, 2.23, "b"), (0.675, 56.54, 2.24, "n"), (0.153, 59.15, 2.28, "b"),
    (0.945, 60.72, 2.15, "a"), (1.430, 60.24, 2.66, "b"), (1.907, 63.67, 3.35, "a"),
    (12.252, -58.75, 2.79, "a"), (12.356, -60.40, 3.59, "n"), (5.585, -5.39, 2.77, "a"),
    (16.006, -22.62, 2.29, "a"), (16.091, -19.81, 2.62, "a"), (16.836, -34.29, 2.29, "n"),
    (17.708, -39.03, 2.39, "a"), (17.793, -37.04, 2.99, "a"), (19.162, -29.88, 2.60, "b"),
    (18.403, -34.38, 1.85, "b"), (18.350, -29.83, 2.70, "n"), (19.771, 10.61, 2.72, "n"),
    (20.370, 40.26, 2.23, "b"), (19.512, 27.96, 3.05, "n"), (20.770, 33.97, 2.48, "n"),
    (3.792, 24.11, 2.87, "a"), (21.736, 9.88, 2.39, "n"), (22.137, -46.96, 1.74, "a"),
    (8.159, -47.34, 1.78, "a"), (9.133, -43.43, 2.21, "n"), (8.745, -54.71, 1.96, "a"),
    (0.438, -42.31, 2.40, "n"), (23.063, 28.08, 2.42, "r"), (23.079, 15.21, 2.49, "b"),
    (0.221, 15.18, 2.83, "b"), (3.038, 4.09, 2.54, "r"), (15.578, 26.71, 2.22, "b"),
    (16.504, 21.49, 2.78, "n"), (17.244, 14.39, 2.78, "r"), (10.333, 19.84, 2.08, "n"),
    (11.235, 20.52, 2.56, "b"), (4.330, 15.63, 3.40, "n"), (3.791, 24.37, 3.70, "a"),
]

# Ecuatorial (J2000) -> galactico: para poner la Via Lactea donde esta de verdad
ECU_A_GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                      [0.4941094279, -0.4448296300, 0.7469822445],
                      [-0.8676661490, -0.1980763734, 0.4559837762]])

COLOR_ESTRELLA = {"a": (1.0, 0.86, 0.72), "b": (1.0, 0.97, 0.94),     # BGR
                  "n": (0.78, 0.90, 1.0), "r": (0.62, 0.78, 1.0)}


def _vector_ra_dec(ra_h, dec_g):
    ra, dec = np.asarray(ra_h) * 15.0 * _RAD, np.asarray(dec_g) * _RAD
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def catalogo_estrellas(n_debiles=9000, mag_lim=6.8, semilla=5):
    """-> (direcciones ecuatoriales (N,3), magnitudes (N,), colores BGR (N,3))."""
    rng = np.random.default_rng(semilla)
    ra, dec, mag, col = zip(*BRILLANTES)
    dirs = [_vector_ra_dec(ra, dec)]
    mags = [np.array(mag)]
    cols = [np.array([COLOR_ESTRELLA[c] for c in col])]

    # Debiles: N(<m) crece ~x2.8 por magnitud; mas densas cerca del plano galactico
    u = rng.random(n_debiles)
    k = 0.45 * math.log(10)
    m0 = 3.0
    m = m0 + np.log(1 + u * (math.exp(k * (mag_lim - m0)) - 1)) / k
    v = rng.normal(size=(n_debiles * 3, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    b_gal = np.abs((v @ ECU_A_GAL.T)[:, 2])                     # |sin(latitud galactica)|
    acepta = rng.random(len(v)) < (0.35 + 0.65 * np.exp(-(b_gal / 0.25) ** 2))
    v = v[acepta][:n_debiles]
    m = m[:len(v)]
    t = rng.random(len(v))
    c = np.where(t[:, None] < 0.22, COLOR_ESTRELLA["a"],
                 np.where(t[:, None] < 0.70, COLOR_ESTRELLA["b"],
                          np.where(t[:, None] < 0.93, COLOR_ESTRELLA["n"], COLOR_ESTRELLA["r"])))
    dirs.append(v)
    mags.append(m)
    cols.append(np.asarray(c, float))
    return np.concatenate(dirs), np.concatenate(mags), np.concatenate(cols)
