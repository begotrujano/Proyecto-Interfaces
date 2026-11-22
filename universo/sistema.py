"""El Sistema Solar: donde esta cada planeta ahora mismo y como esta orientado.

Solo depende de numpy (y astros.py). Todo en unidades de astros.py: el marco de la Tierra
("ECEF"), distancias en radios terrestres. Para las orbitas se usan unidades astronomicas
(UA) en el marco ECLIPTICO heliocentrico (x hacia el equinoccio, z hacia el polo de la
eclíptica), que es el plano natural del sistema solar.

Efemerides: elementos keplerianos de E. M. Standish (JPL, "Keplerian Elements for
Approximate Positions of the Major Planets", valido 1800-2050): ~0.01 grados para los
planetas interiores y ~0.05 para los exteriores. Mas que suficiente para ver cada planeta
donde esta de verdad, su orbita elíptica y su fase/ilumination reales.

Agregar un planeta nuevo al programa = ponerlo en config.PLANETAS_ACTIVOS y darle datos
fisicos en FISICOS (radio, polo y rotacion) y una textura (config.TEXTURA_<PLANETA> y un Tierra(...) en render.Escena.__init__).
"""

import math

import numpy as np

from universo import astros as A

_RAD = math.pi / 180.0
UA_RT = A.UA_KM / A.RADIO_TIERRA_KM          # radios terrestres en una UA (~23 455)
ORDEN = ["mercurio", "venus", "tierra", "marte", "jupiter", "saturno", "urano", "neptuno"]
NOMBRES = {"sol": "Sol", "mercurio": "Mercurio", "venus": "Venus", "tierra": "Tierra", "luna": "Luna",
           "marte": "Marte", "jupiter": "Júpiter", "saturno": "Saturno", "urano": "Urano", "neptuno": "Neptuno"}
ARTICULOS = {"sol": "el", "luna": "la", "tierra": "la"}   # "Viajando a la Luna", "a Marte" (sin articulo)
COLORES = {"mercurio": (150, 150, 160), "venus": (175, 215, 240), "tierra": (230, 170, 90),
           "marte": (70, 110, 205), "jupiter": (150, 185, 215), "saturno": (150, 200, 225),
           "urano": (220, 225, 160), "neptuno": (230, 140, 80), "sol": (200, 235, 255), "luna": (200, 200, 200)}

# (valor en J2000, cambio por siglo) de: a [UA], e, I [grados], L [grados], long. del perihelio, long. del nodo
ELEMENTOS = {
    "mercurio": ((0.38709927, 0.00000037), (0.20563593, 0.00001906), (7.00497902, -0.00594749),
                 (252.25032350, 149472.67411175), (77.45779628, 0.16047689), (48.33076593, -0.12534081)),
    "venus": ((0.72333566, 0.00000390), (0.00677672, -0.00004107), (3.39467605, -0.00078890),
              (181.97909950, 58517.81538729), (131.60246718, 0.00268329), (76.67984255, -0.27769418)),
    "tierra": ((1.00000261, 0.00000562), (0.01671123, -0.00004392), (-0.00001531, -0.01294668),
               (100.46457166, 35999.37244981), (102.93768193, 0.32327364), (0.0, 0.0)),
    "marte": ((1.52371034, 0.00001847), (0.09339410, 0.00007882), (1.84969142, -0.00813131),
              (-4.55343205, 19140.30268499), (-23.94362959, 0.44441088), (49.55953891, -0.29257343)),
    "jupiter": ((5.20288700, -0.00011607), (0.04838624, -0.00013253), (1.30439695, -0.00183714),
                (34.39644051, 3034.74612775), (14.72847983, 0.21252668), (100.47390909, 0.20469106)),
    "saturno": ((9.53667594, -0.00125060), (0.05386179, -0.00050991), (2.48599187, 0.00193609),
                (49.95424423, 1222.49362201), (92.59887831, -0.41897216), (113.66242448, -0.28867794)),
    "urano": ((19.18916464, -0.00196176), (0.04725744, -0.00004397), (0.77263783, -0.00242939),
              (313.23810451, 428.48202785), (170.95427630, 0.40805281), (74.01692503, 0.04240589)),
    "neptuno": ((30.06992276, 0.00026291), (0.00859048, 0.00005105), (1.77004347, 0.00035372),
                (-55.12002969, 218.45945325), (44.96476227, -0.32241464), (131.78422574, -0.00508664)),
}
PRECESION_GRADOS_SIGLO = 1.396971     # los elementos son de J2000; esto los lleva al equinoccio de la fecha

# Datos fisicos de los astros que se dibujan como esfera (IAU 2009): radio medio [km], polo
# (ascension recta, declinacion: valor J2000 y cambio por siglo) y meridiano 0 (W0 y grados por dia).
FISICOS = {
    "sol": dict(radio_km=A.RADIO_SOL_KM, ra=(286.13, 0.0), dec=(63.87, 0.0), w=(84.176, 14.1844000)),
    "mercurio": dict(radio_km=2439.7, ra=(281.0097, -0.0328), dec=(61.4143, -0.0049), w=(329.5469, 6.1385025)),
    # Venus gira al reves (retrogrado): su meridiano 0 avanza hacia atras, una vuelta cada 243 dias
    "venus": dict(radio_km=6051.8, ra=(272.76, 0.0), dec=(67.16, 0.0), w=(160.20, -1.4813688)),
    "marte": dict(radio_km=3389.5, ra=(317.68143, -0.1061), dec=(52.88650, -0.0609), w=(176.630, 350.89198226)),
}


def _rz(a):                       # rotacion pasiva (de marcos) alrededor de z
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _elementos(nombre, t):
    """Elementos a la fecha (t = siglos desde J2000): a, e, I, L, perihelio, nodo (angulos en radianes)."""
    a, e, i, L, w, n = [v0 + v1 * t for v0, v1 in ELEMENTOS[nombre]]
    return a, e, i * _RAD, L * _RAD, w * _RAD, n * _RAD


def _kepler(M, e):
    """Anomalia excentrica para la anomalia media M (radianes)."""
    M = (M + math.pi) % (2 * math.pi) - math.pi
    E = M + e * math.sin(M) if e < 0.8 else math.pi
    for _ in range(12):
        d = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < 1e-12:
            break
    return E


def _plano_a_ecliptica(x, y, i, w_per, nodo):
    """Pasa (x, y) del plano orbital al marco ecliptico (J2000). Acepta arreglos."""
    w = w_per - nodo
    cw, sw, cn, sn, ci, si = math.cos(w), math.sin(w), math.cos(nodo), math.sin(nodo), math.cos(i), math.sin(i)
    X = (cw * cn - sw * sn * ci) * x + (-sw * cn - cw * sn * ci) * y
    Y = (cw * sn + sw * cn * ci) * x + (-sw * sn + cw * cn * ci) * y
    Z = (sw * si) * x + (cw * si) * y
    return np.stack([X, Y, Z], -1)


def _precesion(v, t):
    # v @ R = R.T @ v: con R = _rz(+p) los vectores giran +p alrededor del polo, asi que las longitudes
    # crecen p (antes iba con -p y todo quedaba 2p corrido: ~0.5 grados en 2019, validado con los transitos)
    return v @ _rz(PRECESION_GRADOS_SIGLO * _RAD * t)


def heliocentrica(nombre, jd):
    """Posicion del planeta respecto al Sol, en UA, marco ecliptico de la fecha."""
    t = (jd - 2451545.0) / 36525.0
    a, e, i, L, w, n = _elementos(nombre, t)
    E = _kepler(L - w, e)
    x, y = a * (math.cos(E) - e), a * math.sqrt(1.0 - e * e) * math.sin(E)
    return _precesion(_plano_a_ecliptica(x, y, i, w, n), t)


def orbita(nombre, jd, n=240):
    """Puntos de la orbita elíptica completa (n, 3), en UA, marco ecliptico de la fecha."""
    t = (jd - 2451545.0) / 36525.0
    a, e, i, L, w, nodo = _elementos(nombre, t)
    E = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    x, y = a * (np.cos(E) - e), a * math.sqrt(1.0 - e * e) * np.sin(E)
    return _precesion(_plano_a_ecliptica(x, y, i, w, nodo), t)


def ecliptica_a_tierra(jd):
    """Matriz que pasa vectores del marco ecliptico de la fecha al marco de la Tierra."""
    eps = A._oblicuidad(jd)
    ce, se = math.cos(eps), math.sin(eps)
    a_ecuatorial = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return A.ecuatorial_a_tierra(jd) @ a_ecuatorial


def geocentrica(nombre, jd):
    """Vector Tierra -> planeta en el marco de la Tierra, en radios terrestres."""
    v = heliocentrica(nombre, jd) - heliocentrica("tierra", jd)
    return ecliptica_a_tierra(jd) @ v * UA_RT


def posicion(nombre, jd):
    """Centro de un astro en el marco de la Tierra (radios terrestres)."""
    if nombre == "sol":
        return A.sol_posicion(jd)
    if nombre == "luna":
        return A.luna(jd)
    if nombre == "tierra":
        return np.zeros(3)
    return geocentrica(nombre, jd)


def rotacion(nombre, jd):
    """Matriz que pasa vectores del marco de la Tierra al marco del astro (x en su longitud 0,
    z hacia su polo norte), segun los elementos de rotacion de la IAU."""
    f = FISICOS[nombre]
    t = (jd - 2451545.0) / 36525.0
    d = jd - 2451545.0
    ra = (f["ra"][0] + f["ra"][1] * t) * _RAD
    dec = (f["dec"][0] + f["dec"][1] * t) * _RAD
    W = ((f["w"][0] + f["w"][1] * d) % 360.0) * _RAD
    icrf_a_astro = _rz(W) @ _rx(math.pi / 2 - dec) @ _rz(math.pi / 2 + ra)
    return icrf_a_astro @ A.ecuatorial_a_tierra(jd).T


def radio_rt(nombre):
    """Radio de un astro en radios terrestres."""
    if nombre == "luna":
        return A.RADIO_LUNA_KM / A.RADIO_TIERRA_KM
    if nombre == "tierra":
        return 1.0
    return FISICOS[nombre]["radio_km"] / A.RADIO_TIERRA_KM
