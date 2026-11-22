"""La Tierra vista desde una camara en orbita, dibujada solo con OpenCV.

La imagen es la de una camara que orbita (giro/inclinacion = donde esta la camara);
la luz viene del Sol real, asi que al orbitar ves pasar el dia, el terminador y la
noche con las luces de las ciudades (NASA Black Marble).

Como se hace rapido:
  - Una "tabla" precalculada guarda, para cada punto del disco, su normal 3D (x, y, z).
  - Inclinar = rotar esas normales alrededor del eje horizontal y convertirlas a
    latitud/longitud (2 addWeighted + 2 phase + 1 magnitude). Solo se recalcula
    cuando cambia la inclinacion; mientras solo gira se reutiliza.
  - Girar = sumar un desplazamiento a la longitud (eje polar de la Tierra).
  - Cada cuadro: 1 addWeighted (aplica el giro) + 1 cv2.remap de la textura.
    Todo en C, nada en Python puro.
  - La iluminacion depende del Sol: n.sol se calcula con 1 cv2.transform y todo lo
    demas (dia, ocaso naranja, luces de noche, reflejo del mar, atmosfera) son tablas LUT.
  - Piramides (mipmaps) de textura y tablas: sin parpadeo al alejar y el costo es
    proporcional al tamano en pantalla.
"""

import functools
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from universo import config as C

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


def _leer(ruta):
    """Imagen BGR o None. Rutas relativas a la carpeta del proyecto."""
    if not ruta:
        return None
    p = Path(ruta)
    p = p if p.is_absolute() else C.RAIZ / p
    if not p.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)   # rutas con acentos


def cargar_textura(ruta, respaldo=None):
    """Equirectangular BGR (ancho = 2 x alto). `ruta` puede ser una lista (se usa la primera que
    exista, p. ej. la de la NASA y luego la procedural). Si no hay ninguna: respaldo() o un planeta inventado."""
    rutas = ruta if isinstance(ruta, (list, tuple)) else [ruta]
    for r in rutas:
        img = _leer(r)
        if img is not None:
            return img, Path(r).name
    if respaldo is not None:
        return respaldo(), "procedural"
    return textura_procedural(), "procedural"


def mascara_oceano(bgr):
    """Donde hay mar (para el reflejo del sol). Mar = azul dominante y oscuro."""
    f = bgr.astype(np.float32)
    dif = f[..., 0] - np.maximum(f[..., 1], f[..., 2])
    lum = f.sum(-1)
    m = _suave(2.0, 12.0, dif) * (1.0 - _suave(330.0, 480.0, lum))
    sigma = max(bgr.shape[1] / 1000.0, 1.0)              # suave: el mar oscuro de la foto tiene bloques JPEG
    return (cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) * 255).astype(np.uint8)


def _piramide(img, ancho_min=128):
    """Niveles de la textura (cualquier numero de canales), reduciendo a la mitad con
    envoltura horizontal."""
    nivel = np.ascontiguousarray(img)
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
@functools.lru_cache(maxsize=None)
def niveles_esfera(R, atmosfera, interior):
    """Piramide de tablas de la esfera, COMPARTIDA entre astros iguales (~64 MB cada una con R=600):
    Tierra y Venus usan la misma, Mercurio y Marte otra. Solo se lee, nunca se modifica."""
    return tuple(piramide_tablas(*tablas_esfera(R, atmosfera=atmosfera, interior=interior)))


def tablas_esfera(R=600, margen=1.18, atmosfera=True, interior=True):
    """Tabla de normales (marco de la camara) y geometria fija del disco.
    La iluminacion ya NO se precalcula: depende de donde este el Sol respecto a la camara."""
    S = 2 * int(math.ceil(margen * R)) + 2
    c = (S - 1) / 2.0
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    u, v = (xs - c) / R, (ys - c) / R
    r = np.sqrt(u * u + v * v)

    # --- Normal 3D de cada punto (x derecha, y abajo, z hacia ti). Fuera del disco
    #     se prolonga el borde (asi el halo de la atmosfera "sabe" de que lado le da el Sol). ---
    rc = np.minimum(r, 0.9995)
    k = np.where(r > 1e-9, rc / np.maximum(r, 1e-9), 1.0)
    uc, vc = u * k, v * k
    zc = np.sqrt(np.clip(1.0 - uc * uc - vc * vc, 0.0, 1.0))
    tabla = np.dstack([uc, vc, zc]).astype(np.float32)

    # --- Geometria: borde del disco, opacidad y cuanta atmosfera atraviesa la vista ---
    z = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    a_disco = np.clip((1.0 - r) * R + 0.5, 0.0, 1.0)
    fuera = r > 1.0
    halo = np.exp(-(r - 1.0) / 0.03) * fuera * (0.85 if atmosfera else 0.0)
    halo *= np.clip((margen - 0.02 - r) / 0.06, 0.0, 1.0)
    halo *= 1.0 - a_disco
    alfa = np.clip(a_disco + 0.6 * halo, 0.0, 1.0)
    atm = (a_disco * (1.0 - z) ** 2.4 * 0.95 * (1.0 if interior else 0.0) + halo) * (1.0 if atmosfera else 0.0)

    geo = np.dstack([a_disco, alfa, np.clip(atm, 0, 1), np.zeros_like(r)]) * 255.0
    return tabla, (geo + 0.5).astype(np.uint8), R


def piramide_tablas(tabla, geo, R, paso=0.7071, radio_min=6.0):
    S = tabla.shape[0]
    niveles, f = [], 1.0
    while R * f >= radio_min:
        lado = max(int(round(S * f)), 8)
        if lado == S:
            t, g = tabla, geo
        else:
            t = cv2.resize(tabla, (lado, lado), interpolation=cv2.INTER_AREA)
            g = cv2.resize(geo, (lado, lado), interpolation=cv2.INTER_AREA)
        niveles.append((np.ascontiguousarray(t), np.ascontiguousarray(g), R * lado / S))
        f *= paso
    return niveles


# ----------------------------------------------------------------------
# Curvas de luz (tablas de 256 valores; el indice codifica un coseno: v = 127.5 * (1 + cos))
# ----------------------------------------------------------------------
_COS = np.arange(256, dtype=np.float64) / 127.5 - 1.0


def _luts_luz(contraluz, ocaso_naranja=True):
    """-> (dia BGR x200, noche BGR 0..255, dia gris 0..255, atmosfera 0..255) en funcion de n.sol."""
    x = _COS
    lambert = np.clip(x, 0.0, 1.0) ** 0.6
    dia = _suave(-0.10, 0.20, x) * (0.42 + 0.80 * lambert)
    # crepusculo: la luz que cruza mucha atmosfera se pone naranja junto al terminador
    ocaso = np.exp(-((x - 0.04) / 0.09) ** 2) * (1.0 if ocaso_naranja else 0.0)
    tinte = np.stack([1.0 - 0.45 * ocaso, 1.0 - 0.12 * ocaso, 1.0 + 0.25 * ocaso], -1)
    lut_dia = np.clip(dia[:, None] * tinte * ESCALA_MUL, 0, 255)
    # las ciudades se encienden al anochecer (despues del terminador)
    noche = 1.0 - _suave(-0.16, 0.03, x)
    lut_noche = np.clip(noche[:, None] * np.array([1.0, 1.0, 1.0]) * 255.0, 0, 255)
    lut_dia1 = np.clip(_suave(-0.02, 0.20, x) * 255.0, 0, 255)
    # atmosfera: brilla donde le da el Sol; a contraluz (Sol detras) se enciende todo el anillo
    atm = 0.10 + 0.90 * _suave(-0.30, 0.35, x)
    atm = np.maximum(atm, contraluz * np.exp(-(x / 0.35) ** 2))
    lut_atm = np.clip(atm * 255.0, 0, 255)
    a8 = lambda a: a.astype(np.uint8)
    return (a8(lut_dia).reshape(1, 256, 3), a8(lut_noche).reshape(1, 256, 3),
            a8(lut_dia1).reshape(1, 256, 1), a8(lut_atm).reshape(1, 256, 1))


def _lut_brillo():
    """Reflejo del Sol en el mar: n.h -> brillo. El indice es 255 * n.h (solo la parte
    positiva, el doble de precision que n.sol: sin anillos en el punto brillante)."""
    x = np.arange(256, dtype=np.float64) / 255.0
    return np.clip((0.30 * x ** 28 + 0.07 * x ** 6) * 255.0, 0, 255).astype(np.uint8).reshape(1, 256, 1)


LUT_BRILLO_MAR = _lut_brillo()


def iluminar_textura(dia, mar, noche, sol_tierra, ocaso_naranja=True, emisivo=False):
    """Hornea la luz del Sol en la textura (marco de la Tierra): dia, ocaso, noche con
    luces y, en el canal A, donde puede reflejar el mar. -> BGRA del mismo tamano.

    Esto no depende de la camara, solo de donde esta el Sol, que respecto a la Tierra
    se mueve 0.25 grados por minuto: se recalcula pocas veces y en otro hilo."""
    H, W = dia.shape[:2]
    if emisivo:                                           # una estrella no se ilumina: emite
        color = cv2.convertScaleAbs(dia, alpha=C.BRILLO_SOL)
        return np.ascontiguousarray(np.dstack([color, np.zeros((H, W), np.uint8)]))
    lat = (math.pi / 2 - (np.arange(H) + 0.5) / H * math.pi).astype(np.float32)
    lon = ((np.arange(W) + 0.5) / W * 2 * math.pi - math.pi).astype(np.float32)
    s = np.asarray(sol_tierra, np.float64)
    lat_s, lon_s = math.asin(float(np.clip(s[2], -1, 1))), math.atan2(s[1], s[0])
    # n.sol = cos(lat) cos(lat_s) cos(lon - lon_s) + sin(lat) sin(lat_s)  (separable: filas x columnas)
    a = (np.cos(lat) * math.cos(lat_s) * 127.5)[:, None]
    b = np.cos(lon - lon_s)[None, :]
    c = (np.sin(lat) * math.sin(lat_s) * 127.5 + 127.5)[:, None]
    nd8 = np.clip(a * b + c, 0, 255).astype(np.uint8)
    lut_dia, lut_noche, lut_dia1, _ = _luts_luz(0.0, ocaso_naranja)
    nd3 = cv2.cvtColor(nd8, cv2.COLOR_GRAY2BGR)
    color = cv2.multiply(dia, cv2.LUT(nd3, lut_dia), scale=1.0 / ESCALA_MUL)
    if noche is not None:
        color = cv2.add(color, cv2.multiply(noche, cv2.LUT(nd3, lut_noche), scale=C.BRILLO_LUCES / 255.0))
    reflejo = cv2.multiply(mar, cv2.LUT(nd8, lut_dia1), scale=1.0 / 255.0)
    return np.ascontiguousarray(np.dstack([color, reflejo]))


# ----------------------------------------------------------------------
class Tierra:
    def __init__(self, ruta_textura=None, R=600, ruta_noche=None, atmosfera=True, emisivo=False,
                 color_atm=None, oceanos=True, desfase_lon=0.0, respaldo=None):
        """emisivo: el Sol (brilla por si mismo, se oscurece hacia el borde, halo calido).
        oceanos=False: sin reflejo del Sol (Luna, Marte). desfase_lon: grados que se gira la textura
        si su longitud 0 no esta al centro de la imagen. respaldo: funcion que la genera si no hay archivo."""
        self.atmosfera = atmosfera                        # la Luna no tiene: ni halo azul ni ocaso naranja
        self.emisivo = emisivo
        self._atm = np.array(ATM if color_atm is None else color_atm, np.float32)
        bgr, self.origen = cargar_textura(ruta_textura, respaldo)
        if bgr.shape[1] != 2 * bgr.shape[0]:
            bgr = cv2.resize(bgr, (2 * bgr.shape[0], bgr.shape[0]), interpolation=cv2.INTER_AREA)
        if desfase_lon:
            bgr = np.ascontiguousarray(np.roll(bgr, int(round(desfase_lon / 360.0 * bgr.shape[1])), axis=1))
        self._dia = bgr
        self._mar = mascara_oceano(bgr) if (oceanos and not emisivo) else np.zeros(bgr.shape[:2], np.uint8)
        self._noche, self.origen_noche = None, "sin luces"
        noche = _leer(ruta_noche)
        if noche is not None:
            if noche.shape[:2] != bgr.shape[:2]:
                noche = cv2.resize(noche, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
            self._noche, self.origen_noche = noche, Path(ruta_noche).name
        self.niveles = niveles_esfera(R, bool(atmosfera), not emisivo)
        self._cache = None
        self._cache_incl = None
        self._luts = (None, None)

        # textura iluminada (piramide BGRA). Se rehace en otro hilo cuando el Sol se mueve.
        self.texturas = None                              # se hornea con el primer Sol que llegue
        self._sol_horneado = None
        self._hilo = None
        self._lock_primera = threading.Lock()

    # ------------------------------------------------------------------
    def iluminar(self, sol_tierra, esperar=False):
        """Pide hornear la luz para esta posicion del Sol (marco de la Tierra). Solo trabaja si
        el Sol se movio mas de config.SOL_REHORNEAR_GRADOS desde la ultima vez."""
        s = np.asarray(sol_tierra, np.float64)
        s = s / np.linalg.norm(s)
        if self.texturas is None:                         # primera vez: hornear ya, o esperar a otro hilo
            with self._lock_primera:                      # (p. ej. el precalentado) que ya este en ello;
                if self.texturas is None:                 # si no, dibujar() veria texturas = None
                    self._sol_horneado = s
                    lit = iluminar_textura(self._dia, self._mar, self._noche, s, self.atmosfera, self.emisivo)
                    self.texturas = _piramide(lit)
            return
        if self.emisivo:
            return
        if self._sol_horneado is not None:
            ang = math.degrees(math.acos(float(np.clip(s @ self._sol_horneado, -1, 1))))
            if ang < C.SOL_REHORNEAR_GRADOS:
                return
        esperar = esperar or self.texturas is None        # la primera vez no hay nada que mostrar aun
        if self._hilo is not None and self._hilo.is_alive():
            if not esperar:
                return
            self._hilo.join()
        self._sol_horneado = s

        def trabajo():
            lit = iluminar_textura(self._dia, self._mar, self._noche, s, self.atmosfera, self.emisivo)
            self.texturas = _piramide(lit)               # cambio atomico: el dibujo nunca ve media textura

        if esperar:
            trabajo()
        else:
            self._hilo = threading.Thread(target=trabajo, daemon=True, name="iluminar_tierra")
            self._hilo.start()

    def dibujar(self, buf, cx, cy, radio, giro, inclinacion=0.0, sol=(-0.62, -0.38, 0.69), sol_tierra=None,
                rodar=0.0):
        """Compone la Tierra sobre buf (BGR uint8) centrada en (cx, cy).
        giro en radianes: positivo = la superficie de enfrente se mueve a la derecha.
        inclinacion en radianes: positivo = la superficie de enfrente sube (se ve el sur).
        rodar en radianes: cuanto esta girada la camara sobre su eje de vista (0 = norte arriba).
        sol: direccion hacia el Sol en el marco de la camara (x derecha, y abajo, z hacia ti).
        sol_tierra: la misma direccion en el marco de la Tierra (para hornear el dia y la noche)."""
        if sol_tierra is not None:
            self.iluminar(sol_tierra)
        elif self.texturas is None:
            self.iluminar(np.array([0.70, -0.42, 0.58]))  # sin Sol real: de frente, arriba a la izquierda
        if radio < 1.0:
            return
        alto, ancho = buf.shape[:2]

        radio = round(radio * 20.0) / 20.0               # permite reutilizar la geometria entre cuadros
        clave = (radio, round(cx * 20.0), round(cy * 20.0), ancho, alto)
        if self._cache is None or self._cache[0] != clave:
            self._cache = (clave,) + self._preparar(radio, cx, cy, ancho, alto)
            self._cache_incl = None
        _, region, normales, mitad, a_disco, w_disco, w_fondo, atm_mag, limbo = self._cache
        if region is None:
            return
        x0, y0, x1, y1 = region
        w, h = x1 - x0, y1 - y0

        # textura: ~1 texel por pixel en el ecuador
        texturas = self.texturas
        tex = texturas[0]
        for cand in texturas:
            if cand.shape[1] >= 2 * math.pi * radio:
                tex = cand
            else:
                break
        Ht, Wt = tex.shape[:2]

        # lon/lat de cada pixel: solo se recalcula si cambia la inclinacion (~1/4 de pixel)
        paso = 0.25 / max(radio, 1.0)
        incl = round(inclinacion / paso) * paso
        rod = round(rodar / paso) * paso
        if self._cache_incl is None or self._cache_incl[:3] != (incl, rod, Ht):
            lon, colat = self._latlon(normales, incl, rod)
            # colatitud [0, pi] -> fila de la textura, sin tocar la fila del polo opuesto
            mapa_y = cv2.addWeighted(colat, Ht * 0.999 / math.pi, colat, 0.0, Ht * 0.0005 - 0.5)
            self._cache_incl = (incl, rod, Ht, lon, mapa_y)
        _, _, _, lon, mapa_y = self._cache_incl

        # girar = desplazar la longitud: lon [0, 2pi) -> columna de la textura (lon 0 a la mitad)
        desplaz = -(giro % (2 * math.pi)) / (2 * math.pi) * Wt
        mapa_x = cv2.addWeighted(lon, Wt / (2 * math.pi), lon, 0.0, Wt / 2 + desplaz - 0.5)
        muestra = cv2.remap(tex, mapa_x, mapa_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        color = cv2.cvtColor(muestra, cv2.COLOR_BGRA2BGR)
        if limbo is not None:                             # estrella: se oscurece hacia el borde
            color = cv2.multiply(color, limbo, scale=1.0 / 255.0)

        # --- Lo que depende de la vista (a media resolucion: es suave) ---
        s = np.asarray(sol, np.float64)
        s = s / (np.linalg.norm(s) + 1e-12)
        hv = s + np.array([0.0, 0.0, 1.0])
        hv = hv / np.linalg.norm(hv) if np.linalg.norm(hv) > 1e-6 else np.array([0.0, 0.0, 1.0])
        m_nd = np.array([[127.5 * s[0], 127.5 * s[1], 127.5 * s[2], 127.5]], np.float32)
        m_nh = np.array([[255.0 * hv[0], 255.0 * hv[1], 255.0 * hv[2], 0.0]], np.float32)
        contraluz = round(float(np.clip(-s[2], 0.0, 1.0)) ** 3 * 1.6, 2)
        if self._luts[0] != contraluz:
            self._luts = (contraluz, _luts_luz(contraluz)[3])
        lut_atm = self._luts[1]

        # atmosfera (azul) + reflejo del mar (blanco), todo a media resolucion y luego se escala
        hm, wm = mitad.shape[:2]
        nd8 = cv2.convertScaleAbs(cv2.transform(mitad, m_nd))
        nh8 = cv2.convertScaleAbs(cv2.max(cv2.transform(mitad, m_nh), 0.0))   # solo la parte positiva
        atm = atm_mag if self.emisivo else cv2.multiply(atm_mag, cv2.LUT(nd8, lut_atm), scale=1.0 / 255.0)
        mar = cv2.resize(cv2.extractChannel(muestra, 3), (wm, hm), interpolation=cv2.INTER_AREA)
        brillo = cv2.multiply(cv2.multiply(mar, cv2.LUT(nh8, LUT_BRILLO_MAR), scale=1.0 / 255.0),
                              a_disco, scale=1.0 / 255.0)   # el reflejo no sale del disco
        resplandor = cv2.merge([cv2.addWeighted(atm, float(self._atm[0]), brillo, 1.0, 0.0),
                                cv2.addWeighted(atm, float(self._atm[1]), brillo, 1.0, 0.0),
                                cv2.addWeighted(atm, float(self._atm[2]), brillo, 1.0, 0.0)])
        resplandor = cv2.resize(resplandor, (w, h), interpolation=cv2.INTER_LINEAR)

        if w_fondo is None:                               # la Tierra llena toda la region (zoom grande)
            cv2.add(color, resplandor, dst=buf[y0:y1, x0:x1])
        else:
            roi = buf[y0:y1, x0:x1]
            buf[y0:y1, x0:x1] = cv2.add(cv2.blendLinear(color, roi, w_disco, w_fondo), resplandor)

    @staticmethod
    def _latlon(normales, incl, rodar=0.0):
        """Rueda (rodar) e inclina las normales (rotacion alrededor del eje x de la pantalla) y las
        pasa a (longitud [0, 2pi), colatitud [0, pi]) en radianes. Solo OpenCV sobre planos sueltos."""
        nx, ny, nz = normales
        if abs(rodar) > 1e-4:                             # camara girada sobre su eje de vista
            c, s = math.cos(rodar), math.sin(rodar)
            nx, ny = cv2.addWeighted(nx, c, ny, s, 0.0), cv2.addWeighted(nx, -s, ny, c, 0.0)
        c, s = math.cos(incl), math.sin(incl)
        menos_y = cv2.addWeighted(ny, -c, nz, -s, 0.0)    # -y de la Tierra (norte = arriba)
        z = cv2.addWeighted(ny, -s, nz, c, 0.0)
        lon = cv2.phase(z, nx)                            # atan2(x, z): este a la derecha
        colat = cv2.phase(menos_y, cv2.magnitude(nx, z))  # 0 = polo norte, pi = polo sur
        return lon, colat

    def _preparar(self, radio, cx, cy, ancho, alto):
        """Lo que solo depende del tamano y la posicion (no del giro, la inclinacion ni el Sol):
        se calcula una vez y se reutiliza mientras la Tierra no cambie de tamano."""
        tabla, geo, Rl = self.niveles[0]
        for niv in self.niveles:                          # el mas chico que aun alcanza
            if niv[2] >= radio:
                tabla, geo, Rl = niv
            else:
                break
        s = radio / Rl
        lado = tabla.shape[0]
        c = (lado - 1) / 2.0
        mitad = s * lado / 2.0 + 2.0
        x0, x1 = max(int(cx - mitad), 0), min(int(cx + mitad) + 1, ancho)
        y0, y1 = max(int(cy - mitad), 0), min(int(cy + mitad) + 1, alto)
        if x1 <= x0 or y1 <= y0:
            return (None,) * 8
        w, h = x1 - x0, y1 - y0
        M = np.array([[s, 0, cx - s * c - x0], [0, s, cy - s * c - y0]], np.float32)
        t = cv2.warpAffine(tabla, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        g = cv2.warpAffine(geo, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a_disco, alfa, atm, _ = cv2.split(g)
        tam_mitad = (max(w // 2, 1), max(h // 2, 1))
        t_mitad = cv2.resize(t, tam_mitad, interpolation=cv2.INTER_AREA)
        a_mitad = cv2.resize(a_disco, tam_mitad, interpolation=cv2.INTER_AREA)
        atm_mitad = cv2.resize(atm, tam_mitad, interpolation=cv2.INTER_AREA)
        w_disco = a_disco.astype(np.float32) / 255.0
        w_fondo = 1.0 - alfa.astype(np.float32) / 255.0
        if int(a_disco.min()) == 255:                     # todo es Tierra: no hace falta mezclar con el fondo
            w_fondo = None
        limbo = None
        if self.emisivo:                                  # oscurecimiento al borde: 1 en el centro, ~0.45 en el limbo
            mu = np.clip(t[..., 2], 0.0, 1.0)
            l8 = np.clip((0.40 + 0.60 * mu ** 0.65) * 255.0, 0, 255).astype(np.uint8)
            limbo = cv2.merge([l8, l8, l8])
        return ((x0, y0, x1, y1), tuple(cv2.split(t)), t_mitad, a_mitad, w_disco, w_fondo, atm_mitad, limbo)
