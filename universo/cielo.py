"""El cielo que rodea a la Tierra: Via Lactea, estrellas, el Sol y la Luna.

Todo esta fijo en el espacio (marco ecuatorial) y se proyecta con la camara, asi
que al orbitar la Tierra las estrellas se mueven junto con ella, como en la realidad.

Rapidez:
  - Estrellas: proyeccion de ~9 000 vectores (una multiplicacion de matrices) y
    "estampado" de manchas gaussianas de 5x5 directo sobre sus pixeles (sin bucles).
  - Via Lactea: textura esferica muestreada a 1/16 de resolucion y escalada (es difusa).
  - Sol: sprite precalculado (disco + resplandor) que se suma; solo cambia de lugar.
  - Luna: esfera pequena con numpy sobre su recuadro (casi siempre < 40 px).
"""

import math

import cv2
import numpy as np

from universo import astros as A
from universo import config as C


def _suave(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ruido(forma, rng, sigmas, pesos):
    """Ruido suave que da la vuelta en horizontal (sin costura en la longitud 180)."""
    total = np.zeros(forma, np.float32)
    pad = int(3 * max(sigmas)) + 1
    for s, p in zip(sigmas, pesos):
        r = rng.random(forma).astype(np.float32)
        r = np.concatenate([r[:, -pad:], r, r[:, :pad]], axis=1)
        total += p * cv2.GaussianBlur(r, (0, 0), s, borderType=cv2.BORDER_REFLECT)[:, pad:-pad]
    total -= total.mean()
    return total / (total.std() + 1e-6)


# ----------------------------------------------------------------------
# Via Lactea (textura equirectangular en coordenadas ecuatoriales)
# ----------------------------------------------------------------------
def textura_via_lactea(ancho=1024, semilla=3):
    alto = ancho // 2
    rng = np.random.default_rng(semilla)
    ra = (np.arange(ancho) + 0.5) / ancho * 2 * math.pi
    dec = math.pi / 2 - (np.arange(alto) + 0.5) / alto * math.pi
    RA, DEC = np.meshgrid(ra, dec)
    v = np.stack([np.cos(DEC) * np.cos(RA), np.cos(DEC) * np.sin(RA), np.sin(DEC)], -1)
    g = v @ A.ECU_A_GAL.T
    b = np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1)))         # latitud galactica
    l = np.degrees(np.arctan2(g[..., 1], g[..., 0]))             # longitud galactica (0 = centro)

    ancho_banda = 7.0 + 5.0 * np.exp(-(l / 40.0) ** 2)
    disco = np.exp(-(b / ancho_banda) ** 2) * (0.35 + 0.65 * np.exp(-(l / 70.0) ** 2))
    bulbo = np.exp(-((l / 14.0) ** 2 + (b / 9.0) ** 2))
    n = _ruido((alto, ancho), rng, (3.0, 8.0, 18.0), (0.25, 0.45, 0.30))
    nubes = np.clip(0.78 + 0.38 * n, 0.0, 1.6)
    polvo = 1.0 - 0.75 * np.exp(-((b + 0.8) / 1.6) ** 2) * np.exp(-(l / 55.0) ** 2) * np.clip(0.6 + 0.5 * n, 0, 1)
    brillo = (disco * nubes + 0.9 * bulbo) * polvo
    brillo = brillo / brillo.max()
    color = np.stack([0.80 + 0.20 * (1 - bulbo), 0.86 + 0.0 * bulbo, 0.90 + 0.10 * bulbo], -1)   # BGR, bulbo calido
    img = brillo[..., None] * color * C.BRILLO_VIA_LACTEA
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Sprite del Sol
# ----------------------------------------------------------------------
def sprite_sol(radio_disco, radio_total):
    S = 2 * int(radio_total) + 1
    c = S // 2
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = xs - c, ys - c
    r = np.sqrt(dx * dx + dy * dy)
    rd = max(radio_disco, 0.8)
    disco = np.clip(rd + 0.5 - r, 0.0, 1.0) * 3.0                  # el disco satura (blanco puro)
    corona = 1.6 / (1.0 + (r / (1.8 * rd)) ** 2.2)
    resplandor = 0.55 * np.exp(-r / (0.05 * radio_total)) + 0.16 * np.exp(-r / (0.22 * radio_total))
    rayos = np.zeros_like(r)
    for a0 in (math.pi / 4, 3 * math.pi / 4):                   # 4 rayos tenues (difraccion)
        perp = np.abs(dx * math.sin(a0) - dy * math.cos(a0))
        rayos += np.exp(-(perp / 0.8) ** 2) * np.exp(-r / (0.12 * radio_total))
    rayos *= 0.30
    borde = np.clip((radio_total - r) / (0.15 * radio_total), 0.0, 1.0)
    blanco = (disco + corona) * borde
    calido = (resplandor + rayos) * borde
    img = (blanco[..., None] * np.array([1.0, 1.0, 1.0])
           + calido[..., None] * np.array([0.72, 0.88, 1.0]))      # BGR: resplandor amarillento
    return img.astype(np.float32)


# ----------------------------------------------------------------------
# Textura de la Luna (procedural: mares oscuros sobre tierras altas grises)
# ----------------------------------------------------------------------
# (latitud, longitud, radio en grados) de los mares principales de la cara visible
MARES = [(25, -15, 18), (8, 30, 14), (30, 15, 10), (17, 59, 9), (-3, 40, 9), (-10, -15, 11),
         (-20, -38, 9), (-20, 25, 8), (5, -50, 22), (55, 0, 8), (-15, 55, 6), (10, -20, 6)]


def textura_luna(ancho=512, semilla=9):
    alto = ancho // 2
    rng = np.random.default_rng(semilla)
    lon = (np.arange(ancho) + 0.5) / ancho * 360.0 - 180.0
    lat = 90.0 - (np.arange(alto) + 0.5) / alto * 180.0
    LON, LAT = np.meshgrid(lon, lat)
    la, lo = np.radians(LAT), np.radians(LON)
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    n = _ruido((alto, ancho), rng, (1.0, 3.0, 8.0), (0.3, 0.4, 0.3))
    albedo = 0.80 + 0.06 * n
    for la, lo, r in MARES:
        c = A.punto_superficie(la, lo)
        d = np.degrees(np.arccos(np.clip(v @ c, -1, 1)))
        albedo -= 0.30 * _suave(r * 1.15, r * 0.65, d + 4.0 * n)
    for _ in range(60):                                          # crateres brillantes
        la, lo = rng.uniform(-80, 80), rng.uniform(-180, 180)
        c = A.punto_superficie(la, lo)
        d = np.degrees(np.arccos(np.clip(v @ c, -1, 1)))
        r = rng.uniform(0.8, 3.0)
        albedo += 0.10 * np.exp(-(d / r) ** 2)
    return np.clip(albedo, 0.25, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
class Cielo:
    def __init__(self, ancho, alto, focal_px):
        self.w, self.h, self.f = ancho, alto, focal_px
        dirs, mags, cols = A.catalogo_estrellas()
        self.dirs = dirs.astype(np.float64)
        flujo = 10.0 ** (-0.4 * (mags - 1.0))                   # magnitud 1 -> 1
        intens = np.minimum(flujo, 6.0) ** C.CONTRASTE_ESTRELLAS * C.BRILLO_ESTRELLAS
        self.color = (cols * intens[:, None]).astype(np.float32)   # energia total por estrella (BGR)
        self.sigma = (0.48 + 0.30 * np.clip(np.log10(flujo + 1e-6) + 1.0, 0.0, 1.6)).astype(np.float32)
        self.brillantes = np.where(mags < 1.6)[0]

        self.via = textura_via_lactea()
        # malla de rayos a 1/16 de resolucion, en los puntos exactos que usa cv2.resize al escalar x16
        p = 16
        gw, gh = -(-ancho // p), -(-alto // p)
        gx, gy = np.meshgrid((np.arange(gw) + 0.5) * p - 0.5, (np.arange(gh) + 0.5) * p - 0.5)
        self._rayos_px = np.stack([gx, gy], -1).astype(np.float64)
        self._via_tam = (gw * p, gh * p)

        self.sol_radio_px = focal_px * math.tan(math.radians(C.SOL_ANGULO_RADIO_GRADOS))
        self.sol = sprite_sol(self.sol_radio_px, C.SOL_RESPLANDOR_REL * alto)
        self._sol_u8 = {}
        self.luna_tex = textura_luna()

        # huella de 5x5 para las estrellas
        o = np.arange(-2, 3)
        self._ox, self._oy = np.meshgrid(o, o)
        self._ox, self._oy = self._ox.ravel(), self._oy.ravel()

    # ------------------------------------------------------------------
    def fondo(self, buf, cam, eq_a_tierra):
        """Via Lactea + estrellas sobre buf (se sobrescribe)."""
        R = cam.M @ eq_a_tierra                                 # ecuatorial -> camara
        self._via_lactea(buf, cam, R)
        self._estrellas(buf, cam, R)

    def _via_lactea(self, buf, cam, R):
        px = self._rayos_px
        d = np.empty(px.shape[:2] + (3,))
        d[..., 0] = px[..., 0] - cam.cx
        d[..., 1] = px[..., 1] - cam.cy
        d[..., 2] = -cam.f
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        e = d @ R                                               # camara -> ecuatorial (R ortonormal)
        ra = np.arctan2(e[..., 1], e[..., 0]) % (2 * math.pi)
        dec = np.arcsin(np.clip(e[..., 2], -1, 1))
        Ht, Wt = self.via.shape[:2]
        mx = (ra / (2 * math.pi) * Wt - 0.5).astype(np.float32)
        my = ((math.pi / 2 - dec) / math.pi * Ht - 0.5).astype(np.float32)
        chico = cv2.remap(self.via, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        chico = cv2.GaussianBlur(chico, (3, 3), 0.8)            # sin "rombos" al escalar
        if self._via_tam == (self.w, self.h):
            cv2.resize(chico, self._via_tam, dst=buf, interpolation=cv2.INTER_LINEAR)
        else:
            buf[:] = cv2.resize(chico, self._via_tam, interpolation=cv2.INTER_LINEAR)[:self.h, :self.w]

    def _estrellas(self, buf, cam, R):
        """Cada estrella es una mancha gaussiana de 5x5 con posicion sub-pixel, sumada
        directamente sobre sus 25 pixeles (sin buferes de pantalla completa)."""
        c = self.dirs @ R.T
        z = -c[:, 2]
        ok = z > 0.05
        idx = np.nonzero(ok)[0]
        x = cam.cx + cam.f * c[idx, 0] / z[idx]
        y = cam.cy + cam.f * c[idx, 1] / z[idx]
        dentro = (x > 2) & (x < self.w - 3) & (y > 2) & (y < self.h - 3)
        idx, x, y = idx[dentro], x[dentro], y[dentro]
        if len(idx) == 0:
            return
        xi, yi = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        fx, fy = (x - xi - 0.5).astype(np.float32), (y - yi - 0.5).astype(np.float32)
        sig2 = 2.0 * self.sigma[idx] ** 2
        g = np.exp(-((self._ox[None, :] - fx[:, None]) ** 2 + (self._oy[None, :] - fy[:, None]) ** 2)
                   / sig2[:, None])
        g /= g.sum(axis=1, keepdims=True)
        plano = buf.reshape(-1, 3)
        lin = ((yi[:, None] + self._oy[None, :]) * self.w + xi[:, None] + self._ox[None, :]).ravel()
        val = (g[:, :, None] * self.color[idx][:, None, :]).reshape(-1, 3)
        # (si dos estrellas comparten pixel gana una: con miles de pixeles libres casi nunca pasa)
        plano[lin] = np.minimum(plano[lin] + val, 255.0).astype(np.uint8)
        for i in np.nonzero(np.isin(idx, self.brillantes))[0]:   # resplandor de Sirio, Canopo, Vega...
            self._mancha(buf, x[i], y[i], 3.0, self.color[idx[i]] * 0.06)

    def _mancha(self, buf, cx, cy, sigma, color):
        rad = int(3 * sigma) + 1
        x0, y0 = int(cx) - rad, int(cy) - rad
        gx = np.arange(x0, x0 + 2 * rad + 1)
        gy = np.arange(y0, y0 + 2 * rad + 1)
        xa, xb = max(x0, 0), min(x0 + 2 * rad + 1, self.w)
        ya, yb = max(y0, 0), min(y0 + 2 * rad + 1, self.h)
        if xb <= xa or yb <= ya:
            return
        g = np.exp(-((gx[None, xa - x0:xb - x0] + 0.5 - cx) ** 2 + (gy[ya - y0:yb - y0, None] + 0.5 - cy) ** 2)
                   / (2 * sigma ** 2))
        roi = buf[ya:yb, xa:xb]
        roi[:] = np.minimum(roi + g[..., None] * color, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    def dibujar_sol(self, buf, x, y, visible):
        """Suma el Sol (disco + resplandor) en (x, y). visible 0..1 (tapado por la Tierra)."""
        if visible <= 0.01:
            return
        nivel = round(visible * 20) / 20
        spr = self._sol_u8.get(nivel)
        if spr is None:
            spr = np.clip(self.sol * (255.0 * nivel), 0, 255).astype(np.uint8)
            self._sol_u8[nivel] = spr
        S = spr.shape[0]
        x0, y0 = int(round(x)) - S // 2, int(round(y)) - S // 2
        xa, xb = max(x0, 0), min(x0 + S, self.w)
        ya, yb = max(y0, 0), min(y0 + S, self.h)
        if xb <= xa or yb <= ya:
            return
        buf[ya:yb, xa:xb] = cv2.add(buf[ya:yb, xa:xb], spr[ya - y0:yb - y0, xa - x0:xb - x0])

    def dibujar_luna(self, buf, x, y, radio_px, sol_cam, cam, luna_tierra):
        """Luna como esfera iluminada por el Sol (fase real), con la cara visible hacia la Tierra.
        sol_cam: direccion del Sol en el marco de la camara. luna_tierra: vector Tierra->Luna."""
        if radio_px < 0.6:
            # demasiado lejos: un punto con el brillo de su fase
            hacia_cam = -cam.a_camara(luna_tierra - cam.pos)
            hacia_cam /= np.linalg.norm(hacia_cam) + 1e-12
            fase = 0.5 * (1 + float(np.dot(hacia_cam, sol_cam)))
            v = int(200 * fase)
            if 0 <= int(x) < self.w and 0 <= int(y) < self.h and v > 0:
                cv2.circle(buf, (int(round(x * 16)), int(round(y * 16))), 16, (v, v, v), -1, cv2.LINE_AA, 4)
            return
        rad = int(math.ceil(radio_px)) + 2
        x0, y0 = int(math.floor(x)) - rad, int(math.floor(y)) - rad
        xa, xb = max(x0, 0), min(x0 + 2 * rad + 1, self.w)
        ya, yb = max(y0, 0), min(y0 + 2 * rad + 1, self.h)
        if xb <= xa or yb <= ya:
            return
        gx, gy = np.meshgrid(np.arange(xa, xb) + 0.5 - x, np.arange(ya, yb) + 0.5 - y)
        u, v = gx / radio_px, gy / radio_px
        r2 = u * u + v * v
        a = np.clip((1.0 - np.sqrt(r2)) * radio_px + 0.5, 0.0, 1.0)
        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        n = np.stack([u, v, z], -1)                            # normal en el marco de la camara
        mu0 = np.clip(n @ sol_cam, 0.0, 1.0)
        mu = np.maximum(z, 1e-3)
        luz = 2.0 * mu0 / (mu0 + mu)                           # Lommel-Seeliger: la Luna casi no se oscurece al borde

        # orientacion de la Luna: su cara visible mira a la Tierra (rotacion sincronica)
        hacia_tierra = -luna_tierra / np.linalg.norm(luna_tierra)
        norte = np.array([0.0, 0.0, 1.0]) - hacia_tierra * hacia_tierra[2]
        norte /= np.linalg.norm(norte)
        este = np.cross(norte, hacia_tierra)
        nt = n @ cam.M                                          # camara -> marco de la Tierra
        lx, ly, lz = nt @ hacia_tierra, nt @ este, nt @ norte
        Ht, Wt = self.luna_tex.shape
        mx = ((np.arctan2(ly, lx) / (2 * math.pi) + 0.5) * Wt - 0.5).astype(np.float32)
        my = ((0.5 - np.arcsin(np.clip(lz, -1, 1)) / math.pi) * Ht - 0.5).astype(np.float32)
        albedo = cv2.remap(self.luna_tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        brillo = (albedo * (luz * C.BRILLO_LUNA + C.LUZ_TIERRA_EN_LUNA)) * a
        col = brillo[..., None] * np.array([0.93, 0.97, 1.0]) * 255.0
        roi = buf[ya:yb, xa:xb].astype(np.float32)
        buf[ya:yb, xa:xb] = np.clip(roi * (1.0 - a[..., None]) + col, 0, 255).astype(np.uint8)
