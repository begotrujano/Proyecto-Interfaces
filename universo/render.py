"""Todo lo que se dibuja encima: estrellas, Tierra, guias de las manos, barra de zoom,
textos (con acentos, via Pillow) y la camara en miniatura."""

import math
from collections import OrderedDict, deque

import cv2
import numpy as np

from universo import config as C
from universo.tierra import Tierra

COL_IZQ = (40, 178, 255)        # BGR ambar   -> zoom
COL_DER = (255, 208, 84)        # BGR cian    -> giro
COL_GRIS = (175, 175, 175)
COL_OK = (110, 230, 120)
COL_TEXTO = (235, 235, 235)
COL_TENUE = (150, 150, 150)
CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ----------------------------------------------------------------------
# Estrellas (se generan una vez)
# ----------------------------------------------------------------------
def generar_estrellas(w, h, semilla=11):
    rng = np.random.default_rng(semilla)
    n = int(w * h / 1400)
    img = np.zeros((h, w, 3), np.float32)
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    brillo = 0.10 + 0.90 * rng.random(n) ** 5
    tipo = rng.random(n)
    colores = np.where(tipo[:, None] < 0.25, np.array([0.62, 0.82, 1.0]),
                       np.where(tipo[:, None] < 0.75, np.array([1.0, 1.0, 1.0]),
                                np.array([1.0, 0.88, 0.76])))

    def mancha(cx, cy, sigma, amp, color):
        rad = int(math.ceil(3 * sigma)) + 1
        x0, y0 = int(cx) - rad, int(cy) - rad
        gx = np.arange(x0, x0 + 2 * rad + 1)
        gy = np.arange(y0, y0 + 2 * rad + 1)
        g = np.exp(-((gx[None, :] + 0.5 - cx) ** 2 + (gy[:, None] + 0.5 - cy) ** 2) / (2 * sigma ** 2)) * amp
        xa, xb = max(x0, 0), min(x0 + 2 * rad + 1, w)
        ya, yb = max(y0, 0), min(y0 + 2 * rad + 1, h)
        if xb > xa and yb > ya:
            img[ya:yb, xa:xb] += g[ya - y0:yb - y0, xa - x0:xb - x0, None] * color

    for i in range(n):
        mancha(xs[i], ys[i], 0.55 + 0.9 * brillo[i], brillo[i], colores[i])
        if brillo[i] > 0.55:
            mancha(xs[i], ys[i], 3.2, 0.10 * brillo[i], colores[i])
    # un velo muy tenue de "via lactea" para que el fondo no sea plano
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    banda = np.exp(-((yy - (0.35 * h + 0.25 * xx)) / (0.22 * h)) ** 2)
    ruido = cv2.GaussianBlur(rng.random((h // 8, w // 8)).astype(np.float32), (0, 0), 2.0)
    ruido = cv2.resize(ruido, (w, h), interpolation=cv2.INTER_CUBIC)
    img += (banda * (0.5 + ruido) * 0.035)[..., None] * np.array([1.0, 0.85, 0.80])
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------
# Texto con Pillow (acentos, buena tipografia) y cache de "sellos"
# ----------------------------------------------------------------------
class Texto:
    FUENTES = {
        False: ["segoeui.ttf", "C:/Windows/Fonts/segoeui.ttf", "arial.ttf",
                "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        True: ["seguisb.ttf", "segoeuib.ttf", "C:/Windows/Fonts/segoeuib.ttf", "arialbd.ttf",
               "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    }

    def __init__(self):
        from PIL import Image, ImageDraw, ImageFont
        self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
        self._fuentes = {}
        self._cache = OrderedDict()

    def _fuente(self, tam, negrita):
        clave = (tam, negrita)
        if clave not in self._fuentes:
            f = None
            for nombre in self.FUENTES[negrita]:
                try:
                    f = self.ImageFont.truetype(nombre, tam)
                    break
                except OSError:
                    continue
            if f is None:
                f = self.ImageFont.load_default(tam)
            self._fuentes[clave] = f
        return self._fuentes[clave]

    def sello(self, texto, tam, color, negrita):
        clave = (texto, tam, color, negrita)
        if clave in self._cache:
            self._cache.move_to_end(clave)
            return self._cache[clave]
        f = self._fuente(tam, negrita)
        borde = max(1, tam // 9)
        l, t, r, b = f.getbbox(texto, stroke_width=borde)
        w, h = max(r - l + 2, 1), max(b - t + 2, 1)
        m_txt = self.Image.new("L", (w, h), 0)
        m_bor = self.Image.new("L", (w, h), 0)
        self.ImageDraw.Draw(m_bor).text((1 - l, 1 - t), texto, font=f, fill=255,
                                        stroke_width=borde, stroke_fill=255)
        self.ImageDraw.Draw(m_txt).text((1 - l, 1 - t), texto, font=f, fill=255)
        a_txt = np.asarray(m_txt, np.float32) / 255.0
        a_bor = np.asarray(m_bor, np.float32) / 255.0
        alfa = np.maximum(a_txt, a_bor * 0.7)
        pre = a_txt[..., None] * np.array(color, np.float32)
        s = ((pre + 0.5).astype(np.uint8), (alfa * 255 + 0.5).astype(np.uint8), -l + 1, -t + 1)
        self._cache[clave] = s
        if len(self._cache) > 400:
            self._cache.popitem(last=False)
        return s

    def dibujar(self, buf, texto, x, y, tam=18, color=COL_TEXTO, negrita=False, ancla="i", opacidad=1.0):
        """ancla: 'i' izquierda, 'c' centro, 'd' derecha. y = linea superior del texto."""
        if not texto or opacidad <= 0.01:
            return 0
        pre, alfa, _, _ = self.sello(texto, tam, tuple(int(c) for c in color), negrita)
        h, w = alfa.shape
        x = int(round(x - (w / 2 if ancla == "c" else w if ancla == "d" else 0)))
        y = int(round(y))
        H, W = buf.shape[:2]
        xa, xb, ya, yb = max(x, 0), min(x + w, W), max(y, 0), min(y + h, H)
        if xb <= xa or yb <= ya:
            return w
        p = pre[ya - y:yb - y, xa - x:xb - x].astype(np.float32)
        a = alfa[ya - y:yb - y, xa - x:xb - x, None].astype(np.float32) / 255.0
        if opacidad < 1.0:
            p, a = p * opacidad, a * opacidad
        roi = buf[ya:yb, xa:xb]
        roi[:] = np.clip(roi * (1.0 - a) + p, 0, 255).astype(np.uint8)
        return w


def _p(x, y):
    return (int(round(x * 16)), int(round(y * 16)))


def _flecha(buf, cx, cy, ang, col, tam=7):
    pts = []
    for a, r in ((0.0, tam), (2.5, tam * 0.9), (-2.5, tam * 0.9)):
        pts.append(_p(cx + r * math.cos(ang + a), cy + r * math.sin(ang + a)))
    cv2.fillPoly(buf, [np.array(pts, np.int32)], col, cv2.LINE_AA, 4)


# ----------------------------------------------------------------------
class Escena:
    def __init__(self, ancho=C.ANCHO, alto=C.ALTO, textura=C.TEXTURA):
        self.w, self.h = ancho, alto
        self.estrellas = generar_estrellas(ancho, alto)
        self.tierra = Tierra(textura)
        self.texto = Texto()
        self.buf = np.empty_like(self.estrellas)
        self.radio_base = C.RADIO_BASE_REL * alto
        self._rastros = {}

    # ------------------------------------------------------------------
    def componer(self, log_escala, giro, estado, info, frame=None, ver_camara=True, ver_info=True, t=0.0,
                 inclinacion=None):
        if inclinacion is None:
            inclinacion = estado.inclinacion
        np.copyto(self.buf, self.estrellas)
        self.tierra.dibujar(self.buf, self.w / 2.0, self.h / 2.0,
                            self.radio_base * math.exp(log_escala), giro, inclinacion)
        if estado.activo:
            self._barra_zoom(log_escala, estado.zoom_agarrado)
        self._manos(estado)
        if ver_camara and frame is not None:
            self._camara(frame, estado)
        self._hud(log_escala, giro, inclinacion, estado, info, ver_info)
        self._avisos(estado, t)
        return self.buf

    # ------------------------------------------------------------------
    def _manos(self, estado):
        b = self.buf
        vivos = set()
        for v in estado.manos:
            if not v.visible:
                continue
            vivos.add(v.id)
            col = (COL_IZQ if v.rol == "izq" else COL_DER) if estado.activo else COL_GRIS
            x, y = v.x, v.y

            ras = self._rastros.setdefault(v.id, deque(maxlen=10))
            if v.accion in ("swipe", "swipe_v"):
                ras.append((x, y))
            elif ras:
                ras.popleft()
            pts = list(ras)
            for i in range(1, len(pts)):
                k = i / len(pts)
                cv2.line(b, _p(*pts[i - 1]), _p(*pts[i]), tuple(int(c * k) for c in col),
                         max(1, int(1 + 7 * k)), cv2.LINE_AA, 4)

            if v.accion in ("zoom", "agarre"):
                cv2.circle(b, _p(x, y), 10 * 16, col, -1, cv2.LINE_AA, 4)
                cv2.circle(b, _p(x, y), 22 * 16, col, 2, cv2.LINE_AA, 4)
                angs = ((-math.pi / 2, math.pi / 2) if v.accion == "zoom"
                        else (math.pi, 0.0, -math.pi / 2, math.pi / 2))    # agarre: en 4 direcciones
                for a in angs:
                    _flecha(b, x + 34 * math.cos(a), y + 34 * math.sin(a), a, col)
            elif v.accion in ("swipe", "swipe_v"):
                cv2.circle(b, _p(x, y), 18 * 16, col, 3, cv2.LINE_AA, 4)
                cv2.circle(b, _p(x, y), 4 * 16, col, -1, cv2.LINE_AA, 4)
            else:
                cv2.circle(b, _p(x, y), 16 * 16, col, 2, cv2.LINE_AA, 4)
                cv2.circle(b, _p(x, y), 3 * 16, col, -1, cv2.LINE_AA, 4)

            if v.progreso_ok > 0:
                fin = int(360 * v.progreso_ok)
                cv2.ellipse(b, _p(x, y), (30 * 16, 30 * 16), -90, 0, fin, COL_OK, 4, cv2.LINE_AA, 4)
                self.texto.dibujar(b, "OK", x, y - 58, 16, COL_OK, True, "c")

            if estado.activo:
                etiqueta = "ZOOM" if v.rol == "izq" else "GIRO"
                self.texto.dibujar(b, etiqueta, x, y + 36, 14, col, True, "c")
        for hid in list(self._rastros):
            if hid not in vivos:
                del self._rastros[hid]

    def _barra_zoom(self, log_escala, agarrado):
        b = self.buf
        x, y0, y1 = 48, int(self.h * 0.28), int(self.h * 0.72)
        lo, hi = math.log(C.ESCALA_MIN), math.log(C.ESCALA_MAX)
        col = COL_IZQ if agarrado else tuple(int(c * 0.55) for c in COL_IZQ)
        cv2.line(b, (x, y0), (x, y1), col, 2, cv2.LINE_AA)
        for marca in (0.1, 1.0, 5.0):
            if C.ESCALA_MIN <= marca <= C.ESCALA_MAX:
                ym = y1 - (math.log(marca) - lo) / (hi - lo) * (y1 - y0)
                cv2.line(b, (x - 6, int(ym)), (x + 6, int(ym)), col, 1, cv2.LINE_AA)
                self.texto.dibujar(b, f"×{marca:g}", x + 12, ym - 9, 13, COL_TENUE)
        yv = y1 - (log_escala - lo) / (hi - lo) * (y1 - y0)
        cv2.circle(b, _p(x, yv), (9 if agarrado else 7) * 16, COL_IZQ, -1, cv2.LINE_AA, 4)
        self.texto.dibujar(b, "ZOOM", x, y0 - 30, 14, col, True, "c")

    def _camara(self, frame, estado, ancho=240):
        alto = int(ancho * frame.shape[0] / frame.shape[1])
        pip = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)
        pip = cv2.convertScaleAbs(pip, alpha=0.55)
        for v in estado.manos:
            if not v.visible:
                continue
            col = (COL_IZQ if v.rol == "izq" else COL_DER) if estado.activo else COL_GRIS
            pts = [(int(p[0] * ancho), int(p[1] * alto)) for p in v.norm]
            for i, j in CONEXIONES:
                cv2.line(pip, pts[i], pts[j], col, 1, cv2.LINE_AA)
        m = C.MARGEN_CAMARA
        cv2.rectangle(pip, (int(m * ancho), int(m * alto)), (int((1 - m) * ancho), int((1 - m) * alto)),
                      (70, 70, 70), 1, cv2.LINE_AA)
        x0, y0 = self.w - ancho - 16, 46                    # arriba a la derecha, bajo los atajos
        self.buf[y0:y0 + alto, x0:x0 + ancho] = pip
        cv2.rectangle(self.buf, (x0, y0), (x0 + ancho, y0 + alto), (90, 90, 90), 1, cv2.LINE_AA)

    def _hud(self, log_escala, giro, inclinacion, estado, info, ver_info):
        b, T = self.buf, self.texto
        if estado.activo:
            cv2.circle(b, _p(26, 31), 6 * 16, COL_OK, -1, cv2.LINE_AA, 4)
            T.dibujar(b, "CONTROL ACTIVO", 40, 19, 18, COL_OK, True)
        else:
            cv2.circle(b, _p(26, 31), 6 * 16, COL_GRIS, -1, cv2.LINE_AA, 4)
            T.dibujar(b, "EN PAUSA", 40, 19, 18, COL_GRIS, True)
        lon = -math.degrees(giro)
        lon = (lon + 180.0) % 360.0 - 180.0
        lado = "O" if lon < 0 else "E"
        lat = -math.degrees(inclinacion)                   # inclinar hacia arriba = se ve el sur
        hemi = "S" if lat < -0.5 else "N"
        T.dibujar(b, f"Escala ×{math.exp(log_escala):.2f}   ·   Centro {abs(lat):.0f}° {hemi}, {abs(lon):.0f}° {lado}",
                  20, 48, 16, COL_TEXTO)

        if ver_info:
            exp = info.get("exposicion")
            T.dibujar(b, f"dibujo {info['fps_render']:.0f} fps  ·  cámara {info.get('fps_cam', 0):.0f} fps "
                         f"(exp. {'auto' if exp is None else f'{exp:+.0f}'})  ·  detección {info['fps_det']:.0f} fps "
                         f"({info['lat_ms']:.0f} ms)  ·  giro {estado.omega:+.2f}  incl. {estado.omega_v:+.2f} rad/s  ·  swipe: {estado.swipe}",
                      20, 74, 13, COL_TENUE)
            y = 94
            for v in sorted(estado.manos, key=lambda v: v.rol):
                if not v.visible:
                    continue
                nombre = "Izquierda" if v.rol == "izq" else "Derecha"
                pose = "PUÑO" if v.puno else ("OK" if v.ok else "abierta")
                T.dibujar(b, f"{nombre}: {pose:7s}  puño {v.ratio_puno:.2f} (<{C.PUNO_ON})   "
                             f"ok {v.ratio_ok:.2f} (<{C.OK_ON})   dedos {v.estirados}/3",
                          20, y, 13, COL_IZQ if v.rol == "izq" else COL_DER)
                y += 19
            T.dibujar(b, "Espacio activar/pausar · R reiniciar · I intercambiar manos · V cámara · "
                         "[ ] exposición · H info · F pantalla completa · Q salir", self.w - 16, 22, 13, COL_TENUE, ancla="d")

        if estado.activo:
            msg = ("Izquierda: puño y sube o baja = zoom     ·     Derecha: swipe a los lados = girar, "
                   "arriba/abajo = inclinar; puño = agarrar     ·     OK = pausa")
        else:
            msg = "Haz el signo de OK con cualquier mano para activar el control"
        T.dibujar(b, msg, self.w / 2, self.h - 40, 17 if not estado.activo else 15, COL_TEXTO, ancla="c")

    def _avisos(self, estado, t):
        for i, (texto, t0) in enumerate(estado.avisos[-1:]):
            edad = t - t0
            op = 1.0 if edad < 1.2 else max(0.0, 1.0 - (edad - 1.2) / 0.8)
            col = COL_OK if "activ" in texto.lower() and "pausa" not in texto.lower() else COL_TEXTO
            self.texto.dibujar(self.buf, texto, self.w / 2, self.h - 104, 30, col, True, "c", op)
