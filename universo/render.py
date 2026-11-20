"""Todo lo que se dibuja: el cielo real (estrellas, Via Lactea, Sol, Luna), la Tierra
vista desde la camara en orbita, guias de las manos, barra de zoom, textos (con
acentos, via Pillow) y la camara en miniatura."""

import math
from collections import OrderedDict, deque

import cv2
import numpy as np

from datetime import datetime, timezone

from universo import astros as A
from universo import config as C
from universo.cielo import Cielo
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
def fecha_actual():
    """La fecha del cielo: config.FECHA si esta fijada, si no, ahora mismo (UTC)."""
    if C.FECHA:
        f = datetime.fromisoformat(C.FECHA)
        return f if f.tzinfo else f.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


NOMBRES = {"tierra": "Tierra", "luna": "Luna"}


def _km(km):
    if km >= 1e6:
        return f"{km / 1e6:,.1f} millones de km".replace(",", " ")
    return f"{km:,.0f} km".replace(",", " ")


class Escena:
    def __init__(self, ancho=C.ANCHO, alto=C.ALTO, textura=C.TEXTURA):
        self.w, self.h = ancho, alto
        self.focal = (alto / 2.0) / math.tan(math.radians(C.CAMPO_VISION_V) / 2.0)
        self.cielo = Cielo(ancho, alto, self.focal)
        self.tierra = Tierra(textura, ruta_noche=C.TEXTURA_NOCHE)
        self.luna = Tierra(C.TEXTURA_LUNA, R=300, atmosfera=False)
        self.texto = Texto()
        self.buf = np.zeros((alto, ancho, 3), np.uint8)
        self.radio_base = C.RADIO_BASE_REL * alto
        self._rastros = {}
        self.astros = {}

    # ------------------------------------------------------------------
    def componer(self, log_escala, giro, estado, info, frame=None, ver_camara=True, ver_info=True, t=0.0,
                 inclinacion=None, fecha=None, foco="tierra", viaje=None, seleccion=None):
        if inclinacion is None:
            inclinacion = estado.inclinacion
        self._espacio(log_escala, giro, inclinacion, fecha or fecha_actual(), foco, viaje, t)
        if seleccion is not None:
            self._seleccion(seleccion)
        if estado.activo:
            self._barra_zoom(log_escala, estado.zoom_agarrado)
        self._manos(estado)
        if ver_camara and frame is not None:
            self._camara(frame, estado)
        self._hud(log_escala, giro, inclinacion, estado, info, ver_info)
        self._avisos(estado, t)
        return self.buf

    # ------------------------------------------------------------------
    def _cuerpos(self, jd):
        """Astros que se dibujan como esfera: nombre -> (centro, radio en radios terrestres,
        dibujante, rotacion marco Tierra -> marco del astro)."""
        luna = A.luna(jd)
        alfa = math.atan2(-luna[1], -luna[0])             # la cara visible de la Luna mira a la Tierra
        return {
            "tierra": (np.zeros(3), 1.0, self.tierra, np.eye(3)),
            "luna": (luna, A.RADIO_LUNA_KM / A.RADIO_TIERRA_KM, self.luna, A.rot_z(-alfa)),
        }

    def camara(self, log_escala, giro, inclinacion, foco, jd):
        """Camara en orbita del astro en foco: el zoom fija su tamano en pantalla.
        giro/inclinacion se miden en el marco del astro (su longitud 0 al centro)."""
        centro, R, _, rot = self._cuerpos(jd)[foco]
        radio = self.radio_base * math.exp(log_escala)
        d = R * A.distancia_para_radio(radio, self.focal)
        atras_astro = A.punto_superficie(-math.degrees(inclinacion), -math.degrees(giro))
        g, i = A.giro_inclinacion(rot.T @ atras_astro)
        return A.Camara(g, i, d, self.focal, self.w / 2.0, self.h / 2.0, centro)

    def _espacio(self, log_escala, giro, inclinacion, fecha, foco="tierra", viaje=None, t=0.0):
        """Cielo, Sol y astros vistos desde la camara, en orden de profundidad."""
        b = self.buf
        jd = A.dia_juliano(fecha)
        cuerpos = self._cuerpos(jd)
        hiper = 0.0
        if viaje is not None:
            pos, objetivo, hiper = viaje.camara(t)
            cam = A.Camara.mirando(pos, objetivo, self.focal, self.w / 2.0, self.h / 2.0)
        else:
            cam = self.camara(log_escala, giro, inclinacion, foco, jd)
        sol_dir, sol_km = A.sol(jd)
        sol_cam = cam.a_camara(sol_dir)

        self.cielo.fondo(b, cam, A.ecuatorial_a_tierra(jd))

        # proyectar los astros; el Sol queda tapado tras cualquier disco
        vistos, vis_sol = [], 1.0
        ang_sol = math.radians(C.SOL_ANGULO_RADIO_GRADOS)
        for nombre, (centro, R, dib, rot) in cuerpos.items():
            x, y, z = cam.proyectar_punto(centro)
            dist = float(np.linalg.norm(cam.pos - centro))
            if z <= 0 or dist <= R:
                continue
            r = A.radio_aparente_px(R, dist, self.focal)
            vistos.append((z, nombre, x, y, r))
            hacia = (centro - cam.pos) / dist
            sep = math.acos(float(np.clip(hacia @ sol_dir, -1, 1)))
            ang = math.asin(min(R / dist, 1.0))
            vis_sol = min(vis_sol, float(np.clip((sep - ang + ang_sol) / (2 * ang_sol), 0.0, 1.0)))
        sx, sy, delante = cam.proyectar_direccion(sol_dir)
        if delante:
            self.cielo.dibujar_sol(b, float(sx), float(sy), vis_sol)

        objetivos = {}
        for z, nombre, x, y, r in sorted(vistos, reverse=True):        # de lejos a cerca
            centro, R, dib, rot = cuerpos[nombre]
            if r < 1.2:                                                  # muy lejos: un punto
                if 0 <= x < self.w and 0 <= y < self.h:
                    col = (230, 170, 90) if nombre == "tierra" else (200, 200, 200)
                    cv2.circle(b, (int(round(x * 16)), int(round(y * 16))), 16, col, -1, cv2.LINE_AA, 4)
            else:
                g, i = A.giro_inclinacion(rot @ (cam.pos - centro))
                dib.dibujar(b, x, y, r, g, i, sol_cam, rot @ sol_dir)
            if -r < x < self.w + r and -r < y < self.h + r:
                objetivos[nombre] = (x, y, r)
        for z, nombre, x, y, r in vistos:                                # tapado por otro: no se elige
            for z2, n2, x2, y2, r2 in vistos:
                if n2 != nombre and z2 < z and math.hypot(x - x2, y - y2) + r < r2:
                    objetivos.pop(nombre, None)

        luna = cuerpos["luna"][0]
        self.astros = {
            "camara": cam,
            "foco": foco,
            "altitud_km": (float(np.linalg.norm(cam.pos - cuerpos[foco][0])) - cuerpos[foco][1]) * A.RADIO_TIERRA_KM,
            "luna_km": float(np.linalg.norm(luna)) * A.RADIO_TIERRA_KM,
            "sol_km": sol_km,
            "sol_px": (float(sx), float(sy)) if delante and vis_sol > 0.05 else None,
            "objetivos": objetivos,
            "hiper": hiper,
        }
        if C.ETIQUETAS_ASTROS and hiper < 0.05:
            self._etiquetas()

    def _etiquetas(self):
        a, T = self.astros, self.texto
        dentro = lambda x, y: 0 <= x < self.w - 60 and 60 <= y < self.h - 60
        if a["sol_px"] and dentro(*a["sol_px"]):
            x, y = a["sol_px"]
            T.dibujar(self.buf, "Sol", x + 16, y + 10, 15, COL_TEXTO, True)
            T.dibujar(self.buf, _km(a["sol_km"]), x + 16, y + 29, 12, COL_TENUE)
        for nombre, (x, y, r) in a["objetivos"].items():
            if nombre == a["foco"] or not dentro(x, y):
                continue
            T.dibujar(self.buf, NOMBRES[nombre], x + r + 10, y - 9, 15, COL_TEXTO, True)
            T.dibujar(self.buf, _km(a["luna_km"]), x + r + 10, y + 10, 12, COL_TENUE)

    def _seleccion(self, sel):
        """Anillo blanco semitransparente que se llena al dejar la palma sobre un astro."""
        x, y, p = sel
        if x is None or p <= 0:
            return
        capa = self.buf.copy()
        cv2.circle(capa, _p(x, y), 34 * 16, (255, 255, 255), 2, cv2.LINE_AA, 4)
        cv2.ellipse(capa, _p(x, y), (34 * 16, 34 * 16), -90, 0, int(360 * p), (255, 255, 255), 6, cv2.LINE_AA, 4)
        cv2.addWeighted(capa, 0.55, self.buf, 0.45, 0, dst=self.buf)

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
        alt = self.astros.get("altitud_km", 0.0)
        T.dibujar(b, f"Altitud {_km(alt)}   ·   Sobre {abs(lat):.0f}° {hemi}, {abs(lon):.0f}° {lado}",
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
