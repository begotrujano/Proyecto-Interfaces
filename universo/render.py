"""Todo lo que se dibuja: el cielo real (estrellas, Via Lactea, Sol, Luna), la Tierra
vista desde la camara en orbita, guias de las manos, barra de zoom, textos (con
acentos, via Pillow) y la camara en miniatura."""

import math
import threading
from collections import OrderedDict, deque

import cv2
import numpy as np

from datetime import datetime, timezone

from universo import astros as A
from universo import sistema as S
from universo import config as C
from universo import etiquetas as ET
from universo.fichas import FICHAS
from universo.reloj import fecha_inicial, texto_velocidad
from universo.carga import AstroBajoDemanda
from universo.cielo import Cielo
from universo.planetas import textura_marte, textura_mercurio, textura_sol, textura_venus
from universo.tierra import Tierra, niveles_esfera
from universo.vista_sistema import VistaSistema

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
    """La fecha del cielo si no se pasa otra: config.FECHA si esta fijada, si no, ahora mismo (UTC).
    (La aplicacion usa un universo.reloj.Reloj, que puede ir mas rapido o hacia atras.)"""
    return fecha_inicial()


_MESES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def texto_fecha(f):
    """'22 nov 2026 · 17:05' en la hora local de la computadora."""
    f = f.astimezone()
    return f"{f.day} {_MESES[f.month - 1]} {f.year} · {f.hour:02d}:{f.minute:02d}"


NOMBRES = S.NOMBRES


def _km(km):
    if km >= 1e6:
        return f"{km / 1e6:,.1f} millones de km".replace(",", " ")
    return f"{km:,.0f} km".replace(",", " ")


def _ua(ua):
    return f"{ua:.2f} UA" if ua >= 0.1 else f"{ua:.3f} UA"


class Escena:
    def __init__(self, ancho=C.ANCHO, alto=C.ALTO, textura=C.TEXTURA):
        self.w, self.h = ancho, alto
        self.focal = (alto / 2.0) / math.tan(math.radians(C.CAMPO_VISION_V) / 2.0)
        self.cielo = Cielo(ancho, alto, self.focal)
        self.tierra = Tierra(textura, ruta_noche=C.TEXTURA_NOCHE)
        self.luna = Tierra(C.TEXTURA_LUNA, R=300, atmosfera=False)
        # planetas: se cargan bajo demanda (universo/carga.py); el atributo se llama como el planeta
        self._foco_actual, self._destino = "tierra", None
        bd = lambda nombre, ruta, generar, ancho, **kw: AstroBajoDemanda(
            nombre, ruta, generar, S.COLORES[nombre], ancho, al_pedir=self._al_pedir, **kw)
        self.mercurio = bd("mercurio", C.TEXTURA_MERCURIO, textura_mercurio, 2048, atmosfera=False, oceanos=False,
                           desfase_lon=C.DESFASE_TEXTURA_MERCURIO)
        self.venus = bd("venus", C.TEXTURA_VENUS, textura_venus, 1024, oceanos=False,     # atmosfera densa: halo
                        color_atm=C.COLOR_ATM_VENUS, desfase_lon=C.DESFASE_TEXTURA_VENUS)
        self.marte = bd("marte", C.TEXTURA_MARTE, textura_marte, 2048, atmosfera=False, oceanos=False,
                        desfase_lon=C.DESFASE_TEXTURA_MARTE)
        self.sol = Tierra(C.TEXTURA_SOL, emisivo=True, color_atm=C.COLOR_HALO_SOL,
                          desfase_lon=C.DESFASE_TEXTURA_SOL, respaldo=lambda: textura_sol(1024))
        self.texto = Texto()
        self.buf = np.zeros((alto, ancho, 3), np.uint8)
        self.radio_base = C.RADIO_BASE_REL * alto
        self._rastros = {}
        self.astros = {}
        self.vista_sistema = VistaSistema(self)
        self._modo, self._fundido = "astro", None
        self.resaltado = None                              # (x, y, radio, nombre) del astro bajo el cursor del raton
        self._ocupado = []                                 # zonas donde no van etiquetas (la ficha)
        self._fichas_cache = {}
        self._hilo_pre = threading.Thread(target=self._precalentar, daemon=True, name="precalentar")
        self._hilo_pre.start()

    def esperar_iconos(self):
        """Espera a que esten las versiones chicas de todos los planetas (lo usan las pruebas)."""
        self._hilo_pre.join()
        for a in self._bajo_demanda():
            a.preparar_chico(esperar=True)

    def _bajo_demanda(self):
        return [a for a in vars(self).values() if isinstance(a, AstroBajoDemanda)]

    def preparar(self, nombre, esperar=False):
        """Empieza a cargar la textura grande de un astro (al elegirlo como destino de un viaje)."""
        a = getattr(self, nombre, None)
        if isinstance(a, AstroBajoDemanda):
            self._destino = nombre
            a.preparar(esperar)

    def _al_pedir(self, astro):
        """Se pidio la version grande de un astro: si ya hay demasiadas, suelta las menos usadas
        (nunca la del astro en foco ni la del destino del viaje)."""
        protegidos = {astro.nombre, self._foco_actual, self._destino}
        otros = [a for a in self._bajo_demanda() if a.pedido and a is not astro]
        exceso = len(otros) + 1 - max(C.PLANETAS_EN_MEMORIA, 1)
        libres = sorted((a for a in otros if a.nombre not in protegidos), key=lambda a: a.uso)
        for a in libres[:max(exceso, 0)]:
            a.liberar()

    def _precalentar(self):
        """Hornea la luz del Sol y prepara las versiones chicas de los planetas (las grandes
        se cargan cuando se necesitan: ver universo/carga.py)."""
        try:
            niveles_esfera(600, False, True)               # tablas de las esferas sin atmosfera (Mercurio, Marte)
            jd = A.dia_juliano(fecha_actual())
            cuerpos = self._cuerpos(jd)
            for nombre in [n for n in C.PLANETAS_ACTIVOS if n != "tierra" and n in cuerpos] + ["sol"]:
                centro, _, dib, rot = cuerpos[nombre]
                d = cuerpos["sol"][0] - centro
                dib.iluminar(rot @ (d / np.linalg.norm(d)) if nombre != "sol" else np.array([1.0, 0.0, 0.0]))
        except Exception:                                  # solo es una optimizacion
            pass

    # ------------------------------------------------------------------
    def componer(self, log_escala, giro, estado, info, frame=None, ver_camara=True, ver_info=True, t=0.0,
                 inclinacion=None, fecha=None, foco="tierra", viaje=None, seleccion=None, modo="astro"):
        if inclinacion is None:
            inclinacion = estado.inclinacion
        self._foco_actual = foco                          # para no liberar su textura (carga bajo demanda)
        ficha = info.get("ficha")                          # (nombre, opacidad): ficha educativa del astro
        ficha_visible = ficha is not None and modo == "astro" and viaje is None and ficha[1] > 0.01
        self._ocupado = [self._ficha_diseno(ficha[0])[0]] if ficha_visible and ficha[0] in FICHAS else []
        self._destino = viaje.nombre if viaje is not None else None
        if modo != self._modo:                             # fundido entre la vista de un astro y la del sistema
            self._fundido = (self.buf.copy(), t)
            self._modo = modo
        if modo == "sistema":
            self._sistema(log_escala, giro, inclinacion, fecha or fecha_actual(), foco)
        else:
            self._espacio(log_escala, giro, inclinacion, fecha or fecha_actual(), foco, viaje, t)
        if seleccion is not None:
            self._seleccion(seleccion)
        if self.resaltado is not None and modo == self._modo:
            x, y, r, nombre = self.resaltado
            cv2.circle(self.buf, _p(x, y), int((max(r, 12.0) + 8) * 16), (255, 255, 255), 1, cv2.LINE_AA, 4)
            accion = "volver" if (modo == "sistema" and nombre == foco) else "ir a"
            self.texto.dibujar(self.buf, f"clic: {accion} {NOMBRES.get(nombre, nombre)}" if accion == "ir a"
                               else f"clic: volver a {NOMBRES.get(nombre, nombre)}",
                               x, y + max(r, 12.0) + 16, 13, COL_TEXTO, False, "c")
        modos = info.get("modos")
        if modos is not None:
            corte, gravedad, comparador, quiz = modos
            corte.dibujar(self, foco)
            gravedad.dibujar(self, foco)
            comparador.dibujar(self, A.dia_juliano(fecha or fecha_actual()), t)
            quiz.dibujar(self, t)
        if estado.activo:
            self._barra_zoom(log_escala, estado.zoom_agarrado)
        menu = info.get("menu")
        if menu is not None:
            menu.dibujar(self)
        self._manos(estado)
        if ver_camara and frame is not None:
            self._camara(frame, estado)
        if ficha_visible and not (modos and modos[2].activo):
            self._ficha(*ficha)
        self._hud(log_escala, giro, inclinacion, estado, info, ver_info, modo)
        self._avisos(estado, t)
        if self._fundido is not None:
            a = 1.0 - (t - self._fundido[1]) / max(C.SISTEMA_TRANSICION, 1e-3)
            if a > 0.0 and self._fundido[0].shape == self.buf.shape:
                cv2.addWeighted(self._fundido[0], a, self.buf, 1.0 - a, 0.0, dst=self.buf)
            else:
                self._fundido = None
        return self.buf

    # ------------------------------------------------------------------
    def _cuerpos(self, jd):
        """Astros que se dibujan como esfera: nombre -> (centro, radio en radios terrestres,
        dibujante, rotacion marco Tierra -> marco del astro). El centro va en el marco de la Tierra."""
        luna = A.luna(jd)
        alfa = math.atan2(-luna[1], -luna[0])             # la cara visible de la Luna mira a la Tierra
        cuerpos = {
            "sol": (S.posicion("sol", jd), S.radio_rt("sol"), self.sol, S.rotacion("sol", jd)),
            "tierra": (np.zeros(3), 1.0, self.tierra, np.eye(3)),
            "luna": (luna, S.radio_rt("luna"), self.luna, A.rot_z(-alfa)),
        }
        for nombre in C.PLANETAS_ACTIVOS:                  # los demas planetas: ver config.PLANETAS_ACTIVOS
            if nombre in S.FISICOS:
                capa = "nubes" if (nombre == "venus" and C.VENUS_NUBES_SUPERROTAN) else None
                cuerpos[nombre] = (S.posicion(nombre, jd), S.radio_rt(nombre), getattr(self, nombre),
                                   S.rotacion(nombre, jd, capa))
        return cuerpos

    def camara(self, log_escala, giro, inclinacion, foco, jd, cuerpos=None):
        """Camara en orbita del astro en foco: el zoom fija su tamano en pantalla.
        giro/inclinacion se miden en el marco del astro (su longitud 0 al centro, su polo norte arriba)."""
        centro, R, _, rot = (cuerpos or self._cuerpos(jd))[foco]
        radio = self.radio_base * math.exp(log_escala)
        d = R * A.distancia_para_radio(radio, self.focal)
        return A.Camara.en_astro(giro, inclinacion, d, self.focal, self.w / 2.0, self.h / 2.0, centro, rot)

    def _espacio(self, log_escala, giro, inclinacion, fecha, foco="tierra", viaje=None, t=0.0):
        """Cielo y astros vistos desde la camara, de lejos a cerca (el Sol tambien es un astro)."""
        b = self.buf
        jd = A.dia_juliano(fecha)
        cuerpos = self._cuerpos(jd)
        hiper = 0.0
        if viaje is not None:
            pos, objetivo, hiper, arriba = viaje.camara(t)
            cam = A.Camara.mirando(pos, objetivo, self.focal, self.w / 2.0, self.h / 2.0, arriba)
        else:
            cam = self.camara(log_escala, giro, inclinacion, foco, jd, cuerpos)
        sol_centro = cuerpos["sol"][0]

        self.cielo.fondo(b, cam, A.ecuatorial_a_tierra(jd))

        vistos = []
        for nombre, (centro, R, dib, rot) in cuerpos.items():
            x, y, z = cam.proyectar_punto(centro)
            dist = float(np.linalg.norm(cam.pos - centro))
            if z <= 0 or dist <= R:
                continue
            vistos.append((z, nombre, x, y, A.radio_aparente_px(R, dist, self.focal)))

        objetivos = {}
        for z, nombre, x, y, r in sorted(vistos, reverse=True):        # de lejos a cerca
            centro, R, dib, rot = cuerpos[nombre]
            if nombre == "sol":
                self.cielo.dibujar_sol(b, float(x), float(y), 1.0)        # resplandor: va detras del disco
            if r < C.PUNTO_ASTRO_MIN or (nombre == "sol" and r < 6.0):     # muy lejos: un punto (el Sol: solo su resplandor)
                if nombre != "sol" and 0 <= x < self.w and 0 <= y < self.h:
                    col = S.COLORES.get(nombre, (200, 200, 200))
                    cv2.circle(b, (int(round(x * 16)), int(round(y * 16))), 16, col, -1, cv2.LINE_AA, 4)
            else:
                M = cam.ejes_hacia(centro)                                 # los ejes de pantalla en el sitio del astro
                g, i, rod = A.orientacion_vista(M, rot)
                hacia_sol = (sol_centro - centro) / (np.linalg.norm(sol_centro - centro) + 1e-12)
                if nombre == "sol":                                        # no se ilumina: emite
                    hacia_sol = np.array([1.0, 0.0, 0.0])
                dib.dibujar(b, x, y, r, g, i, M @ hacia_sol, rot @ hacia_sol, rod)
            if -r < x < self.w + r and -r < y < self.h + r:
                objetivos[nombre] = (float(x), float(y), float(r))
        for z, nombre, x, y, r in vistos:                                # tapado por otro: no se elige
            for z2, n2, x2, y2, r2 in vistos:
                if n2 != nombre and z2 < z and math.hypot(x - x2, y - y2) + r < r2:
                    objetivos.pop(nombre, None)

        c_foco, R_foco = cuerpos[foco][0], cuerpos[foco][1]
        self.astros = {
            "modo": "astro",
            "camara": cam,
            "foco": foco,
            "altitud_km": (float(np.linalg.norm(cam.pos - c_foco)) - R_foco) * A.RADIO_TIERRA_KM,
            "dist_km": {n: float(np.linalg.norm(c[0] - c_foco)) * A.RADIO_TIERRA_KM for n, c in cuerpos.items()},
            "luna_km": float(np.linalg.norm(cuerpos["luna"][0])) * A.RADIO_TIERRA_KM,
            "sol_km": A.sol(jd)[1],
            "sol_px": objetivos["sol"][:2] if "sol" in objetivos else None,
            "objetivos": objetivos,
            "hiper": hiper,
        }
        if C.ETIQUETAS_ASTROS and hiper < 0.05:
            self._etiquetas()

    def _sistema(self, log_escala, giro, inclinacion, fecha, foco):
        self.astros = self.vista_sistema.dibujar(self.buf, A.dia_juliano(fecha), log_escala, giro, inclinacion, foco)

    def medir_texto(self, texto, tam, negrita):
        alfa = self.texto.sello(texto, tam, COL_TEXTO, negrita)[1]
        return alfa.shape[1], alfa.shape[0]

    def dibujar_etiquetas(self, etiquetas, discos):
        """Acomoda (universo/etiquetas.py) y dibuja etiquetas de astros sin que se encimen."""
        for e in ET.colocar(ET.agrupar(etiquetas), self.medir_texto, self.w, self.h, discos, (8, 72, 8, 78),
                            cajas=self._ocupado):
            for (texto, tam, col, neg), (x, y) in zip(e.lineas, e.pos):
                self.texto.dibujar(self.buf, texto, x, y, tam, col, neg)

    def _etiquetas(self):
        a = self.astros
        etiquetas, discos = [], []
        for nombre, (x, y, r) in a["objetivos"].items():
            if nombre == a["foco"]:
                continue                                    # (el astro en foco puede quedar debajo de una etiqueta)
            dist = a["dist_km"].get(nombre, 0.0)
            discos.append((x, y, max(r, 3.0)))
            etiquetas.append(ET.Etiqueta([nombre], x, y, r, [(NOMBRES.get(nombre, nombre), 15, COL_TEXTO, True),
                                                             (_km(dist), 12, COL_TENUE, False)],
                                         0.0 if nombre == "sol" else 1.0 + dist / 1e12))
        self.dibujar_etiquetas(etiquetas, discos)

    def _envolver(self, texto, tam, ancho):
        """Parte un texto en renglones que caben en `ancho` px."""
        renglones, actual = [], ""
        for palabra in texto.split():
            prueba = (actual + " " + palabra).strip()
            if actual and self.medir_texto(prueba, tam, False)[0] > ancho:
                renglones.append(actual)
                actual = palabra
            else:
                actual = prueba
        return renglones + ([actual] if actual else [])

    def _ficha_diseno(self, nombre):
        """-> (caja (x0, y0, x1, y1), renglones) de la ficha de un astro. Se calcula una vez."""
        if nombre in self._fichas_cache:
            return self._fichas_cache[nombre]
        f = FICHAS[nombre]
        W, pad = C.FICHA_ANCHO, 14
        col = max(self.medir_texto(e, 13, False)[0] for e, _ in f["filas"]) + 14
        filas = [(e, self._envolver(v, 13, W - 2 * pad - col)) for e, v in f["filas"]]
        tipo = self._envolver(f["tipo"], 13, W - 2 * pad)
        dato = self._envolver(f["dato"], 13, W - 2 * pad)
        alto = pad + 30 + 18 * len(tipo) + 8 + sum(19 * len(v) + 3 for _, v in filas) + 8 + 18 * len(dato) + pad
        x0, y1 = 20, self.h - 92
        caja = (x0, max(y1 - alto, 80), x0 + W, y1)
        self._fichas_cache[nombre] = (caja, (col, tipo, filas, dato))
        return self._fichas_cache[nombre]

    def _ficha_sprite(self, nombre):
        """La ficha ya compuesta, una vez por astro: (1 - alfa) y color premultiplicado (float32).
        Se dibuja sobre negro y sobre blanco: de la diferencia sale la transparencia de cada pixel."""
        clave = ("sprite", nombre)
        if clave not in self._fichas_cache:
            (x0, y0, x1, y1), _ = self._ficha_diseno(nombre)
            negro = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            blanco = np.full_like(negro, 255)
            for lienzo in (negro, blanco):
                self._ficha_en(lienzo, nombre)
            inv = (blanco.astype(np.float32) - negro.astype(np.float32)) / 255.0
            self._fichas_cache[clave] = (inv, negro.astype(np.float32))
        return self._fichas_cache[clave]

    def _ficha_en(self, lienzo, nombre):
        """Dibuja la ficha sobre un lienzo del tamano del panel."""
        _, (col, tipo, filas, dato) = self._ficha_diseno(nombre)
        pad, T = 14, self.texto
        cv2.addWeighted(lienzo, 0.38, np.full_like(lienzo, (22, 16, 12)), 0.62, 0, dst=lienzo)
        cv2.rectangle(lienzo, (0, 0), (lienzo.shape[1] - 1, lienzo.shape[0] - 1), (90, 80, 70), 1, cv2.LINE_AA)
        y = pad
        T.dibujar(lienzo, NOMBRES.get(nombre, nombre), pad, y, 22, COL_TEXTO, True)
        y += 30
        for renglon in tipo:
            T.dibujar(lienzo, renglon, pad, y, 13, COL_DER)
            y += 18
        y += 8
        for etiqueta, valor in filas:
            T.dibujar(lienzo, etiqueta, pad, y, 13, COL_TENUE)
            for renglon in valor:
                T.dibujar(lienzo, renglon, pad + col, y, 13, COL_TEXTO)
                y += 19
            y += 3
        y += 8
        for renglon in dato:
            T.dibujar(lienzo, renglon, pad, y, 13, COL_IZQ)
            y += 18

    def _ficha(self, nombre, opacidad=1.0):
        """Ficha educativa (universo/fichas.py) abajo a la izquierda, sobre un panel translucido."""
        if nombre not in FICHAS:
            return
        (x0, y0, x1, y1), _ = self._ficha_diseno(nombre)
        inv, pre = self._ficha_sprite(nombre)
        roi = self.buf[y0:y1, x0:x1]
        if opacidad < 0.999:                               # desvaneciendose: mezcla con la opacidad
            inv = 1.0 - opacidad * (1.0 - inv)
            pre = pre * opacidad
        roi[:] = cv2.convertScaleAbs(cv2.add(cv2.multiply(roi.astype(np.float32), inv), pre))

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

    def _hud(self, log_escala, giro, inclinacion, estado, info, ver_info, modo="astro"):
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
        if modo == "sistema":
            T.dibujar(b, f"SISTEMA SOLAR   ·   1 UA = {_km(A.UA_KM)}   ·   inclinación de la vista "
                         f"{self.astros.get('inclinacion_vista', 0.0):.0f}°   ·   escala {self.astros.get('px_ua', 0.0):.0f} px/UA",
                      20, 48, 16, COL_TEXTO)
        else:
            T.dibujar(b, f"{NOMBRES.get(self.astros.get('foco', 'tierra'), '')}   ·   Altitud {_km(alt)}   ·   "
                         f"Sobre {abs(lat):.0f}° {hemi}, {abs(lon):.0f}° {lado}", 20, 48, 16, COL_TEXTO)

        reloj = info.get("reloj")                          # (fecha, velocidad): el control del tiempo
        if reloj is not None:
            f, v = reloj
            T.dibujar(b, f"{texto_fecha(f)}   ·   {texto_velocidad(v)}", self.w - 16, 48, 15,
                      COL_TENUE if v == 1 else COL_IZQ, v != 1, ancla="d")

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
            T.dibujar(b, "Espacio pausa · R reset · S sistema · C comparar · G gravedad · X corte · K quiz · , . tiempo · T ahora · D ficha · H info · F full · Q salir",
                      self.w - 16, 22, 11, COL_TENUE, ancla="d")

        if estado.activo:
            msg = ("Izquierda: puño y sube o baja = zoom     ·     Derecha: swipe a los lados = girar, "
                   "arriba/abajo = inclinar; puño = agarrar     ·     OK = pausa")
        elif self.astros.get("objetivos"):
            msg = ("En pausa: deja la palma sobre un astro para viajar a él     ·     "
                   "S = " + ("volver" if modo == "sistema" else "ver todo el Sistema Solar"))
        else:
            msg = "Haz el signo de OK con cualquier mano para activar el control"
        T.dibujar(b, msg, self.w / 2, self.h - 40, 17 if not estado.activo else 15, COL_TEXTO, ancla="c")
        T.dibujar(b, "Ratón / touchpad: arrastrar = girar e inclinar  ·  rueda o dos dedos = zoom  ·  clic en un astro = viajar",
                  self.w / 2, self.h - 66, 13, COL_TENUE, ancla="c")

    def _avisos(self, estado, t):
        for i, (texto, t0) in enumerate(estado.avisos[-1:]):
            edad = t - t0
            op = 1.0 if edad < 1.2 else max(0.0, 1.0 - (edad - 1.2) / 0.8)
            col = COL_OK if "activ" in texto.lower() and "pausa" not in texto.lower() else COL_TEXTO
            self.texto.dibujar(self.buf, texto, self.w / 2, self.h - 104, 30, col, True, "c", op)