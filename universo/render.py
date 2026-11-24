"""Todo lo que se dibuja: el cielo real (estrellas, Via Lactea, Sol, Luna), la Tierra
vista desde la camara en orbita, guias de las manos, barra de zoom, textos (con
acentos, via Pillow) y la camara en miniatura."""

import math
import threading
import time
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
from universo.anillos import FACTOR_EXTERIOR, Saturno
from universo.interfaz import Interfaz
from universo.planetas import (textura_jupiter, textura_marte, textura_mercurio, textura_saturno, textura_sol,
                               textura_neptuno, textura_urano, textura_venus)
from universo.tierra import Tierra, niveles_esfera
from universo.vista_sistema import VistaSistema

COL_IZQ = (40, 178, 255)        # BGR ambar   -> zoom
COL_DER = (255, 208, 84)        # BGR cian    -> giro
COL_GRIS = (175, 175, 175)
COL_OK = (110, 230, 120)
COL_SALIDA = (255, 255, 255)    # anillo de carga para salir al Sistema Solar
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
    """Textos con Pillow (acentos, buena tipografia), en cache como "sellos" ya listos para mezclar.
    peso: False/True (normal / seminegrita) o un numero 100-900 (300 = fina, como el ejemplo de interfaz).
    espaciado: px extra entre letras (titulos en mayusculas, estilo consola espacial)."""
    # respaldo si no esta Exo 2 (config.UI_FUENTE): por peso, el primer archivo que exista
    RESPALDO = [
        (250, ["segoeuil.ttf", "C:/Windows/Fonts/segoeuil.ttf", "DejaVuSans-ExtraLight.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf"]),
        (350, ["segoeuisl.ttf", "C:/Windows/Fonts/segoeuisl.ttf", "segoeuil.ttf"]),
        (450, ["segoeui.ttf", "C:/Windows/Fonts/segoeui.ttf", "arial.ttf", "DejaVuSans.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]),
        (650, ["seguisb.ttf", "C:/Windows/Fonts/seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
        (1000, ["segoeuib.ttf", "C:/Windows/Fonts/segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
    ]

    def __init__(self):
        from PIL import Image, ImageDraw, ImageFont
        self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
        self._fuentes = {}
        self._nucleos = {}
        self._cache = OrderedDict()
        ruta = C.RAIZ / C.UI_FUENTE
        self.variable = str(ruta) if ruta.is_file() else None     # Exo 2 (fuente variable: un archivo, todos los pesos)

    @staticmethod
    def peso(negrita):
        if negrita is True:
            return 600
        if negrita is False or negrita is None:
            return 400
        return int(negrita)

    def _fuente(self, tam, negrita):
        p = self.peso(negrita)
        clave = (tam, p)
        if clave in self._fuentes:
            return self._fuentes[clave]
        f = None
        if self.variable is not None:
            try:
                f = self.ImageFont.truetype(self.variable, tam)
                f.set_variation_by_axes([p])
            except Exception:                              # Pillow sin soporte de fuentes variables
                f = None
        if f is None:
            for limite, nombres in self.RESPALDO:
                if p > limite:
                    continue
                for nombre in nombres:
                    try:
                        f = self.ImageFont.truetype(nombre, tam)
                        break
                    except OSError:
                        continue
                if f is not None:
                    break
        if f is None:
            f = self.ImageFont.load_default(tam)
        self._fuentes[clave] = f
        return f

    def _nucleo(self, borde):
        if borde not in self._nucleos:
            self._nucleos[borde] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * borde + 1, 2 * borde + 1))
        return self._nucleos[borde]

    def sello(self, texto, tam, color, negrita, espaciado=0):
        clave = (texto, tam, color, self.peso(negrita), espaciado)
        if clave in self._cache:
            self._cache.move_to_end(clave)
            return self._cache[clave]
        f = self._fuente(tam, negrita)
        borde = max(1, tam // 9)
        l, t, r, b = f.getbbox(texto, stroke_width=borde)
        extra = int(round(espaciado * max(len(texto) - 1, 0)))
        w, h = max(r - l + 2 + extra, 1), max(b - t + 2, 1)
        m_txt = self.Image.new("L", (w, h), 0)
        d = self.ImageDraw.Draw(m_txt)
        if espaciado:                                      # letra por letra, con aire entre ellas
            x = 1 - l
            for ch in texto:
                d.text((x, 1 - t), ch, font=f, fill=255)
                x += f.getlength(ch) + espaciado
        else:
            d.text((1 - l, 1 - t), texto, font=f, fill=255)    # (un solo trazado de Pillow)
        a_txt = np.asarray(m_txt, np.float32) / 255.0
        # el contorno es el texto "engordado" borde px: dilatar es ~50 veces mas barato que trazar otra vez con Pillow
        a_bor = cv2.dilate(a_txt, self._nucleo(borde))
        # contorno oscuro solo en textos claros (sobre un boton claro, un texto oscuro con contorno se ve borroso)
        alfa = np.maximum(a_txt, a_bor * 0.7) if sum(color) > 300 else a_txt
        pre = a_txt[..., None] * np.array(color, np.float32)
        s = ((pre + 0.5).astype(np.uint8), (alfa * 255 + 0.5).astype(np.uint8), -l + 1, -t + 1)
        self._cache[clave] = s
        if len(self._cache) > 600:
            self._cache.popitem(last=False)
        return s

    def ancho(self, texto, tam, negrita=False, espaciado=0):
        return self.sello(texto, tam, COL_TEXTO, negrita, espaciado)[1].shape[1] if texto else 0

    def dibujar(self, buf, texto, x, y, tam=18, color=COL_TEXTO, negrita=False, ancla="i", opacidad=1.0, espaciado=0):
        """ancla: 'i' izquierda, 'c' centro, 'd' derecha. y = linea superior del texto."""
        if not texto or opacidad <= 0.01:
            return 0
        pre, alfa, _, _ = self.sello(texto, tam, tuple(int(c) for c in color), negrita, espaciado)
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
        self.jupiter = bd("jupiter", C.TEXTURA_JUPITER, textura_jupiter, 2048, oceanos=False,     # sin superficie: bruma
                          color_atm=C.COLOR_ATM_JUPITER, desfase_lon=C.DESFASE_TEXTURA_JUPITER)
        self.neptuno = bd("neptuno", C.TEXTURA_NEPTUNO, textura_neptuno, 2048, oceanos=False,
                          color_atm=C.COLOR_ATM_NEPTUNO, desfase_lon=C.DESFASE_TEXTURA_NEPTUNO)
        self.urano = bd("urano", C.TEXTURA_URANO, textura_urano, 2048, oceanos=False,
                        color_atm=C.COLOR_ATM_URANO, desfase_lon=C.DESFASE_TEXTURA_URANO)
        self.saturno = bd("saturno", C.TEXTURA_SATURNO, textura_saturno, 2048, oceanos=False, clase=Saturno,
                          color_atm=C.COLOR_ATM_SATURNO, desfase_lon=C.DESFASE_TEXTURA_SATURNO)
        self.sol = Tierra(C.TEXTURA_SOL, emisivo=True, color_atm=C.COLOR_HALO_SOL,
                          desfase_lon=C.DESFASE_TEXTURA_SOL, respaldo=lambda: textura_sol(1024))
        self.texto = Texto()
        self.ui = Interfaz(self)
        self.buf = np.zeros((alto, ancho, 3), np.uint8)
        self._lentos = {}                 # clave -> (cuando, texto): textos que cambian siempre y se refrescan solo unas veces por s
        self.radio_base = C.RADIO_BASE_REL * alto
        self._rastros = {}
        self.astros = {}
        self.vista_sistema = VistaSistema(self)
        self._modo, self._fundido = "astro", None
        self.explosion = None                              # explosion.Explosion (la pone componer desde info)
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
                 inclinacion=None, fecha=None, foco="tierra", viaje=None, seleccion=None, modo="astro",
                 salida=None):
        if inclinacion is None:
            inclinacion = estado.inclinacion
        self._foco_actual = foco                          # para no liberar su textura (carga bajo demanda)
        self.explosion = info.get("explosion")
        # salida (viaje.Salida): vuelo hacia afuera; primero en la vista del astro, luego en la del sistema
        fase = salida.estado(t) if salida is not None else None
        if fase is not None:
            modo = fase["modo"]
        ficha = info.get("ficha")                          # (nombre, opacidad): ficha educativa del astro
        ficha_visible = (ficha is not None and modo == "astro" and viaje is None and fase is None
                         and ficha[1] > 0.01)
        menu = info.get("menu")
        n_int = len(getattr(menu, "interruptores", ()) or ())
        derecha = self.ui.caja_camara(n_int) if ver_camara else self.ui.caja_vista(max(n_int, 1))
        self._ocupado = [(0, 0, self.w, C.UI_BARRA_ALTO + 8),            # zonas sin etiquetas de astros
                         (derecha[0], C.UI_PANEL_Y, derecha[2], derecha[3]),
                         (self.w / 2 - 360, self.h - 120, self.w / 2 + 360, self.h)]
        if ficha_visible and ficha[0] in FICHAS:
            self._ocupado.append(self.ui.caja_astro(ficha[0]))
        self._destino = viaje.nombre if viaje is not None else None
        if modo != self._modo:                             # fundido entre la vista de un astro y la del sistema
            if fase is None:                               # (la salida no lo necesita: el corte coincide)
                self._fundido = (self.buf.copy(), t, C.SISTEMA_TRANSICION)
            self._modo = modo
        if modo == "sistema":
            self._sistema(log_escala, giro, inclinacion, fecha or fecha_actual(), foco,
                          fase if fase is not None and fase["modo"] == "sistema" else None)
        else:
            self._espacio(log_escala, giro, inclinacion, fecha or fecha_actual(), foco, viaje, t,
                          fase["camara"] if fase is not None else None)
        if seleccion is not None:
            self._seleccion(seleccion)
        modos = info.get("modos")
        comparando = bool(modos and modos[2].activo)       # el comparador usa toda la pantalla: lo demas se aparta
        quiz_activo = bool(modos and modos[3].activo)
        if self.resaltado is not None and modo == self._modo and not comparando:
            x, y, r, nombre = self.resaltado
            cv2.circle(self.buf, _p(x, y), int((max(r, 12.0) + 8) * 16), (255, 255, 255), 1, cv2.LINE_AA, 4)
            accion = "volver" if (modo == "sistema" and nombre == foco) else "ir a"
            self.texto.dibujar(self.buf, f"clic: {accion} {NOMBRES.get(nombre, nombre)}" if accion == "ir a"
                               else f"clic: volver a {NOMBRES.get(nombre, nombre)}",
                               x, y + max(r, 12.0) + 16, 13, COL_TEXTO, False, "c")
        if modos is not None:
            corte, gravedad, comparador, quiz = modos
            if not comparando:
                corte.dibujar(self, foco)
                gravedad.dibujar(self, foco)
            comparador.dibujar(self, A.dia_juliano(fecha or fecha_actual()), t)
            quiz.dibujar(self, t)
        if estado.activo and not comparando and fase is None:
            self._barra_zoom(log_escala, estado.zoom_agarrado)
        if menu is not None:
            menu.dibujar(self)
        self._manos(estado)
        if ver_camara and frame is not None:
            self._camara(frame, estado, n_int)
        if ficha_visible and not comparando:
            self.ui.panel_astro(self.buf, *ficha)
        self._hud(log_escala, giro, inclinacion, estado, info, ver_info, modo, comparando or quiz_activo)
        self._avisos(estado, t)
        if self._fundido is not None:
            a = 1.0 - (t - self._fundido[1]) / max(self._fundido[2], 1e-3)
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

    def _espacio(self, log_escala, giro, inclinacion, fecha, foco="tierra", viaje=None, t=0.0, cam=None):
        """Cielo y astros vistos desde la camara, de lejos a cerca (el Sol tambien es un astro).
        cam: camara dada (la salida al sistema); sin etiquetas, como en el viaje."""
        b = self.buf
        jd = A.dia_juliano(fecha)
        cuerpos = self._cuerpos(jd)
        hiper = 0.0
        if viaje is not None:
            pos, objetivo, hiper, arriba = viaje.camara(t)
            cam = A.Camara.mirando(pos, objetivo, self.focal, self.w / 2.0, self.h / 2.0, arriba)
        elif cam is not None:
            hiper = 1.0
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
            ex = self.explosion
            if ex is not None and ex.oculta(nombre, t):                    # ya estallo: solo sus pedazos
                ex.dibujar(b, nombre, float(x), float(y), float(r), t)
                if -r < x < self.w + r and -r < y < self.h + r:
                    objetivos[nombre] = (float(x), float(y), float(r))
                continue
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
                if ex is not None and ex.necesita_captura(nombre) and r >= C.EXPLOSION_RADIO_MIN:
                    self._dibujar_y_capturar(dib, ex, nombre, x, y, r, (g, i, M @ hacia_sol, rot @ hacia_sol, rod), t)
                else:
                    dib.dibujar(b, x, y, r, g, i, M @ hacia_sol, rot @ hacia_sol, rod)
                if ex is not None:
                    ex.dibujar(b, nombre, float(x), float(y), float(r), t)  # la carga (grietas y brillo)
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

    def _dibujar_y_capturar(self, dib, ex, nombre, x, y, r, args, t):
        """Dibuja el astro y le pasa a la explosion su "foto": el parche y lo que cambio al dibujarlo."""
        b = self.buf
        ext = FACTOR_EXTERIOR if nombre == "saturno" else 1.12        # anillos / halo de la atmosfera
        m = int(math.ceil(r * ext)) + 3
        x0, y0 = max(int(x) - m, 0), max(int(y) - m, 0)
        x1, y1 = min(int(x) + m + 1, self.w), min(int(y) + m + 1, self.h)
        antes = b[y0:y1, x0:x1].copy()
        dib.dibujar(b, x, y, r, *args)
        despues = b[y0:y1, x0:x1]
        if x1 <= x0 or y1 <= y0:
            return
        mascara = cv2.absdiff(antes, despues).max(axis=2) > 6       # anillos y halo
        disco = np.zeros(mascara.shape, np.uint8)                   # y todo el disco (el lado de noche es negro)
        cv2.circle(disco, (int(round((x - x0) * 16)), int(round((y - y0) * 16))), int(r * 16), 1, -1, cv2.LINE_8, 4)
        mascara |= disco.astype(bool)
        ex.capturar(despues.copy(), mascara, x - x0, y - y0, r, t)

    def _sistema(self, log_escala, giro, inclinacion, fecha, foco, transicion=None):
        self.astros = self.vista_sistema.dibujar(self.buf, A.dia_juliano(fecha), log_escala, giro, inclinacion, foco,
                                                 transicion)

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
                salida = getattr(v, "progreso_salida", 0.0)
                if v.accion == "zoom" and salida > 0:        # jalando bajo el tope: carga hacia el sistema
                    cv2.ellipse(b, _p(x, y), (22 * 16, 22 * 16), -90, 0, int(360 * salida),
                                COL_SALIDA, 5, cv2.LINE_AA, 4)
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
        self.ui.barra_zoom(self.buf, log_escala, agarrado)

    def _camara(self, frame, estado, n_interruptores=0):
        ancho = C.UI_PANEL_DER_ANCHO - 16
        alto = int(ancho * 3 / 4)
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
        self.ui.camara(self.buf, pip, n_interruptores)

    def _lento(self, clave, hz, producir):
        """Texto que cambia en cada cuadro (altitud, fps...): se rehace como mucho `hz` veces por segundo.
        Si no, la cache de sellos fallaba en cada cuadro y Pillow lo volvia a trazar (hasta 13 ms por cuadro)."""
        ahora = time.perf_counter()
        v = self._lentos.get(clave)
        if v is None or ahora - v[0] >= 1.0 / hz:
            v = self._lentos[clave] = (ahora, producir())
        return v[1]

    def _hud(self, log_escala, giro, inclinacion, estado, info, ver_info, modo="astro", comparando=False):
        b, ui = self.buf, self.ui
        ui.barra(b, estado, info, modo, giro, inclinacion)
        if ver_info:                                       # datos tecnicos (tecla H o el panel de vista)
            x, y = C.UI_MARGEN + C.UI_PANEL_IZQ_ANCHO + 64, C.UI_BARRA_ALTO + 12
            exp = info.get("exposicion")
            T = self.texto
            T.dibujar(b, self._lento("depuracion", C.INFO_REFRESCO_HZ, lambda:
                      f"dibujo {info['fps_render']:.0f} fps  ·  cámara {info.get('fps_cam', 0):.0f} fps "
                      f"(exp. {'auto' if exp is None else f'{exp:+.0f}'})  ·  detección {info['fps_det']:.0f} fps "
                      f"({info['lat_ms']:.0f} ms)  ·  giro {estado.omega:+.2f}  incl. {estado.omega_v:+.2f} rad/s  ·  swipe: {estado.swipe}"),
                      x, y, 11, COL_TENUE)
            y += 18
            for v in sorted(estado.manos, key=lambda v: v.rol):
                if not v.visible:
                    continue
                nombre = "Izquierda" if v.rol == "izq" else "Derecha"
                pose = "PUÑO" if v.puno else ("OK" if v.ok else "abierta")
                T.dibujar(b, self._lento(("mano", v.rol), C.INFO_REFRESCO_HZ, lambda:
                          f"{nombre}: {pose:7s}  puño {v.ratio_puno:.2f} (<{C.PUNO_ON})   "
                          f"ok {v.ratio_ok:.2f} (<{C.OK_ON})   dedos {v.estirados}/3"),
                          x, y, 11, COL_IZQ if v.rol == "izq" else COL_DER)
                y += 17
            T.dibujar(b, "Espacio pausa · R reset · , . tiempo · T ahora · D ficha · V cámara · H datos · F pantalla · Q salir",
                      x, y, 10, COL_TENUE)
        if comparando:                                     # comparador y quiz ponen su propia ayuda
            return
        if estado.activo:
            msg = "Izquierda: puño ↕ = zoom   ·   Derecha: desliza = girar, puño = arrastrar   ·   OK = pausa"
        elif self.astros.get("objetivos"):
            msg = "Palma sobre un astro o un botón = elegirlo   ·   OK = mover la cámara"
        else:
            msg = "Haz el signo de OK con la mano para empezar"
        menu = info.get("menu")
        tope = getattr(menu, "tope", None) or (self.h - C.UI_MARGEN - C.UI_BOTON_ALTO)
        self.ui.ayuda(b, msg, tope - 26)

    def _avisos(self, estado, t):
        for texto, t0 in estado.avisos[-1:]:
            edad = t - t0
            op = 1.0 if edad < 1.2 else max(0.0, 1.0 - (edad - 1.2) / 0.8)
            if op <= 0.0:
                continue
            col = (130, 214, 128) if "activ" in texto.lower() and "pausa" not in texto.lower() else COL_TEXTO
            self.ui.aviso(self.buf, texto, op, col)
