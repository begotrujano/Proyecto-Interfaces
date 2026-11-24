"""Interfaz en pantalla con estilo de consola de mision espacial, pensada para todo publico:

    barra superior   logo y nombre a la izquierda; estado, altitud, coordenadas y fecha a la derecha
    panel izquierdo  el astro: nombre grande, que es, 5 numeros con barras que lo comparan con los demas y un dato curioso
    panel derecho    interruptores de vista (camara, ficha, datos tecnicos, voz) y debajo la camara con tus manos
    abajo al centro  los botones de los modos, con su tecla (S C G X K); encima, los de contexto

Todo lo fijo (fondos de paneles, botones) se pinta una vez como "lamina" (1 - alfa y color premultiplicado) y por
cuadro solo se mezcla; los textos que cambian siempre pasan por Escena._lento. Colores en BGR."""

import math
from datetime import datetime

import cv2
import numpy as np

from universo import config as C
from universo import sistema as S
from universo.fichas import DATOS_FICHA, FICHAS

FONDO = (24, 21, 19)
BORDE = (78, 72, 66)
LINEA = (56, 52, 48)
TEXTO = (238, 238, 238)
TENUE = (152, 148, 143)
APAGADO = (104, 100, 96)
TEAL = (186, 212, 112)
ROJO = (84, 82, 230)
VERDE = (130, 214, 128)
AMBAR = (60, 178, 250)
CLARO = (226, 229, 232)
OSCURO = (36, 33, 31)

FINA, NORMAL, MEDIA, NEGRITA = 300, 400, 500, 600
ORDEN = ["sol", "mercurio", "venus", "tierra", "luna", "marte", "jupiter", "saturno", "urano", "neptuno"]
TECLAS = {"sistema": "S", "comparar": "C", "gravedad": "G", "corte": "X", "quiz": "K"}


def poligono(w, h, c):
    """Rectangulo con las esquinas superior izquierda e inferior derecha cortadas."""
    c = int(min(c, w // 3, h // 3))
    return np.array([[c, 0], [w - 1, 0], [w - 1, h - 1 - c], [w - 1 - c, h - 1], [0, h - 1], [0, c]], np.int32)


def pintar_panel(lienzo, alfa=0.86, color=FONDO, borde=BORDE, chaflan=None):
    h, w = lienzo.shape[:2]
    pts = poligono(w, h, C.UI_CHAFLAN if chaflan is None else chaflan)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
    a = (m.astype(np.float32) * (alfa / 255.0))[..., None]
    lienzo[:] = (lienzo.astype(np.float32) * (1.0 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    if borde is not None:
        cv2.polylines(lienzo, [pts], True, borde, 1, cv2.LINE_AA)


def _miles(v):
    return f"{v:,.0f}".replace(",", " ")


def _frac_log(v, lo, hi):
    v = max(abs(v), 1e-9)
    return float(np.clip((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0.04, 1.0))


def filas_astro(nombre):
    """[(etiqueta, valor, unidad, fraccion 0..1 de la barra)]: la barra compara con el resto del Sistema Solar."""
    d = DATOS_FICHA[nombre]
    filas = [("DIÁMETRO", _miles(d["diametro_km"]), "km", _frac_log(d["diametro_km"], 3000, 1.5e6)),
             ("TEMPERATURA", f"{d['temp_c']:,}".replace(",", " ").replace("-", "−"), "°C",
              1.0 if d["temp_c"] > 500 else float(np.clip((d["temp_c"] + 230) / 730.0, 0.04, 1.0)))]
    if nombre == "sol":
        filas.append(("VUELTA A LA GALAXIA", "230", "mill. años", 1.0))
    else:
        v = d["vuelta_dias"]
        etiqueta = "VUELTA A LA TIERRA" if nombre == "luna" else "UN AÑO DURA"
        filas.append((etiqueta, _miles(v) if v >= 100 else f"{v:g}", "días", _frac_log(v, 20, 60200)))
    h = abs(d["giro_h"])
    dia = (f"{h:.0f}" if h >= 10 else f"{h:.1f}", "horas") if h < 48 else (_miles(h / 24.0), "días")
    filas.append(("UN DÍA DURA" + (" ↺" if d["giro_h"] < 0 else ""), dia[0], dia[1], _frac_log(h, 9, 6000)))
    n = d["lunas"]
    filas.append(("PLANETAS" if nombre == "sol" else "LUNAS", f"{n:02d}", "", math.log1p(n) / math.log1p(274) if n else 0.0))
    return filas


class Interfaz:
    def __init__(self, escena):
        self.e = escena
        self._laminas = {}

    # ---------------------------------------------------------------- laminas
    def lamina(self, clave, w, h, pintar):
        """Pinta una vez sobre negro y sobre blanco: de la diferencia sale la transparencia de cada pixel."""
        lam = self._laminas.get(clave)
        if lam is None:
            negro = np.zeros((h, w, 3), np.uint8)
            blanco = np.full((h, w, 3), 255, np.uint8)
            pintar(negro)
            pintar(blanco)
            lam = self._laminas[clave] = (cv2.subtract(blanco, negro), negro)
        return lam

    @staticmethod
    def poner(buf, x, y, lam, opacidad=1.0):
        inv, pre = lam
        h, w = inv.shape[:2]
        H, W = buf.shape[:2]
        x, y = int(x), int(y)
        xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if xb <= xa or yb <= ya or opacidad <= 0.01:
            return
        i, p = inv[ya - y:yb - y, xa - x:xb - x], pre[ya - y:yb - y, xa - x:xb - x]
        if opacidad < 0.999:
            i = (255.0 - (255.0 - i.astype(np.float32)) * opacidad).astype(np.uint8)
            p = (p.astype(np.float32) * opacidad).astype(np.uint8)
        roi = buf[ya:yb, xa:xb]
        roi[:] = cv2.add(cv2.multiply(roi, i, scale=1.0 / 255.0), p)

    def txt(self, buf, texto, x, y, tam, color=TEXTO, peso=NORMAL, ancla="i", esp=0, op=1.0):
        return self.e.texto.dibujar(buf, texto, x, y, tam, color, peso, ancla, op, esp)

    def ancho(self, texto, tam, peso=NORMAL, esp=0):
        return self.e.texto.ancho(texto, tam, peso, esp)

    # ---------------------------------------------------------------- cajas (las usan render y el menu)
    def caja_astro(self, nombre):
        lam = self._lamina_astro(nombre)
        h, w = lam[0].shape[:2]
        return (C.UI_MARGEN, C.UI_PANEL_Y, C.UI_MARGEN + w, C.UI_PANEL_Y + h)

    def caja_vista(self, n):
        x1 = self.e.w - C.UI_MARGEN
        return (x1 - C.UI_PANEL_DER_ANCHO, C.UI_PANEL_Y, x1, C.UI_PANEL_Y + 44 + n * C.UI_INTERRUPTOR_ALTO + 10)

    def caja_camara(self, n):
        x0, _, x1, y1 = self.caja_vista(n) if n else (self.e.w - C.UI_MARGEN - C.UI_PANEL_DER_ANCHO, 0, 0,
                                                      C.UI_PANEL_Y - 12)
        x1 = x0 + C.UI_PANEL_DER_ANCHO
        alto_img = int((C.UI_PANEL_DER_ANCHO - 16) * 3 / 4)
        return (x0, y1 + 12, x1, y1 + 12 + 34 + alto_img + 8)

    # ---------------------------------------------------------------- barra superior
    def _pintar_barra(self, lienzo):
        h, w = lienzo.shape[:2]
        a = 0.9
        lienzo[:] = (lienzo.astype(np.float32) * (1 - a) + np.array(FONDO, np.float32) * a).astype(np.uint8)
        cv2.line(lienzo, (0, h - 1), (w, h - 1), LINEA, 1)
        cx, cy = C.UI_MARGEN + 16, h // 2
        for r, grueso in ((14, 2), (6, -1)):
            pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
            if grueso < 0:
                cv2.fillPoly(lienzo, [pts], TEXTO, cv2.LINE_AA)
            else:
                cv2.polylines(lienzo, [pts], True, TEXTO, grueso, cv2.LINE_AA)
        x = cx + 30
        x += self.txt(lienzo, "ASTER", x, cy - 13, 22, TEXTO, FINA, esp=9) + 18
        self.txt(lienzo, "EXPLORADOR DEL SISTEMA SOLAR", x, cy - 6, 10, TENUE, MEDIA, esp=3)

    def barra(self, buf, estado, info, modo, giro, inclinacion):
        w, h = self.e.w, C.UI_BARRA_ALTO
        self.poner(buf, 0, 0, self.lamina(("barra", w), w, h, self._pintar_barra))
        items = []                                         # (etiqueta, valor, color del valor), de izquierda a derecha
        voz = info.get("voz")
        if voz is not None:
            items.append(("VOZ", {"escuchando": "ESCUCHA", "silencio": "SILENCIO"}.get(voz, voz.upper()),
                          TEAL if voz == "escuchando" else APAGADO))
        items.append(("", "ACTIVO" if estado.activo else "PAUSA", VERDE if estado.activo else ROJO))
        if modo == "sistema":
            items.append(("ESCALA", f"{self.e.astros.get('px_ua', 0.0):.0f} PX/UA", TEXTO))
        else:
            lon = (-math.degrees(giro) + 180.0) % 360.0 - 180.0
            lat = -math.degrees(inclinacion)
            alt = self.e.astros.get("altitud_km", 0.0)
            items.append(("ALT", self.e._lento("ui_alt", C.HUD_REFRESCO_HZ, lambda: (
                f"{alt / 1e6:,.1f} MILL. KM" if alt >= 1e6 else f"{alt:,.0f} KM").replace(",", " ")), TEXTO))
            items.append(("COORD", self.e._lento("ui_coord", C.HUD_REFRESCO_HZ, lambda:
                          f"{abs(lat):.0f}°{'S' if lat < -0.5 else 'N'} · {abs(lon):.0f}°{'O' if lon < 0 else 'E'}"), TEXTO))
        reloj = info.get("reloj")
        if reloj is not None:
            f, v = reloj
            if v != 1:
                from universo.reloj import texto_velocidad
                items.append(("TIEMPO", texto_velocidad(v).upper(), AMBAR))
            fl = f.astimezone()
            items.append(("", self.e._lento("ui_fecha", 2.0, lambda:
                          f"{fl.day:02d}.{fl.month:02d}.{fl.year} · {fl.hour:02d}:{fl.minute:02d}"), (205, 205, 205)))
        x = w - C.UI_MARGEN
        y = h // 2 - 8
        for etiqueta, valor, col in reversed(items):
            x -= self.txt(buf, valor, x, y, 12, col, NEGRITA, "d", esp=2)
            if etiqueta:
                x -= 8 + self.txt(buf, etiqueta, x - 8, y, 11, TENUE, MEDIA, "d", esp=2)
            if etiqueta == "VOZ" and voz == "escuchando":    # volumen del microfono: si no se mueve, no te oye
                ancho = int(36 * min(max(info.get("voz_nivel", 0.0), 0.0), 1.0))
                cv2.line(buf, (x, y + 22), (x + ancho, y + 22), TEAL, 2)
            x -= 30

    # ---------------------------------------------------------------- panel del astro
    def _diseno_astro(self, nombre):
        w, pad = C.UI_PANEL_IZQ_ANCHO, 24
        dato = self.e._envolver(FICHAS[nombre]["dato"], 13, w - 2 * pad) if nombre in FICHAS else []
        tipo = self.e._envolver(FICHAS[nombre]["tipo"], 13, w - 2 * pad) if nombre in FICHAS else []
        filas = filas_astro(nombre) if nombre in DATOS_FICHA else []
        y_filas = 98 + 19 * len(tipo) + 22
        y_dato = y_filas + 54 * len(filas) + 14
        alto = y_dato + 21 * len(dato) + 20
        return w, pad, alto, tipo, filas, dato, y_filas, y_dato

    def _lamina_astro(self, nombre):
        w, pad, alto, tipo, filas, dato, y_filas, y_dato = self._diseno_astro(nombre)

        def pintar(l):
            pintar_panel(l, 0.88)
            i = ORDEN.index(nombre) + 1 if nombre in ORDEN else 0
            self.txt(l, f"OBJETIVO · {i:02d} / {len(ORDEN):02d}", pad, 22, 10, TENUE, MEDIA, esp=3)
            nom = S.NOMBRES.get(nombre, nombre).upper()
            tam = 38
            while tam > 22 and self.ancho(nom, tam, FINA, 6) > w - 2 * pad:
                tam -= 2
            self.txt(l, nom, pad - 2, 42, tam, TEXTO, FINA, esp=6)
            y = 98
            for r in tipo:
                self.txt(l, r, pad, y, 13, TEAL, MEDIA)
                y += 19
            cv2.line(l, (pad, y + 10), (w - pad, y + 10), LINEA, 1)
            for k, (etq, val, uni, frac) in enumerate(filas):
                y0 = y_filas + 54 * k
                xu = w - pad
                if uni:
                    xu -= self.txt(l, uni, xu, y0 + 20, 10, TENUE, NORMAL, "d") + 5
                self.txt(l, val, xu, y0 + 2, 26, TEXTO, FINA, "d")
                self.txt(l, etq, pad, y0 + 22, 10, TENUE, MEDIA, esp=2)
                yb = y0 + 42
                cv2.line(l, (pad, yb), (w - pad, yb), LINEA, 1)
                cv2.line(l, (pad, yb), (pad + int((w - 2 * pad) * frac), yb), (215, 215, 215), 2)
            for k, r in enumerate(dato):
                self.txt(l, r, pad, y_dato + 21 * k, 13, (212, 212, 212), NORMAL)

        return self.lamina(("astro", nombre), w, alto, pintar)

    def panel_astro(self, buf, nombre, opacidad=1.0):
        if nombre not in DATOS_FICHA and nombre not in FICHAS:
            return
        self.poner(buf, C.UI_MARGEN, C.UI_PANEL_Y, self._lamina_astro(nombre), opacidad)

    # ---------------------------------------------------------------- panel de vista (interruptores)
    def panel_vista(self, buf, interruptores, hover, progreso):
        """interruptores: [(id, texto, encendido)]. Las cajas de cada renglon: cajas_interruptores."""
        if not interruptores:
            return
        x0, y0, x1, y1 = self.caja_vista(len(interruptores))
        w = x1 - x0

        def pintar(l):
            pintar_panel(l, 0.88)
            self.txt(l, "VISTA", 18, 18, 10, TENUE, MEDIA, esp=3)
            for k in range(len(interruptores)):
                yb = 44 + (k + 1) * C.UI_INTERRUPTOR_ALTO - 1
                if k < len(interruptores) - 1:
                    cv2.line(l, (18, yb), (w - 18, yb), LINEA, 1)

        self.poner(buf, x0, y0, self.lamina(("vista", len(interruptores), w), w, y1 - y0, pintar))
        for k, (iid, texto, on) in enumerate(interruptores):
            ya = y0 + 44 + k * C.UI_INTERRUPTOR_ALTO
            col = TEAL if iid == hover else (TEXTO if on else TENUE)
            self.txt(buf, texto, x0 + 18, ya + 13, 10, col, NEGRITA if on else MEDIA, esp=2)
            sx1, sy = x1 - 18, ya + 19
            sx0 = sx1 - 26
            cv2.rectangle(buf, (sx0, sy - 6), (sx1, sy + 6), TEXTO if on else APAGADO, 1)
            if on:
                cv2.rectangle(buf, (sx1 - 12, sy - 3), (sx1 - 3, sy + 3), TEXTO, -1)
            else:
                cv2.rectangle(buf, (sx0 + 3, sy - 3), (sx0 + 12, sy + 3), APAGADO, -1)
            if iid == hover and progreso > 0:
                cv2.line(buf, (x0 + 18, ya + C.UI_INTERRUPTOR_ALTO - 4),
                         (x0 + 18 + int((w - 36) * progreso), ya + C.UI_INTERRUPTOR_ALTO - 4), TEAL, 2)

    def cajas_interruptores(self, interruptores):
        x0, y0, x1, _ = self.caja_vista(len(interruptores))
        return [(iid, texto, on, x0, y0 + 44 + k * C.UI_INTERRUPTOR_ALTO, x1, y0 + 44 + (k + 1) * C.UI_INTERRUPTOR_ALTO)
                for k, (iid, texto, on) in enumerate(interruptores)]

    # ---------------------------------------------------------------- camara
    def camara(self, buf, pip, n_interruptores):
        x0, y0, x1, y1 = self.caja_camara(n_interruptores)
        w = x1 - x0

        def pintar(l):
            pintar_panel(l, 0.88)
            self.txt(l, "CÁMARA · TUS MANOS", 18, 14, 9, TENUE, MEDIA, esp=2)

        self.poner(buf, x0, y0, self.lamina(("camara", w, y1 - y0), w, y1 - y0, pintar))
        ih, iw = pip.shape[:2]
        buf[y0 + 34:y0 + 34 + ih, x0 + 8:x0 + 8 + iw] = pip
        cv2.rectangle(buf, (x0 + 8, y0 + 34), (x0 + 8 + iw - 1, y0 + 34 + ih - 1), LINEA, 1)

    # ---------------------------------------------------------------- botones (los pinta el menu)
    def lamina_boton(self, w, h, estado):
        def pintar(l):
            if estado == "activo":
                pintar_panel(l, 0.96, CLARO, CLARO, 9)
            else:
                pintar_panel(l, 0.84, FONDO, TEXTO if estado == "hover" else BORDE, 9)
        return self.lamina(("boton", w, h, estado), w, h, pintar)

    def boton(self, buf, bid, texto, x0, y0, x1, y1, estado, progreso=0.0, tecla=None):
        w, h = int(x1 - x0), int(y1 - y0)
        self.poner(buf, x0, y0, self.lamina_boton(w, h, estado))
        activo = estado == "activo"
        col = OSCURO if activo else TEXTO
        tam = 11 if h >= 36 else 10
        esp = 3 if h >= 36 else 1
        while esp > 0 and self.ancho(texto, tam, NEGRITA, esp) > w - (52 if tecla else 16):
            esp -= 1                                       # nombres largos: menos aire entre letras
        wl = self.ancho(texto, tam, NEGRITA, esp)
        cy = y0 + h / 2 - tam * 0.62
        if tecla:
            wk = self.ancho(tecla, 10, MEDIA)
            x = x0 + (w - (wk + 14 + wl)) / 2
            self.txt(buf, tecla, x, cy + 1, 10, (96, 94, 92) if activo else TENUE, MEDIA)
            self.txt(buf, texto, x + wk + 14, cy, tam, col, NEGRITA, esp=esp)
        else:
            self.txt(buf, texto, x0 + w / 2, cy, tam, col, NEGRITA, "c", esp=esp)
        if progreso > 0:
            cv2.line(buf, (int(x0) + 10, int(y1) - 5), (int(x0) + 10 + int((w - 20) * progreso), int(y1) - 5), TEAL, 2)

    # ---------------------------------------------------------------- ayuda, avisos, zoom
    def ayuda(self, buf, texto, y):
        self.txt(buf, texto, self.e.w / 2, y, 12, TENUE, NORMAL, "c", esp=1)

    def aviso(self, buf, texto, op, color=TEXTO):
        w = self.ancho(texto, 15, NEGRITA, 1) + 44
        h = 36
        x0, y0 = int(self.e.w / 2 - w / 2), C.UI_BARRA_ALTO + 14
        roi = buf[y0:y0 + h, x0:x0 + w]
        if roi.shape[:2] != (h, w):
            return
        lam = self.lamina(("aviso", w), w, h, lambda l: pintar_panel(l, 0.92, FONDO, BORDE, 8))
        self.poner(buf, x0, y0, lam, op)
        self.txt(buf, texto, self.e.w / 2, y0 + 8, 15, color, NEGRITA, "c", esp=1, op=op)

    def barra_zoom(self, buf, log_escala, agarrado):
        x = C.UI_MARGEN + C.UI_PANEL_IZQ_ANCHO + 30
        y0, y1 = int(self.e.h * 0.30), int(self.e.h * 0.70)
        lo, hi = math.log(C.ESCALA_MIN), math.log(C.ESCALA_MAX)
        col = AMBAR if agarrado else TENUE
        cv2.line(buf, (x, y0), (x, y1), LINEA, 1)
        for marca in (0.1, 1.0, 5.0):
            if C.ESCALA_MIN <= marca <= C.ESCALA_MAX:
                ym = int(y1 - (math.log(marca) - lo) / (hi - lo) * (y1 - y0))
                cv2.line(buf, (x - 4, ym), (x + 4, ym), TENUE, 1)
                self.txt(buf, f"×{marca:g}", x + 10, ym - 7, 10, TENUE, MEDIA)
        yv = int(y1 - (log_escala - lo) / (hi - lo) * (y1 - y0))
        pts = np.array([[x, yv - 6], [x + 6, yv], [x, yv + 6], [x - 6, yv]], np.int32)
        cv2.fillPoly(buf, [pts], col, cv2.LINE_AA)
        self.txt(buf, "ZOOM", x, y0 - 24, 10, col, MEDIA, "c", esp=3)
