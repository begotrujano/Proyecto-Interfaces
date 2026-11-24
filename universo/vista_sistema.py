"""Vista del Sistema Solar completo ("orrery"): el Sol al centro, las orbitas elipticas reales y
cada planeta donde esta AHORA, visto desde arriba o inclinado, con el plano de la eclíptica.

Se maneja con los mismos gestos que un astro:
    mano izquierda (puno arriba/abajo) = zoom      mano derecha (swipe) = girar / inclinar
En pausa, la palma sobre un astro lo elige y viajas hasta el. Los tamanos de los astros aqui son
ICONOS (a escala real no se verian): lo que si es real son las distancias entre orbitas y las
posiciones.
"""

import math

import cv2
import numpy as np

from universo import astros as A
from universo import config as C
from universo import etiquetas as ET
from universo import sistema as S
from universo.cielo import sprite_sol
from universo.viaje import heliocentrica_astro, matriz_sistema

COL_TEXTO = (235, 235, 235)
COL_TENUE = (150, 150, 150)
COL_PENDIENTE = (120, 120, 120)
ANILLOS_UA = (0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 50)


def _p(x, y):
    return (int(round(x * 16)), int(round(y * 16)))


def _ua(ua):
    return f"{ua:.2f} UA" if ua >= 0.1 else f"{ua:.3f} UA"


def _fijo(v, f):
    return tuple(int(c * f) for c in v)


def _tramos(pts, m):
    """Pedazos de una curva cerrada donde m es True, cada uno como polilinea (un segmento se dibuja
    si sus dos puntos cumplen). None = la curva completa cumple."""
    if m.all():
        return None
    if not m.any():
        return []
    n = len(m)
    idx = (np.arange(n) + int(np.argmin(m))) % n             # empieza en un punto que no cumple: sin vuelta
    d = np.diff(np.concatenate([[0], m[idx].astype(np.int8), [0]]))
    return [pts[idx[a:b]] for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if b - a >= 2]


class VistaSistema:
    def __init__(self, escena):
        self.e = escena
        a_max = max(S.ELEMENTOS[n][0][0] * (1.0 + S.ELEMENTOS[n][1][0]) for n in C.PLANETAS_ACTIVOS)
        self.px_ua_base = C.SISTEMA_RADIO_REL * escena.h / a_max     # px por UA a zoom x1
        spr = sprite_sol(7.0, 0.16 * escena.h)
        self._glow = np.clip(spr * 150.0, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def matriz(self, giro, inclinacion):
        """Ejes de la vista (filas: derecha, abajo, hacia ti) en el marco ecliptico."""
        return matriz_sistema(giro, inclinacion)

    @staticmethod
    def _radio_icono(n, rp, rs):
        return rs if n == "sol" else rp * float(np.clip(S.radio_rt(n) ** 0.3, 0.7, 2.2))

    def dibujar(self, buf, jd, log_escala, giro, inclinacion, foco, transicion=None):
        """transicion: estado de viaje.Salida (ejes, centro y escala dados; iconos que crecen desde
        un punto; sin etiquetas). None = la vista normal, centrada en el Sol."""
        e = self.e
        w, h = e.w, e.h
        cx, cy = w / 2.0, h / 2.0
        tr = transicion
        if tr is None:
            zoom = math.exp(log_escala)
            s = self.px_ua_base * zoom                                 # px por UA
            M, tilt = self.matriz(giro, inclinacion)
            centro = np.zeros(3)
        else:
            s, M, centro = tr["px_ua"], tr["M"], np.asarray(tr["centro"], float)
            zoom = s / self.px_ua_base
            tilt = math.acos(float(np.clip(M[2, 2], -1.0, 1.0)))
        z_sol = float(M[2] @ centro)                                   # profundidad del centro respecto al Sol
        E = S.ecliptica_a_tierra(jd)                                   # ecliptico -> marco de la Tierra
        M_tierra = M @ E.T                                             # ejes de la vista en el marco de la Tierra

        # cielo real detras (las estrellas giran con la vista)
        cam = A.Camara(0.0, 0.0, 1.0, e.focal, cx, cy)
        cam.M = M_tierra
        e.cielo.fondo(buf, cam, A.ecuatorial_a_tierra(jd))

        def proy(P):
            q = (np.asarray(P, float) - centro) @ M.T
            if tr is None:
                return cx + s * q[..., 0], cy + s * q[..., 1], q[..., 2]
            # en la transicion: perspectiva desde una camara a dist_ua del centro (como la vista del astro
            # en el corte) que se vuelve ortografica; lo que queda detras de la camara es nan
            den = 1.0 - tr["perspectiva"] * q[..., 2] / tr["dist_ua"]
            inv = np.where(den > 0.05, 1.0 / np.maximum(den, 0.05), np.nan)
            return cx + s * q[..., 0] * inv, cy + s * q[..., 1] * inv, q[..., 2] + z_sol   # d: respecto al Sol

        ap = tr["aparicion"] if tr is not None else 1.0                 # en la transicion las orbitas aparecen:
        lineas = buf if ap >= 1.0 else buf.copy()                      # se dibujan en una copia y se mezclan
        nombres = [n for n in S.ORDEN if n in C.PLANETAS_ACTIVOS or C.SISTEMA_MOSTRAR_PENDIENTES]
        if ap > 0.01:
            self._anillos(lineas, proy, s, zoom, tr is None)
            puntos = 220                                               # de muy cerca, la orbita debe pasar por el planeta:
            if tr is not None:                                         # mas puntos (mismo error que a zoom x6)
                puntos = int(np.clip(220.0 * math.sqrt(s / 57.0), 220, 1000))
            for n in nombres:
                self._orbita(lineas, proy, n, jd, n in C.PLANETAS_ACTIVOS, puntos)
            if lineas is not buf:
                cv2.addWeighted(buf, 1.0 - ap, lineas, ap, 0.0, dst=buf)

        # cuerpos: de lejos a cerca
        cuerpos = e._cuerpos(jd)
        items = [("sol", np.zeros(3))] + [(n, S.heliocentrica(n, jd)) for n in nombres]
        zoom_ic = 1.0 if tr is not None else zoom                      # en la transicion: los iconos finales...
        rp = float(np.clip(C.SISTEMA_ICONO_PLANETA * zoom_ic ** 0.35, 3.0, 60.0))
        rs = float(np.clip(C.SISTEMA_ICONO_SOL * zoom_ic ** 0.25, 5.0, 90.0))
        foco_o = "tierra" if foco == "luna" else foco
        k = 1.0
        if tr is not None:                                             # ...que crecen desde el punto que era el astro
            k = (C.SALIDA_RADIO_CORTE / self._radio_icono(foco_o, rp, rs)) ** (1.0 - tr["progreso"])
            if "luna" in cuerpos:                                      # la Luna sigue visible junto a la Tierra
                items.append(("luna", heliocentrica_astro("luna", jd, cuerpos["luna"][0])))
        proyectados = []
        for n, P in items:
            x, y, d = proy(P)
            proyectados.append((float(d), n, float(x), float(y), P))
        objetivos, etiquetas = {}, []
        for d, n, x, y, P in sorted(proyectados, key=lambda c: c[0]):
            activo = n == "sol" or n in C.PLANETAS_ACTIVOS
            if activo and n in cuerpos:                                 # los grandes se ven mas grandes (suave)
                r = self._radio_icono(n, rp, rs)
            else:
                r = max(1.8, rp * 0.28)
            if tr is not None:
                if not (math.isfinite(x) and math.isfinite(y)):           # detras de la camara: no se ve
                    continue
                r *= k * (ap if n == "sol" else 1.0)
            if n == "sol":
                if ap < 1.0:                                           # el Sol como se veia desde el astro...
                    e.cielo.dibujar_sol(buf, x, y, 1.0 - ap)
                self._brillo(buf, x, y, ap)                            # ...que se vuelve el icono del sistema
                if r < 1.5:
                    continue
            if activo and n in cuerpos and r >= 1.5:
                centro, R, dib, rot = cuerpos[n]
                dir_sol = -P / np.linalg.norm(P) if n != "sol" else np.array([1.0, 0.0, 0.0])
                g, i, rod = A.orientacion_vista(M_tierra, rot)
                dib.dibujar(buf, x, y, r, g, i, M @ dir_sol, rot @ (E @ dir_sol), rod)
                objetivos[n] = (x, y, r)
            else:                                                       # planeta pendiente (o muy chico): un punto
                cv2.circle(buf, _p(x, y), int(max(r, 1.0) * 16), _fijo(S.COLORES[n], 0.7), -1, cv2.LINE_AA, 4)
            if n == "luna":
                continue
            etiquetas.append((n, x, y, r, activo, float(np.linalg.norm(P))))
            if n == foco_o and tr is None:
                cv2.circle(buf, _p(x, y), int((r + 7) * 16), (255, 255, 255), 1, cv2.LINE_AA, 4)
        lista, discos = [], [(x, y, r) for _, x, y, r, _, _ in etiquetas]
        for n, x, y, r, activo, dist_ua in etiquetas:
            if not activo and math.hypot(x - cx, y - cy) < 70:             # pendientes cerca del Sol: sin nombre (estorba)
                continue
            col = COL_TEXTO if activo else COL_PENDIENTE
            lineas = [(S.NOMBRES[n], 15 if activo else 13, col, activo)]
            if n != "sol":
                lineas.append((_ua(dist_ua) + " del Sol", 12, COL_TENUE if activo else COL_PENDIENTE, False))
            prioridad = 0.0 if n == "sol" else 1.0 if n == foco_o else (2.0 if activo else 10.0) + dist_ua / 100.0
            lista.append(ET.Etiqueta([n], x, y, r, lineas, prioridad))
        if tr is None:
            e.dibujar_etiquetas(lista, discos)
        return {"modo": "sistema", "camara": None, "foco": foco, "objetivos": objetivos, "hiper": 0.0,
                "px_ua": s, "inclinacion_vista": math.degrees(tilt), "altitud_km": 0.0, "dist_km": {},
                "luna_km": 0.0, "sol_km": 0.0, "sol_px": None}

    # ------------------------------------------------------------------
    def _brillo(self, buf, x, y, k=1.0):
        if k <= 0.01:
            return
        spr = self._glow if k >= 0.999 else cv2.convertScaleAbs(self._glow, alpha=k)
        S_ = spr.shape[0]
        x0, y0 = int(round(x)) - S_ // 2, int(round(y)) - S_ // 2
        xa, xb = max(x0, 0), min(x0 + S_, self.e.w)
        ya, yb = max(y0, 0), min(y0 + S_, self.e.h)
        if xb > xa and yb > ya:
            buf[ya:yb, xa:xb] = cv2.add(buf[ya:yb, xa:xb], spr[ya - y0:yb - y0, xa - x0:xb - x0])

    def _linea(self, buf, x, y, d, col, grosor=1, atras=0.4):
        """Curva cerrada: la mitad que pasa detras del plano (d < 0) se ve mas tenue. Los puntos nan
        (detras de la camara, en la transicion) cortan la curva."""
        xy = np.stack([x * 16, y * 16], -1)
        valido = np.isfinite(xy).all(-1)
        valido[valido] = (np.abs(xy[valido]) < 1e8).all(-1)
        pts = np.round(np.where(valido[:, None], xy, 0.0)).astype(np.int32)
        for m, c in ((valido, _fijo(col, atras)), ((d >= 0) & valido, col)):
            tramos = _tramos(pts, m)
            if tramos is None:
                cv2.polylines(buf, [pts], True, c, grosor, cv2.LINE_AA, 4)
            elif tramos:
                cv2.polylines(buf, tramos, False, c, grosor, cv2.LINE_AA, 4)

    def _orbita(self, buf, proy, nombre, jd, activo, puntos=220):
        x, y, d = proy(S.orbita(nombre, jd, puntos))
        col = _fijo(S.COLORES[nombre], 0.75 if activo else 0.35)
        self._linea(buf, x, y, d, col, 1, 0.45)

    def _anillos(self, buf, proy, s, zoom, rotulos=True):
        """Circulos de referencia en el plano de la eclíptica: 1 UA = distancia Tierra-Sol."""
        t = np.linspace(0.0, 2 * math.pi, 120, endpoint=False)
        for ua in ANILLOS_UA:
            if not 38.0 < s * ua < 2.2 * max(self.e.w, self.e.h):
                continue
            P = np.stack([ua * np.cos(t), ua * np.sin(t), np.zeros_like(t)], -1)
            x, y, d = proy(P)
            self._linea(buf, x, y, d, (62, 62, 62), 1, 0.55)
            if not rotulos:
                continue
            if not np.isfinite(x).any():
                continue
            k = int(np.nanargmax(x))                                    # etiqueta en el punto mas a la derecha
            if s * ua > 80 and 30 < x[k] < self.e.w - 60 and 60 < y[k] < self.e.h - 70:
                self.e.texto.dibujar(buf, f"{ua:g} UA", x[k] + 6, y[k] - 8, 11, (95, 95, 95))
