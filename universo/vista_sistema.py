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
from universo import sistema as S
from universo.cielo import sprite_sol

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
        tilt = float(np.clip(math.radians(C.SISTEMA_INCLINACION) + inclinacion, -1.50, 1.50))
        ct, st = math.cos(tilt), math.sin(tilt)
        M0 = np.array([[1.0, 0.0, 0.0], [0.0, -ct, -st], [0.0, -st, ct]])
        return M0 @ A.rot_z(giro), tilt

    def dibujar(self, buf, jd, log_escala, giro, inclinacion, foco):
        e = self.e
        w, h = e.w, e.h
        cx, cy = w / 2.0, h / 2.0
        zoom = math.exp(log_escala)
        s = self.px_ua_base * zoom                                     # px por UA
        M, tilt = self.matriz(giro, inclinacion)
        E = S.ecliptica_a_tierra(jd)                                   # ecliptico -> marco de la Tierra
        M_tierra = M @ E.T                                             # ejes de la vista en el marco de la Tierra

        # cielo real detras (las estrellas giran con la vista)
        cam = A.Camara(0.0, 0.0, 1.0, e.focal, cx, cy)
        cam.M = M_tierra
        e.cielo.fondo(buf, cam, A.ecuatorial_a_tierra(jd))

        def proy(P):
            q = np.asarray(P, float) @ M.T
            return cx + s * q[..., 0], cy + s * q[..., 1], q[..., 2]

        self._anillos(buf, proy, s, zoom)
        nombres = [n for n in S.ORDEN if n in C.PLANETAS_ACTIVOS or C.SISTEMA_MOSTRAR_PENDIENTES]
        for n in nombres:
            self._orbita(buf, proy, n, jd, n in C.PLANETAS_ACTIVOS)

        # cuerpos: de lejos a cerca
        cuerpos = e._cuerpos(jd)
        items = [("sol", np.zeros(3))] + [(n, S.heliocentrica(n, jd)) for n in nombres]
        rp = float(np.clip(C.SISTEMA_ICONO_PLANETA * zoom ** 0.35, 3.0, 60.0))
        rs = float(np.clip(C.SISTEMA_ICONO_SOL * zoom ** 0.25, 5.0, 90.0))
        proyectados = []
        for n, P in items:
            x, y, d = proy(P)
            proyectados.append((float(d), n, float(x), float(y), P))
        objetivos, etiquetas = {}, []
        foco_o = "tierra" if foco == "luna" else foco
        for d, n, x, y, P in sorted(proyectados, key=lambda k: k[0]):
            activo = n == "sol" or n in C.PLANETAS_ACTIVOS
            r = rs if n == "sol" else rp
            if n == "sol":
                self._brillo(buf, x, y)
            if activo and n in cuerpos:
                centro, R, dib, rot = cuerpos[n]
                dir_sol = -P / np.linalg.norm(P) if n != "sol" else np.array([1.0, 0.0, 0.0])
                g, i, rod = A.orientacion_vista(M_tierra, rot)
                dib.dibujar(buf, x, y, r, g, i, M @ dir_sol, rot @ (E @ dir_sol), rod)
                objetivos[n] = (x, y, r)
            else:                                                       # planeta pendiente: solo un punto
                cv2.circle(buf, _p(x, y), int(max(1.8, rp * 0.28) * 16), _fijo(S.COLORES[n], 0.7), -1, cv2.LINE_AA, 4)
                r = max(1.8, rp * 0.28)
            etiquetas.append((n, x, y, r, activo, float(np.linalg.norm(P))))
            if n == foco_o:
                cv2.circle(buf, _p(x, y), int((r + 7) * 16), (255, 255, 255), 1, cv2.LINE_AA, 4)
        puestas = []                                                    # etiquetas ya colocadas (para no encimar)
        prioridad = {"sol": 0, foco_o: 1}
        for n, x, y, r, activo, dist_ua in sorted(etiquetas, key=lambda k: (prioridad.get(k[0], 2 if k[4] else 3), k[5])):
            if not (20 < x < w - 90 and 50 < y < h - 60):
                continue
            if any(abs(x - px) < 100 and abs(y - py) < 26 for px, py in puestas):
                continue
            puestas.append((x, y))
            if not activo and math.hypot(x - cx, y - cy) < 70:             # pendientes cerca del Sol: sin nombre (estorba)
                continue
            col = COL_TEXTO if activo else COL_PENDIENTE
            e.texto.dibujar(buf, S.NOMBRES[n], x + r + 8, y - 10, 15 if activo else 13, col, activo)
            if n != "sol":
                e.texto.dibujar(buf, _ua(dist_ua) + " del Sol", x + r + 8, y + 8, 12, COL_TENUE if activo else COL_PENDIENTE)
        return {"modo": "sistema", "camara": None, "foco": foco, "objetivos": objetivos, "hiper": 0.0,
                "px_ua": s, "inclinacion_vista": math.degrees(tilt), "altitud_km": 0.0, "dist_km": {},
                "luna_km": 0.0, "sol_km": 0.0, "sol_px": None}

    # ------------------------------------------------------------------
    def _brillo(self, buf, x, y):
        spr = self._glow
        S_ = spr.shape[0]
        x0, y0 = int(round(x)) - S_ // 2, int(round(y)) - S_ // 2
        xa, xb = max(x0, 0), min(x0 + S_, self.e.w)
        ya, yb = max(y0, 0), min(y0 + S_, self.e.h)
        if xb > xa and yb > ya:
            buf[ya:yb, xa:xb] = cv2.add(buf[ya:yb, xa:xb], spr[ya - y0:yb - y0, xa - x0:xb - x0])

    def _linea(self, buf, x, y, d, col, grosor=1, atras=0.4):
        """Curva cerrada: la mitad que pasa detras del plano (d < 0) se ve mas tenue."""
        pts = np.stack([np.round(x * 16), np.round(y * 16)], -1).astype(np.int32)
        cv2.polylines(buf, [pts], True, _fijo(col, atras), grosor, cv2.LINE_AA, 4)
        delante = d >= 0
        sig = np.roll(delante, -1)
        tramos = [np.stack([pts[k], pts[(k + 1) % len(pts)]]) for k in np.nonzero(delante & sig)[0]]
        if tramos:
            cv2.polylines(buf, tramos, False, col, grosor, cv2.LINE_AA, 4)

    def _orbita(self, buf, proy, nombre, jd, activo):
        x, y, d = proy(S.orbita(nombre, jd, 220))
        col = _fijo(S.COLORES[nombre], 0.75 if activo else 0.35)
        self._linea(buf, x, y, d, col, 1, 0.45)

    def _anillos(self, buf, proy, s, zoom):
        """Circulos de referencia en el plano de la eclíptica: 1 UA = distancia Tierra-Sol."""
        t = np.linspace(0.0, 2 * math.pi, 120, endpoint=False)
        for ua in ANILLOS_UA:
            if not 38.0 < s * ua < 2.2 * max(self.e.w, self.e.h):
                continue
            P = np.stack([ua * np.cos(t), ua * np.sin(t), np.zeros_like(t)], -1)
            x, y, d = proy(P)
            self._linea(buf, x, y, d, (62, 62, 62), 1, 0.55)
            k = int(np.argmax(x))                                       # etiqueta en el punto mas a la derecha
            if s * ua > 80 and 30 < x[k] < self.e.w - 60 and 60 < y[k] < self.e.h - 70:
                self.e.texto.dibujar(buf, f"{ua:g} UA", x[k] + 6, y[k] - 8, 11, (95, 95, 95))
