"""Cambiar de astro: elegirlo con la mano (en pausa) y viajar hasta el en hiperespacio.

Solo depende de numpy. No sabe nada de ventanas: render.py lo dibuja y main.py lo usa.

    Selector: con el control EN PAUSA, deja la palma sobre un astro; un anillo blanco
              se llena (como el OK) y al completarse se elige.
    Viaje:    no es un teletransporte. 1) la camara gira hacia el destino, 2) salta al
              hiperespacio (las estrellas se estiran en rayas), 3) sale cerca del astro
              a la misma escala con la que lo dejaste.
    Salida:   del astro a la vista del Sistema Solar alejandose rapido (jalar el zoom bajo el
              tope); el cambio de vista queda oculto a medio vuelo.
"""

import copy
import math

import numpy as np

from universo import astros as A
from universo import config as C
from universo import sistema as S


def _suave(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Selector:
    def __init__(self):
        self.objetivo = None              # astro bajo la palma
        self.mano = None
        self.desde = None
        self.ult = -1.0

    def reiniciar(self):
        self.objetivo = self.mano = self.desde = None

    def actualizar(self, cursores, objetivos, actual, t):
        """cursores: [(id, x, y)] de las manos visibles. objetivos: {nombre: (x, y, radio_px)}.
        actual: el astro donde ya estas (no se puede elegir). -> (vista, elegido)
        vista = (x, y, progreso 0..1) para dibujar el anillo, o None; elegido = nombre o None."""
        encima = None
        for hid, x, y in cursores:
            for nombre, (ox, oy, r) in objetivos.items():
                if nombre == actual:
                    continue
                if math.hypot(x - ox, y - oy) <= max(r, C.SELECCION_RADIO_MIN) + C.SELECCION_MARGEN:
                    encima = (hid, nombre, x, y)
                    break
            if encima:
                break

        if encima and (encima[0], encima[1]) == (self.mano, self.objetivo):
            self.ult = t
        elif encima:
            self.mano, self.objetivo, self.desde, self.ult = encima[0], encima[1], t, t
        elif self.objetivo is not None and t - self.ult > C.SELECCION_TOLERANCIA:
            self.reiniciar()
        if self.objetivo is None:
            return None, None

        p = min((t - self.desde) / C.SELECCION_SOSTENER, 1.0)
        x, y = (encima[2], encima[3]) if encima else (None, None)
        if p >= 1.0 and encima:
            elegido = self.objetivo
            self.reiniciar()
            return (x, y, 1.0), elegido
        return ((x, y, p) if encima else None), None


class Viaje:
    """Trayectoria de la camara entre dos astros.

    pos0: donde esta la camara al salir; origen: centro del astro de salida;
    destino: centro del astro de llegada; radio_destino: su radio (radios terrestres);
    dist_final: a que distancia de su centro termina la camara."""

    def __init__(self, nombre, t0, pos0, origen, destino, radio_destino, dist_final,
                 arriba0=(0.0, 0.0, 1.0), arriba1=(0.0, 0.0, 1.0)):
        self.nombre = nombre
        self.t0 = t0
        self.dur = C.VIAJE_DURACION
        self.pos0 = np.asarray(pos0, float)
        self.origen = np.asarray(origen, float)
        self.destino = np.asarray(destino, float)
        d = self.pos0 - self.destino
        self.dir0 = d / np.linalg.norm(d)
        # rodeas el destino VIAJE_ARCO grados (de lado): al llegar, el astro de donde
        # saliste queda a un costado de la pantalla y puedes elegirlo para volver
        lado = np.cross([0.0, 0.0, 1.0], self.dir0)
        lado = lado / np.linalg.norm(lado) if np.linalg.norm(lado) > 1e-6 else np.array([0.0, 1.0, 0.0])
        a = math.radians(C.VIAJE_ARCO)
        self.dir_final = self.dir0 * math.cos(a) + lado * math.sin(a)
        self.arriba0 = np.asarray(arriba0, float) / np.linalg.norm(arriba0)    # norte del astro de salida
        self.arriba1 = np.asarray(arriba1, float) / np.linalg.norm(arriba1)    # norte del de llegada
        self.d0 = float(np.linalg.norm(d))
        self.d1 = max(dist_final, radio_destino * 1.05)

    def fase(self, t):
        return min(max((t - self.t0) / self.dur, 0.0), 1.0)

    def terminado(self, t):
        return t - self.t0 >= self.dur

    def camara(self, t):
        """-> (posicion, punto al que mira, intensidad del hiperespacio 0..1, hacia donde es "arriba").
        El "arriba" pasa del norte del astro de salida al del de llegada, para que al llegar la
        imagen no gire de golpe."""
        u = self.fase(t)
        g = C.VIAJE_GIRO                                  # fraccion del tiempo para voltear hacia el destino
        mira = _suave(u / g)
        objetivo = self.origen + (self.destino - self.origen) * mira
        avance = _suave((u - g) / (1.0 - g))
        # distancia al destino en escala logaritmica: el astro crece suave, sin "golpe" al final
        dist = math.exp(math.log(self.d0) + (math.log(self.d1) - math.log(self.d0)) * avance)
        a = math.radians(C.VIAJE_ARCO) * avance
        lado = (self.dir_final - self.dir0 * math.cos(math.radians(C.VIAJE_ARCO)))
        lado = lado / (np.linalg.norm(lado) + 1e-12)
        pos = self.destino + (self.dir0 * math.cos(a) + lado * math.sin(a)) * dist
        hiper = math.sin(math.pi * min(max((u - g * 0.6) / (1.0 - g * 0.6), 0.0), 1.0)) ** 0.7
        arriba = self.arriba0 * (1.0 - avance) + self.arriba1 * avance
        return pos, objetivo, hiper, arriba / np.linalg.norm(arriba)


# ----------------------------------------------------------------------
# Salida al Sistema Solar (jalar el zoom bajo el tope): un vuelo hacia afuera sin corte aparente
# ----------------------------------------------------------------------
def matriz_sistema(giro, inclinacion):
    """Ejes de la vista del Sistema Solar (filas: derecha, abajo, hacia ti) en el marco ecliptico,
    e inclinacion real de la vista (rad). La usan vista_sistema.py y la Salida."""
    tilt = float(np.clip(math.radians(C.SISTEMA_INCLINACION) + inclinacion, -1.50, 1.50))
    ct, st = math.cos(tilt), math.sin(tilt)
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, -ct, -st], [0.0, -st, ct]])
    return M0 @ A.rot_z(giro), tilt


def interpolar_ejes(M0, M1, w):
    """Ejes que van de M0 (w = 0) a M1 (w = 1) girando por el camino mas corto."""
    R = M1 @ M0.T
    ang = math.acos(min(max((np.trace(R) - 1.0) / 2.0, -1.0), 1.0))
    if ang < 1e-9:
        return (M1 if w >= 1.0 else M0).copy()
    if math.pi - ang < 1e-6:                              # media vuelta: el eje sale de R + I
        B = R + np.eye(3)
        k = B[:, int(np.argmax(np.linalg.norm(B, axis=0)))]
    else:
        k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    a = ang * min(max(w, 0.0), 1.0)
    return (np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)) @ M0


def heliocentrica_astro(nombre, jd, centro_tierra=None):
    """Posicion de un astro en el marco ecliptico heliocentrico (UA), como lo pone la vista del
    sistema. La Luna se calcula desde su centro en el marco de la Tierra (centro_tierra, en RT)."""
    if nombre == "sol":
        return np.zeros(3)
    if nombre == "luna":
        E = S.ecliptica_a_tierra(jd)
        return S.heliocentrica("tierra", jd) + E.T @ np.asarray(centro_tierra, float) / S.UA_RT
    return S.heliocentrica(nombre, jd)


class Salida:
    """Del astro en foco a la vista del Sistema Solar, alejandose rapido y sin corte aparente.

    1) Vista del astro (perspectiva): la camara retrocede en linea recta, cada vez mas rapido,
       con los mismos ejes; el astro se encoge hasta ser un punto.
    2) Cuando mide SALIDA_RADIO_CORTE px se pasa a la vista del sistema (ortografica) con el MISMO
       encuadre: centrada en el astro, mismos ejes (las estrellas no se mueven) y la escala que
       corresponde a esa distancia. Sigue alejandose a la misma velocidad mientras gira a la
       inclinacion de siempre (por el camino mas corto: queda del lado desde el que mirabas) y el
       centro pasa del astro al Sol.
    La distancia avanza en escala logaritmica con una sola curva suave: no hay salto de velocidad.

    cam: la camara al salir (marco de la Tierra); centro, radio: del astro (RT, marco de la Tierra);
    helio: el astro en la vista del sistema (UA); E: ecliptica -> Tierra; px_ua_final: escala final."""

    def __init__(self, t0, cam, centro, radio, helio, E, px_ua_final):
        self.t0 = t0
        self.dur = max(C.SALIDA_DURACION, 1e-3)
        self.cam0 = cam
        self.centro = np.asarray(centro, float)
        self.atras = cam.M[2].copy()
        self.f = cam.f
        d0 = max(float(np.linalg.norm(cam.pos - self.centro)), radio * 1.01)
        d_corte = radio * A.distancia_para_radio(C.SALIDA_RADIO_CORTE, self.f)
        self.L0 = math.log(d0)
        self.Lc = math.log(max(d_corte, d0 * 1.2))
        self.Le = max(math.log(self.f * S.UA_RT / px_ua_final), self.Lc + 1.0)
        self.px_ua_final = px_ua_final
        self.px_ua_corte = self.f * S.UA_RT / math.exp(self.Lc)
        lo, hi, meta = 0.0, 1.0, (self.Lc - self.L0) / (self.Le - self.L0)
        for _ in range(40):                               # fraccion del vuelo en la que se cambia de vista
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _suave(mid) < meta else (lo, mid)
        self.u_corte = hi
        self.helio = np.asarray(helio, float)
        self.M_ecl = cam.M @ E                            # ejes de la camara en el marco ecliptico
        # giro final: el de la vista del sistema mas parecida a la camara (traza maxima de M_t(g) Mc^T)
        N = self.M_ecl.T @ matriz_sistema(0.0, 0.0)[0]
        self.giro_final = math.atan2(N[0, 1] - N[1, 0], N[0, 0] + N[1, 1])
        self.M_final = matriz_sistema(self.giro_final, 0.0)[0]

    def fase(self, t):
        return min(max((t - self.t0) / self.dur, 0.0), 1.0)

    def terminado(self, t):
        return t - self.t0 >= self.dur

    def _log_dist(self, t):
        return self.L0 + (self.Le - self.L0) * _suave(self.fase(t))

    def estado(self, t):
        """{"modo": "astro", "camara"} en la primera parte; {"modo": "sistema", "M", "centro",
        "px_ua", "progreso", "dist_ua", "perspectiva", "aparicion"} despues del corte.
        "perspectiva" (1 -> 0): en el corte la vista del sistema proyecta igual que la camara (lo lejano
        no salta) y se vuelve ortografica, como la vista normal, al terminar. "aparicion" (0 -> 1 en
        SALIDA_FUNDIDO s): las orbitas se encienden y el Sol pasa de su sprite al icono."""
        L = self._log_dist(t)
        if L < self.Lc:
            cam = copy.copy(self.cam0)
            cam.distancia = math.exp(L)
            cam.pos = self.centro + self.atras * cam.distancia
            return {"modo": "astro", "camara": cam}
        q = (L - self.Lc) / (self.Le - self.Lc)
        s = self.f * S.UA_RT / math.exp(L)
        a = C.SALIDA_PAN_EXP
        rc = (self.px_ua_final / self.px_ua_corte) ** a
        wp = ((self.px_ua_final / s) ** a - rc) / max(1.0 - rc, 1e-12)
        wp = min(max(wp, 0.0), 1.0)
        wr = _suave(q)
        return {"modo": "sistema", "M": interpolar_ejes(self.M_ecl, self.M_final, wr),
                "centro": self.helio * (1.0 - wp), "px_ua": s, "progreso": wr,
                "dist_ua": math.exp(L) / S.UA_RT, "perspectiva": 1.0 - wr,
                "aparicion": min(max((self.fase(t) - self.u_corte) * self.dur / max(C.SALIDA_FUNDIDO, 1e-3),
                                     0.0), 1.0)}
