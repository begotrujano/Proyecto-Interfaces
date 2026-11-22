"""Cambiar de astro: elegirlo con la mano (en pausa) y viajar hasta el en hiperespacio.

Solo depende de numpy. No sabe nada de ventanas: render.py lo dibuja y main.py lo usa.

    Selector: con el control EN PAUSA, deja la palma sobre un astro; un anillo blanco
              se llena (como el OK) y al completarse se elige.
    Viaje:    no es un teletransporte. 1) la camara gira hacia el destino, 2) salta al
              hiperespacio (las estrellas se estiran en rayas), 3) sale cerca del astro
              a la misma escala con la que lo dejaste.
"""

import math

import numpy as np

from universo import config as C


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
