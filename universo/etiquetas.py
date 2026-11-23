"""Acomoda las etiquetas de los astros para que no se encimen entre si ni tapen a otros astros.

Solo python (sin numpy): son pocas etiquetas por cuadro. Lo usan la vista de un astro (render.py)
y la del sistema (vista_sistema.py).

  1. Los astros que caen casi en el mismo punto de la pantalla comparten una etiqueta
     ("Tierra · Luna" vista desde Venus).
  2. Por prioridad, cada etiqueta prueba posiciones alrededor de su astro (derecha, izquierda,
     arriba, abajo...) y se queda con la primera que no choca con otra etiqueta ni con un astro.
  3. Si ninguna cabe, no se dibuja (las de menor prioridad son las que se pierden).
"""

import math

from universo import config as C


class Etiqueta:
    def __init__(self, nombres, x, y, r, lineas, prioridad=0.0):
        """lineas: [(texto, tam, color, negrita)]; prioridad: menor = se coloca antes."""
        self.nombres, self.x, self.y, self.r = list(nombres), float(x), float(y), float(r)
        self.lineas, self.prioridad = list(lineas), prioridad
        self.caja = None                                  # (x0, y0, x1, y1) donde quedo, o None


def _choca(a, b, margen=3.0):
    return a[0] < b[2] + margen and b[0] < a[2] + margen and a[1] < b[3] + margen and b[1] < a[3] + margen


def _choca_disco(caja, x, y, r):
    cx = min(max(x, caja[0]), caja[2])
    cy = min(max(y, caja[1]), caja[3])
    return math.hypot(cx - x, cy - y) < r + 2.0


def agrupar(etiquetas, unir_px=None):
    """Junta las etiquetas cuyos astros estan a menos de unir_px: la de mayor prioridad se queda
    con su texto y su primera linea pasa a 'A · B'."""
    unir_px = C.ETIQUETA_UNIR_PX if unir_px is None else unir_px
    salida = []
    for e in sorted(etiquetas, key=lambda e: e.prioridad):
        for g in salida:
            if math.hypot(e.x - g.x, e.y - g.y) < unir_px + max(g.r, e.r) * 0.5:
                g.nombres += e.nombres
                texto, tam, col, neg = g.lineas[0]
                g.lineas[0] = (texto + " · " + e.lineas[0][0], tam, col, neg)
                g.r = max(g.r, e.r)
                break
        else:
            salida.append(e)
    return salida


def colocar(etiquetas, medir, ancho, alto, discos=(), bordes=(0, 60, 0, 60), interlinea=3, cajas=()):
    """Elige donde va cada etiqueta. medir(texto, tam, negrita) -> (w, h).
    discos: [(x, y, r)] que no se deben tapar (los astros). bordes: (izq, arriba, der, abajo) libres
    para el HUD. cajas: [(x0, y0, x1, y1)] ocupadas (p. ej. la ficha educativa).
    Devuelve las etiquetas colocadas, con .caja y .pos = [(x, y) de cada linea]."""
    puestas = []
    cajas = list(cajas)
    discos = list(discos)
    for e in sorted(etiquetas, key=lambda e: e.prioridad):
        tams = [medir(t, tam, neg) for t, tam, _, neg in e.lineas]
        w = max(t[0] for t in tams)
        h = sum(t[1] for t in tams) + interlinea * (len(tams) - 1)
        d = max(e.r, 4.0) + 9.0
        candidatos = [(e.x + d, e.y - h / 2), (e.x - d - w, e.y - h / 2),           # derecha, izquierda
                      (e.x - w / 2, e.y - d - h), (e.x - w / 2, e.y + d),          # arriba, abajo
                      (e.x + d * 0.7, e.y - d * 0.7 - h), (e.x + d * 0.7, e.y + d * 0.7),
                      (e.x - d * 0.7 - w, e.y - d * 0.7 - h), (e.x - d * 0.7 - w, e.y + d * 0.7)]
        for x0, y0 in candidatos:
            caja = (x0, y0, x0 + w, y0 + h)
            if caja[0] < bordes[0] or caja[1] < bordes[1] or caja[2] > ancho - bordes[2] or caja[3] > alto - bordes[3]:
                continue
            if any(_choca(caja, p.caja) for p in puestas) or any(_choca(caja, c) for c in cajas):
                continue
            if any(_choca_disco(caja, x, y, r) for x, y, r in discos):
                continue
            e.caja = caja
            e.pos, yy = [], y0
            for (tw, th), (_, _, _, _) in zip(tams, e.lineas):
                e.pos.append((x0, yy))
                yy += th + interlinea
            puestas.append(e)
            break
    return puestas
