"""Manos: filtrado, seguimiento de hasta 2 manos con rol estable (izquierda/derecha)
y lectura de poses (puno, OK).

Solo depende de numpy: recibe los puntos que entrega MediaPipe (o los de una
prueba) y no sabe nada de camaras ni de ventanas.
"""

import math

import numpy as np

from universo import config as C

DOS_PI = 2.0 * math.pi
PALMA = (0, 5, 9, 13, 17)                       # muneca y nudillos: no cambian al cerrar la mano
DEDOS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def a_lienzo(p_norm, ancho, alto):
    """Coordenadas de camara (0..1) -> pixeles del lienzo, usando la zona comoda."""
    m = C.MARGEN_CAMARA
    k = 1.0 / (1.0 - 2.0 * m)
    return ((p_norm[0] - m) * k * ancho, (p_norm[1] - m) * k * alto)


# ----------------------------------------------------------------------
# Filtro One Euro: suaviza mucho con la mano quieta y casi nada al moverse rapido
# ----------------------------------------------------------------------
class OneEuro:
    def __init__(self, mincut, beta, dcut=1.0, vectorial=False):
        self.mincut, self.beta, self.dcut = mincut, beta, dcut
        self.vectorial = vectorial
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(corte, dt):
        tau = 1.0 / (DOS_PI * corte)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = max(t - self.t, 1e-4)
        self.t = t
        ad = self._alpha(self.dcut, dt)
        self.dx = ad * (x - self.x) / dt + (1.0 - ad) * self.dx
        vel = (np.linalg.norm(self.dx, axis=-1, keepdims=True) if self.vectorial
               else np.abs(self.dx))
        a = self._alpha(self.mincut + self.beta * vel, dt)
        self.x = a * x + (1.0 - a) * self.x
        return self.x.copy()


# ----------------------------------------------------------------------
# Medidas de pose (invariantes a distancia y orientacion si hay coordenadas 3D)
# ----------------------------------------------------------------------
def _puntos_pose(norm, world, ancho, alto):
    if world is not None:
        return np.asarray(world, dtype=np.float64)
    p = np.zeros((21, 3))
    p[:, 0] = norm[:, 0] * ancho
    p[:, 1] = norm[:, 1] * alto
    return p


def _dist_segmento(p, a, b):
    """Distancia del punto p al segmento a-b (2D o 3D)."""
    ab = b - a
    k = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + k * ab)))


def _ratio_aro(p):
    """Que tan cerrado esta el aro pulgar-indice, relativo al tamano de la palma.
    No solo punta con punta: vale que la punta del pulgar toque la ultima falange
    del indice (o al reves), que es como la gente hace el OK de verdad."""
    palma = max(float(np.linalg.norm(p[9] - p[0])), 1e-9)
    d = min(float(np.linalg.norm(p[4] - p[8])),
            _dist_segmento(p[4], p[7], p[8]),
            _dist_segmento(p[8], p[3], p[4]))
    return d / palma


def medir_pose(norm, world, ancho=640, alto=480):
    """-> (ratio_puno, ratio_ok, dedos_estirados)

    ratio_puno: punta-muneca / nudillo-muneca, promedio de 4 dedos (~2 abierta, ~1 puno)
    ratio_ok:   separacion del aro pulgar-indice / tamano de la palma (~0.1 tocandose).
                Se mide en 3D y en la imagen y se toma el mejor: la profundidad de
                MediaPipe es ruidosa justo cuando los dedos se tapan entre si.
    dedos_estirados: cuantos de medio, anular y menique estan estirados (0..3);
                basta con que esten rectos O lejos de la muneca (dedos relajados cuentan)
    """
    p = _puntos_pose(norm, world, ancho, alto)
    muneca = p[0]
    rs = []
    for mcp, _, _, punta in DEDOS:
        base = max(np.linalg.norm(p[mcp] - muneca), 1e-9)
        rs.append(np.linalg.norm(p[punta] - muneca) / base)

    ok = _ratio_aro(p)
    if world is not None and norm is not None:
        p2 = np.asarray(norm, dtype=np.float64)[:, :2] * np.array([ancho, alto], dtype=np.float64)
        ok = min(ok, C.OK_PESO_2D * _ratio_aro(p2))

    estirados = 0
    for (a, b, c, d), r in zip(DEDOS[1:], rs[1:]):
        largo = (np.linalg.norm(p[b] - p[a]) + np.linalg.norm(p[c] - p[b])
                 + np.linalg.norm(p[d] - p[c]))
        recto = np.linalg.norm(p[d] - p[a]) / max(largo, 1e-9) > C.OK_RECTITUD
        if recto or r > C.OK_EXTENSION:
            estirados += 1
    return float(np.mean(rs)), float(ok), estirados


# ----------------------------------------------------------------------
# Una mano
# ----------------------------------------------------------------------
class Mano:
    def __init__(self, hid, t):
        self.id = hid
        self.nace = t
        self.f_pts = OneEuro(C.OE_MINCUT, C.OE_BETA, C.OE_DCUT, vectorial=True)
        self.f_puno = OneEuro(3.0, 1.0)
        self.f_ok = OneEuro(3.0, 1.0)
        self.norm = None                 # (21, 2) filtrado, 0..1
        self.palma = None                # (2,) centro de la palma, 0..1
        self.vel = np.zeros(2)           # velocidad de la palma filtrada (0..1 por segundo)
        self.vel_cruda = np.zeros(2)     # velocidad sin filtrar (para detectar saltos del rastreo)
        self._palma_cruda = None
        self.visto = t
        self.visible = True
        self.ratio_puno = 2.0
        self.ratio_ok = 1.0
        self.estirados = 3
        self.puno = False
        self.ok = False
        self._on = self._off = None
        self.p_izq = 0.5                 # probabilidad (suavizada) de ser la mano izquierda
        self.rol = None                  # 'izq' | 'der'

    def edad(self, t):
        return t - self.nace

    @property
    def soltando(self):
        """Ya empezo a abrir el puno (aun sin confirmar): se congela el agarre para que
        abrir la mano mientras se mueve no cambie el zoom ni el giro."""
        return self.puno and self._off is not None

    def predicha(self, t):
        """Donde deberia estar la muneca ahora: permite cruzar las manos sin
        intercambiar identidades."""
        return self.norm[0] + self.vel * min(max(t - self.visto, 0.0), 0.1)

    def actualizar(self, norm, world, etiqueta, score, t, ancho, alto):
        previa = None if self.palma is None else (self.palma.copy(), self._palma_cruda, self.visto)
        cruda = np.asarray(norm)[list(PALMA)].mean(axis=0)
        self.norm = self.f_pts(norm, t)
        self.palma = self.norm[list(PALMA)].mean(axis=0)
        self._palma_cruda = cruda
        if previa is not None:
            dt = max(t - previa[2], 1e-3)
            self.vel = 0.5 * self.vel + 0.5 * (self.palma - previa[0]) / dt
            self.vel_cruda = (cruda - previa[1]) / dt
        self.visto = t
        self.visible = True

        rp, ro, est = medir_pose(norm, world, ancho, alto)
        self.ratio_puno = float(self.f_puno(rp, t))
        self.ratio_ok = float(self.f_ok(ro, t))
        self.estirados = est
        self._maquina_puno(t)
        # el OK solo cuenta con la mano casi quieta: un swipe de lado (pulgar delante
        # del indice en la imagen) o una mano que pasa no lo disparan
        quieta = float(np.linalg.norm(self.vel)) < C.OK_V_MAX
        if self.ok:
            self.ok = (self.ratio_ok < C.OK_OFF and est >= C.OK_DEDOS_MIN - 1
                       and not self.puno and quieta)
        else:
            self.ok = (self.ratio_ok < C.OK_ON and est >= C.OK_DEDOS_MIN
                       and not self.puno and quieta)

        if etiqueta in ("Left", "Right"):
            obs = score if etiqueta == "Left" else 1.0 - score
            self.p_izq = 0.75 * self.p_izq + 0.25 * obs if self.rol else obs

    def _maquina_puno(self, t):
        """Histeresis + antirrebote: ni parpadea ni se activa por ruido."""
        if not self.puno:
            if self.ratio_puno < C.PUNO_ON:
                self._on = t if self._on is None else self._on
                if t - self._on >= C.PUNO_DEBOUNCE_ON:
                    self.puno, self._off = True, None
            else:
                self._on = None
        else:
            if self.ratio_puno > C.PUNO_OFF:
                self._off = t if self._off is None else self._off
                if t - self._off >= C.PUNO_DEBOUNCE_OFF:
                    self.puno, self._on = False, None
            else:
                self._off = None


# ----------------------------------------------------------------------
# Seguimiento de hasta 2 manos
# ----------------------------------------------------------------------
class SeguidorManos:
    """Identidad por cercania con el cuadro anterior; rol izquierda/derecha por la
    etiqueta de MediaPipe (suavizada y con histeresis). Si las dos manos dicen ser
    la misma, decide la posicion en pantalla (la de la izquierda es la izquierda).
    La imagen llega en espejo, asi que la etiqueta de MediaPipe coincide con tu mano real."""

    def __init__(self):
        self.manos = {}
        self._sig_id = 0

    def actualizar(self, detecciones, t, ancho, alto):
        """detecciones: lista de (norm(21,2), world(21,3)|None, etiqueta|None, score)."""
        escala = np.array([ancho, alto], dtype=np.float64)

        pares = []
        for di, d in enumerate(detecciones):
            for hid, m in self.manos.items():
                dist = float(np.linalg.norm((d[0][0] - m.predicha(t)) * escala))
                pares.append((dist, di, hid))
        pares.sort()
        asignadas, usadas = {}, set()
        for dist, di, hid in pares:
            if dist > C.MATCH_MAX_REL * ancho:
                break
            if di in asignadas or hid in usadas:
                continue
            asignadas[di] = hid
            usadas.add(hid)

        for di in range(len(detecciones)):
            if di in asignadas:
                continue
            if len(self.manos) >= C.NUM_MANOS:
                fantasmas = [h for h, m in self.manos.items() if not m.visible and h not in usadas]
                if not fantasmas:
                    continue
                del self.manos[min(fantasmas, key=lambda h: self.manos[h].visto)]
            hid = self._sig_id
            self._sig_id += 1
            self.manos[hid] = Mano(hid, t)
            asignadas[di] = hid
            usadas.add(hid)

        for di, hid in asignadas.items():
            norm, world, etiqueta, score = detecciones[di]
            self.manos[hid].actualizar(norm, world, etiqueta, score, t, ancho, alto)

        for hid in list(self.manos):
            if hid in usadas:
                continue
            m = self.manos[hid]
            m.visible = False
            if t - m.visto > C.GRACIA_PERDIDA:
                del self.manos[hid]

        self._asignar_roles()

    def _asignar_roles(self):
        for m in self.manos.values():
            if m.rol is None:
                if abs(m.p_izq - 0.5) > 0.2:
                    m.rol = "izq" if m.p_izq > 0.5 else "der"
                else:
                    m.rol = "izq" if m.palma[0] < 0.5 else "der"
            elif m.rol == "izq" and m.p_izq < 0.25:
                m.rol = "der"
            elif m.rol == "der" and m.p_izq > 0.75:
                m.rol = "izq"
        vis = [m for m in self.manos.values() if m.visible]
        if len(vis) == 2 and vis[0].rol == vis[1].rol:
            a, b = sorted(vis, key=lambda m: m.palma[0])
            a.rol, b.rol = "izq", "der"
            a.p_izq, b.p_izq = 0.8, 0.2
