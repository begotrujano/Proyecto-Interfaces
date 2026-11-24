"""Modos educativos interactivos para niños:
  - Comparador de tamaños (2 puños en vista sistema, menú con palma o tecla C)
  - "¿Cuánto pesarías aquí?" con pelota y gravedad real (menú con palma o tecla G + gesto de lanzar)
  - Corte transversal de capas internas (separar 2 manos abiertas sobre el astro, menú o tecla X)
  - Modo Quiz respondiendo al señalar con el índice (menú con palma o tecla K)
  - Menú flotante activable con la palma en pausa (sin necesidad de teclado)
"""

import math
import random

import cv2
import numpy as np

from universo import astros as A
from universo import config as C
from universo import sistema as S
from universo.anillos import FACTOR_EXTERIOR
from universo.manos import a_lienzo

COL_TEXTO = (240, 240, 240)
COL_TENUE = (175, 175, 175)
COL_ACENTO = (40, 195, 255)       # BGR ámbar
COL_CIAN = (255, 210, 90)         # BGR cian
COL_OK = (110, 235, 130)          # BGR verde
COL_ERR = (100, 115, 255)         # BGR rojo suave

# Datos físicos y capas internas explicadas para niños (para los 6 astros disponibles)
DATOS_EDU = {
    "sol": dict(
        radio_km=A.RADIO_SOL_KM,
        gravedad=274.0,
        capas=[
            (1.00, (35, 145, 255), "Superficie brillante (Fotosfera)"),
            (0.92, (45, 185, 255), "Capa de burbujas de calor"),
            (0.68, (90, 225, 255), "Zona de luz y energía"),
            (0.25, (235, 250, 255), "Corazón ardiente (15 millones °C)"),
        ],
    ),
    "mercurio": dict(
        radio_km=2439.7,
        gravedad=3.70,
        capas=[
            (1.00, (145, 145, 150), "Cáscara rocosa con cráteres"),
            (0.92, (100, 110, 135), "Manto de roca"),
            (0.83, (60, 145, 215), "Núcleo gigante de hierro líquido"),
            (0.42, (175, 225, 255), "Corazón de hierro sólido"),
        ],
    ),
    "venus": dict(
        radio_km=6051.8,
        gravedad=8.87,
        capas=[
            (1.00, (125, 185, 225), "Corteza de roca volcánica"),
            (0.94, (55, 105, 205), "Manto súper caliente"),
            (0.52, (75, 185, 255), "Núcleo de hierro y níquel"),
        ],
    ),
    "tierra": dict(
        radio_km=6371.0,
        gravedad=9.81,
        capas=[
            (1.00, (140, 100, 70), "Corteza (donde vivimos)"),
            (0.95, (40, 90, 215), "Manto de roca muy caliente"),
            (0.54, (45, 170, 255), "Núcleo externo líquido"),
            (0.19, (215, 245, 255), "Núcleo interno sólido"),
        ],
    ),
    "luna": dict(
        radio_km=1737.4,
        gravedad=1.62,
        capas=[
            (1.00, (170, 170, 170), "Corteza llena de cráteres"),
            (0.93, (115, 115, 125), "Manto rocoso"),
            (0.22, (80, 160, 235), "Pequeño corazón de hierro"),
        ],
    ),
    "jupiter": dict(
        radio_km=69911.0,
        gravedad=24.79,
        capas=[
            (1.00, (150, 195, 230), "Nubes de amoníaco y gas de hidrógeno"),
            (0.97, (70, 125, 205), "Océano de hidrógeno líquido"),
            (0.80, (60, 160, 245), "Hidrógeno metálico que conduce la electricidad"),
            (0.20, (215, 245, 255), "Núcleo denso de roca y hielo"),
        ],
    ),
    "urano": dict(
        radio_km=25362.0,
        gravedad=8.69,
        capas=[
            (1.00, (230, 225, 170), "Hidrógeno, helio y metano: el metano le da su color cian"),
            (0.80, (200, 160, 90), "Manto de agua, amoníaco y metano helados"),
            (0.20, (215, 245, 255), "Núcleo pequeño de roca y hierro"),
        ],
    ),
    "neptuno": dict(
        radio_km=24622.0,
        gravedad=11.15,
        capas=[
            (1.00, (230, 140, 70), "Hidrógeno, helio y metano: el metano lo vuelve azul"),
            (0.80, (200, 160, 90), "Manto de agua, amoníaco y metano helados"),
            (0.25, (215, 245, 255), "Núcleo de roca y hierro"),
        ],
    ),
    "saturno": dict(
        radio_km=58232.0,
        gravedad=10.44,
        capas=[
            (1.00, (140, 200, 235), "Nubes doradas de amoníaco"),
            (0.95, (80, 140, 205), "Hidrógeno y helio líquidos"),
            (0.55, (60, 160, 245), "Hidrógeno metálico"),
            (0.18, (215, 245, 255), "Núcleo de roca y hielo"),
        ],
    ),
    "marte": dict(
        radio_km=3389.5,
        gravedad=3.71,
        capas=[
            (1.00, (75, 95, 175), "Corteza de polvo rojo"),
            (0.94, (50, 85, 200), "Manto rocoso"),
            (0.52, (75, 175, 245), "Núcleo líquido de hierro"),
        ],
    ),
}

# Pares rápidos para recorrer con el menú o la tecla C/N (además de agarrarlos con 2 manos)
PARES_DEMO = [
    ("tierra", "marte"),
    ("tierra", "luna"),
    ("tierra", "venus"),
    ("tierra", "mercurio"),
    ("tierra", "jupiter"),
    ("jupiter", "saturno"),
    ("sol", "tierra"),
    ("sol", "jupiter"),
]

_ASTROS_COMP = ["sol", "mercurio", "venus", "tierra", "luna", "marte", "jupiter", "saturno", "urano", "neptuno"]
for _i, _a in enumerate(_ASTROS_COMP):                                                 # y el resto de combinaciones
    for _b in _ASTROS_COMP[_i + 1:]:
        if (_a, _b) not in PARES_DEMO and (_b, _a) not in PARES_DEMO:
            PARES_DEMO.append((_a, _b))

# Preguntas del Modo Quiz (todas respondibles con los 5 astros visibles en vista_sistema.py)
PREGUNTAS_QUIZ = [
    ("¿Cuál es el planeta más caluroso del Sistema Solar?", "venus", "¡Sí! Sus nubes guardan el calor como un horno."),
    ("¿Cuál es el planeta más cercano al Sol?", "mercurio", "¡Exacto! Mercurio es el vecino más cercano."),
    ("¿Cuál es conocido como el Planeta Rojo?", "marte", "¡Muy bien! Su polvo rojo tiene hierro oxidado."),
    ("¿Cuál es nuestra estrella gigante que nos da luz?", "sol", "¡Genial! El Sol ilumina a todos los planetas."),
    ("¿En qué planeta vivimos y hay océanos con vida?", "tierra", "¡Correcto! Es nuestro hogar espacial."),
    ("¿Qué planeta tiene el volcán más grande de todos?", "marte", "¡Bravo! Se llama Monte Olimpo."),
    ("¿Cuál es el planeta más chiquito y veloz?", "mercurio", "¡Así es! Da la vuelta al Sol en solo 88 días."),
    ("¿Cuál es el planeta más grande del Sistema Solar?", "jupiter", "¡Correcto! Dentro de Júpiter cabrían más de mil Tierras."),
    ("¿Qué planeta tiene una tormenta gigante llamada la Gran Mancha Roja?", "jupiter",
     "¡Exacto! Es una tormenta más grande que la Tierra y dura siglos."),
    ("¿Qué planeta tiene los vientos más rápidos del Sistema Solar?", "neptuno", "¡Exacto! Neptuno tiene vientos de hasta 2 000 km/h."),
    ("¿Qué planeta gira «acostado», con su eje casi paralelo a su órbita?", "urano", "¡Correcto! Urano rueda de lado alrededor del Sol."),
    ("¿Qué planeta tiene los anillos más grandes y famosos?", "saturno", "¡Muy bien! Son de hielo y roca y brillan con el Sol."),
    ("¿Qué planeta es tan ligero que flotaría en el agua?", "saturno", "¡Así es! Saturno es casi todo gas, menos denso que el agua."),
    ("¿Qué planeta gira al revés que casi todos los demás?", "venus", "¡Increíble! En Venus el Sol sale por el oeste."),
]


def _panel(buf, x0, y0, x1, y1, borde=(95, 85, 75)):
    """Panel oscuro translúcido igual al estilo de las fichas."""
    H, W = buf.shape[:2]
    xa, xb, ya, yb = max(int(x0), 0), min(int(x1), W), max(int(y0), 0), min(int(y1), H)
    if xb <= xa or yb <= ya:
        return
    roi = buf[ya:yb, xa:xb]
    cv2.addWeighted(roi, 0.32, np.full_like(roi, (22, 16, 12)), 0.68, 0, dst=roi)
    cv2.rectangle(buf, (xa, ya), (xb - 1, yb - 1), borde, 1, cv2.LINE_AA)


def _punta_indice(mano_vista, w, h):
    """Punta del dedo índice (landmark 8) en píxeles del lienzo."""
    return a_lienzo(mano_vista.norm[8], w, h)


def _esta_senalando(mano_vista):
    """True si el índice está estirado y no es puño ni OK."""
    if mano_vista.puno or mano_vista.ok:
        return False
    p = mano_vista.norm
    muneca = p[0]
    base_ind = max(float(np.linalg.norm(p[5] - muneca)), 1e-6)
    ext_ind = float(np.linalg.norm(p[8] - muneca)) / base_ind
    return ext_ind > 1.45


# ----------------------------------------------------------------------
# 1. Comparador de tamaños (2 puños en vista sistema, menú o tecla C)
# ----------------------------------------------------------------------
# Datos extra que solo usa el comparador: rotación (horas, negativa = al revés) e inclinación del eje (grados)
DATOS_COMP = {
    "sol": dict(dia_h=609.1, incl=7.25),
    "mercurio": dict(dia_h=1407.6, incl=0.03),
    "venus": dict(dia_h=-5832.5, incl=177.36),
    "tierra": dict(dia_h=23.93, incl=23.44),
    "luna": dict(dia_h=655.7, incl=6.68),
    "marte": dict(dia_h=24.62, incl=25.19),
    "jupiter": dict(dia_h=9.925, incl=3.13),
    "saturno": dict(dia_h=10.656, incl=26.73),
    "urano": dict(dia_h=-17.24, incl=97.77),
    "neptuno": dict(dia_h=16.11, incl=28.32),
}


def _ease_salida(x):
    """Entrada con un pequeño rebote (0 → 1)."""
    x = min(max(x, 0.0), 1.0)
    c1 = 1.1
    return 1.0 + (c1 + 1.0) * (x - 1.0) ** 3 + c1 * (x - 1.0) ** 2


def _ease_cubica(x):
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def _con_articulo(nombre, mayuscula=False):
    a = S.ARTICULOS.get(nombre, "")
    txt = f"{a} {S.NOMBRES[nombre]}".strip()
    return txt[0].upper() + txt[1:] if mayuscula else txt


def _miles(v, dec=0):
    return f"{v:,.{dec}f}".replace(",", " ")


def _duracion_dia(dia_h):
    h = abs(dia_h)
    txt = f"{h:.1f} horas" if h < 48 else f"{h / 24.0:.0f} días"
    return txt + (" (al revés)" if dia_h < 0 else "")


class ComparadorTamanos:
    """Dos astros a escala real, girando con la luz quieta en pantalla.
    Dinámico: entran creciendo, giran a su ritmo (comprimido), con su inclinación de eje y con los datos de cada uno;
    con el puño derecho se arrastra para girarlos y con inercia. Se recorren los 15 pares con N / el menú."""

    def __init__(self):
        self.activo = False
        self.par = None                   # ("tierra", "marte")
        self.agarrados = {}               # rol ("izq"/"der") -> nombre_astro
        self._idx_demo = 0
        self._t_inicio = 0.0
        self._par_visto = None            # par que ya se empezó a dibujar (reinicia la animación de entrada)
        self._giros = {}                  # nombre -> giro acumulado (rad)
        self._t_prev = None
        self._arr_x = None                # x del puño derecho mientras arrastra
        self._empuje = 0.0                # giro (rad) pedido por el arrastre, aún sin aplicar
        self._vel_inercia = 0.0           # rad/s que queda al soltar
        self.eligiendo = None             # 0 / 1 = se esta eligiendo el astro de esa posicion (menu); None = nada

    def elegir_astro(self, posicion, nombre):
        """Cambia el astro de la posicion 0 (izq.) o 1 (der.) del par; no permite repetir el mismo astro."""
        if not self.par or nombre not in _ASTROS_COMP:
            return False
        par = list(self.par)
        if par[1 - posicion] == nombre:                    # ya esta en la otra posicion: se intercambian
            par[1 - posicion] = par[posicion]
        par[posicion] = nombre
        self.par = (par[0], par[1])
        self.eligiendo = None
        return True

    def alternar_demo(self, foco="tierra"):
        if not self.activo:
            self.par = PARES_DEMO[self._idx_demo % len(PARES_DEMO)]
            self._idx_demo += 1
            self.activo = True
        else:
            self.cerrar()

    def abrir(self, par, t):
        """Abre el comparador con ese par (ordenes por voz); reinicia la animacion de entrada."""
        self.par = (par[0], par[1])
        self.eligiendo = None
        self.activo = True
        self._t_inicio = t

    def siguiente_par(self):
        self.eligiendo = None
        self.par = PARES_DEMO[self._idx_demo % len(PARES_DEMO)]
        self._idx_demo += 1
        self.activo = True

    def cerrar(self):
        self.activo = False
        self.par = None
        self.eligiendo = None
        self._par_visto = None
        self._t_prev = None
        self._arr_x = None
        self._empuje = self._vel_inercia = 0.0
        self.agarrados.clear()
        self._liberar = True              # dibujar() suelta las texturas sin luz en el hilo de dibujo

    def actualizar(self, est, objetivos, modo, t=0.0):
        """En vista sistema: cierra puño sobre un planeta con cada mano y júntalas.
        Con el comparador abierto: puño derecho arrastrando = girar; abre ambas manos y sepáralas = cerrar."""
        if modo != "sistema" and not self.activo:
            self.agarrados.clear()
            return
        vis = [m for m in est.manos if m.visible]
        if self.activo:
            der = next((m for m in vis if m.rol == "der" and m.puno), None)
            if der is not None:
                if self._arr_x is not None:
                    self._empuje += (der.x - self._arr_x) * C.COMPARADOR_GANANCIA_GIRO
                self._arr_x = der.x
            else:
                self._arr_x = None
            abiertas = [m for m in vis if not m.puno and not m.ok]
            if len(abiertas) == 2 and (t - self._t_inicio) > 1.0:
                dist = abs(float(abiertas[0].norm[9, 0] - abiertas[1].norm[9, 0]))
                if dist > 0.58:
                    self.cerrar()
            return

        for m in vis:
            if m.puno:
                if m.rol not in self.agarrados:
                    for nombre, (ox, oy, r) in objetivos.items():
                        if nombre in DATOS_EDU and math.hypot(m.x - ox, m.y - oy) <= max(r, 28.0) + 18.0:
                            self.agarrados[m.rol] = nombre
                            break
            else:
                self.agarrados.pop(m.rol, None)

        if "izq" in self.agarrados and "der" in self.agarrados:
            a, b = self.agarrados["izq"], self.agarrados["der"]
            if a != b:
                m_izq = next((m for m in vis if m.rol == "izq"), None)
                m_der = next((m for m in vis if m.rol == "der"), None)
                if m_izq and m_der:
                    dist = float(np.linalg.norm(m_izq.norm[9] - m_der.norm[9]))
                    if dist < C.COMPARADOR_DIST_JUNTAR:
                        self.par = (a, b)
                        self.activo = True
                        self._t_inicio = t
                        self.agarrados.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _dibujantes(escena, par):
        return {n: getattr(escena, n, None) for n in par}

    def _pedir(self, dibs):
        """Pide la textura grande y las texturas sin luz de cada astro (en otros hilos)."""
        for d in dibs.values():
            if d is None:
                continue
            if hasattr(d, "preparar"):
                d.preparar()
            d.preparar_planas()

    @staticmethod
    def _listos(dibs):
        return all(d is not None and d.listo_comparar() for d in dibs.values())

    def _avanzar_giros(self, par, t):
        """Giro de cada astro: auto (a su ritmo) o arrastrado con el puño, con inercia al soltar."""
        dt = 0.0 if self._t_prev is None else min(max(t - self._t_prev, 0.0), 0.1)
        self._t_prev = t
        arrastrando = self._arr_x is not None
        empuje, self._empuje = self._empuje, 0.0
        if arrastrando and dt > 1e-4:
            self._vel_inercia = 0.5 * self._vel_inercia + 0.5 * empuje / dt
        elif not arrastrando:
            self._vel_inercia *= math.exp(-dt / max(C.COMPARADOR_INERCIA_S, 1e-3))
        for n in par:
            dia_h = DATOS_COMP[n]["dia_h"]
            dias = max(abs(dia_h) / 24.0, 1.0)
            vuelta = C.COMPARADOR_VUELTA_S * dias ** C.COMPARADOR_COMPRESION        # s por vuelta en pantalla
            auto = 0.0 if arrastrando else math.copysign(2 * math.pi / vuelta, dia_h)
            if n not in self._giros:
                self._giros[n] = math.radians(99.0) if n == "tierra" else 0.0     # la Tierra arranca en México
            self._giros[n] += (auto + self._vel_inercia) * dt + empuje

    def _dibujar_astro(self, escena, dib, nombre, x, y, r):
        d = DATOS_COMP[nombre]
        incl = min(d["incl"], 180.0 - d["incl"])                # Venus: 177° = 3° vista desde arriba (gira al revés)
        rodar = math.radians(incl)                              # el polo norte se inclina hacia la derecha
        giro = self._giros[nombre]
        luz = np.array(self._luz_astro.get(nombre, C.COMPARADOR_LUZ), np.float64)
        luz /= np.linalg.norm(luz)
        if nombre == "sol":
            dib.dibujar(escena.buf, x, y, r, giro, C.COMPARADOR_INCLINACION, luz, np.array([1.0, 0.0, 0.0]), rodar)
        else:
            dib.dibujar(escena.buf, x, y, r, giro, C.COMPARADOR_INCLINACION_ASTRO.get(nombre, C.COMPARADOR_INCLINACION), luz,
                        None, rodar, luz_vista=True)

    def dibujar(self, escena, jd, t):
        if getattr(self, "_liberar", False):                    # se cerró: soltar la memoria de las texturas sin luz
            self._liberar = False
            for n in DATOS_COMP:
                d = getattr(escena, n, None)
                if d is not None and hasattr(d, "soltar_planas"):
                    d.soltar_planas()
        if not self.activo or not self.par:
            return
        buf, T = escena.buf, escena.texto
        w, h = escena.w, escena.h
        a, b = self.par
        dibs = self._dibujantes(escena, self.par)
        if self._par_visto != self.par:                          # par nuevo: pedir recursos y reiniciar la entrada
            self._par_visto = self.par
            self._pedir(dibs)
        if not self._listos(dibs):                               # sin textura grande se vería borroso y saltaría
            self._t_inicio = t
            self._t_prev = t
            cv2.addWeighted(buf, 0.18, np.full_like(buf, (14, 10, 8)), 0.82, 0, dst=buf)
            T.dibujar(buf, "Preparando los astros…", w * 0.5, h * 0.5, 20, COL_TENUE, False, "c")
            return
        self._avanzar_giros(self.par, t)

        anim = (t - self._t_inicio) / max(C.COMPARADOR_ENTRADA, 1e-3)
        crece = max(_ease_salida(anim), 0.02)
        fondo = 0.985 * _ease_cubica(anim * 2.0)                  # el fondo se oscurece mientras entran
        cv2.addWeighted(buf, 1.0 - fondo, np.full_like(buf, (14, 10, 8)), fondo, 0, dst=buf)

        ra, rb = DATOS_EDU[a]["radio_km"], DATOS_EDU[b]["radio_km"]
        mayor, menor = max(ra, rb), min(ra, rb)
        ratio = mayor / menor
        es_sol = "sol" in (a, b)
        # espacio vertical: bajo el encabezado (arriba) y sobre la fila de botones del menu (abajo, ver MenuGestos)
        y_top = C.COMPARADOR_Y_ARRIBA
        r_tope = max((h - C.COMPARADOR_Y_RESERVA) / 2.0, 40.0)
        self._luz_astro = {}

        # radios a escala real; el astro que se verría como un punto se amplía (y se avisa)
        r_max = min(w * 0.22, h * 0.30, r_tope) if es_sol else min(w * 0.20, h * 0.26, r_tope)
        # Saturno con sus anillos inclinados ocupa mas alto que su disco: se achica todo (a la misma escala)
        k_alto = {n: (C.ANILLOS_ALTO_COMPARADOR if n == "saturno" else 1.0) for n in (a, b)}
        r_max = min(r_max, r_tope / max(k_alto[a] * ra, k_alto[b] * rb) * mayor)
        r_real = {a: r_max * ra / mayor, b: r_max * rb / mayor}
        alto_v = {n: r_real[n] * k_alto[n] for n in (a, b)}       # media altura con anillos (planeta: su radio)
        cy = y_top + max(alto_v.values())
        chico = a if ra < rb else b
        ampliacion = max(C.COMPARADOR_MIN_PX_SOL / max(r_real[chico], 1e-6), 1.0)
        r_dib = dict(r_real)
        r_dib[chico] = r_real[chico] * ampliacion
        if es_sol:                                                # el Sol es colosal: a la izquierda
            xs = {"sol": w * 0.27, chico: w * 0.72}
            # el Sol queda a su izquierda (Saturno, algo desde arriba: si no, el Sol queda en el plano de los anillos)
            self._luz_astro[chico] = (-0.80, -0.55, 0.40) if chico == "saturno" else (-0.90, -0.05, 0.45)
        else:
            hueco = max(w * 0.07, 50.0)
            ext = {n: r_dib[n] * (FACTOR_EXTERIOR if n == "saturno" else 1.0) for n in (a, b)}   # con sus anillos
            dist = ext[a] + ext[b] + hueco                        # entre CENTROS: extensiones + hueco entre bordes
            xs = {a: w * 0.5 - dist * 0.5, b: w * 0.5 + dist * 0.5}
        rs = {n: r_dib[n] * crece for n in (a, b)}

        # de más grande a más chico: el chico nunca queda tapado por el resplandor del grande
        for n in sorted((a, b), key=lambda n: -rs[n]):
            self._dibujar_astro(escena, dibs[n], n, xs[n], cy, rs[n])

        # --- rótulos: aparecen cuando los astros ya casi llegaron
        op = min(max((anim - 0.55) / 0.45, 0.0), 1.0)
        y_nom = cy + max(alto_v.values()) + 44.0
        for n in (a, b):
            rd = r_dib[n]
            yl = cy + max(rd, alto_v[n] if n != chico or ampliacion <= 1.0 else rd * k_alto[n]) + 14.0   # línea de medida: diámetro
            if op > 0.0 and rd > 3.0:
                x0, x1 = xs[n] - rd, xs[n] + rd
                col = tuple(int(c * (0.35 + 0.65 * op)) for c in COL_TENUE)
                cv2.line(buf, (int(x0), int(yl)), (int(x1), int(yl)), col, 1, cv2.LINE_AA)
                for xx in (x0, x1):
                    cv2.line(buf, (int(xx), int(yl - 5)), (int(xx), int(yl + 5)), col, 1, cv2.LINE_AA)
            if op > 0.0:
                km = f"{_miles(DATOS_EDU[n]['radio_km'] * 2)} km"
                if n == chico and ampliacion > 1.5:
                    T.dibujar(buf, f"{km}  ·  ampliado ×{ampliacion:.0f}", xs[n], yl + 8, 13, COL_ACENTO, False, "c",
                              opacidad=op)
                else:
                    T.dibujar(buf, km, xs[n], yl + 8, 13, COL_TENUE, False, "c", opacidad=op)
                T.dibujar(buf, S.NOMBRES[n], xs[n], y_nom, 21, COL_TEXTO, True, "c", opacidad=op)
                g = DATOS_EDU[n]["gravedad"]
                T.dibujar(buf, f"Gravedad: {g:.1f} m/s²", xs[n], y_nom + 30, 14, COL_TENUE, False, "c", opacidad=op)
                T.dibujar(buf, f"Un día: {_duracion_dia(DATOS_COMP[n]['dia_h'])}", xs[n], y_nom + 50, 14, COL_TENUE,
                          False, "c", opacidad=op)
        if op > 0.0 and ampliacion > 1.5:                         # su tamaño real, de referencia (hacia afuera)
            otro = b if chico == a else a
            lado = 1.0 if xs[chico] >= xs[otro] else -1.0
            xr = xs[chico] + lado * (r_dib[chico] + 70.0)
            cv2.circle(buf, (int(xr), int(cy)), max(int(round(r_real[chico])), 1), COL_TEXTO, -1, cv2.LINE_AA)
            T.dibujar(buf, "tamaño real", xr, cy + 16, 13, COL_TENUE, False, "c", opacidad=op)

        # --- encabezado, con la razón contando hacia arriba
        valor = 1.0 + (ratio - 1.0) * _ease_cubica((t - self._t_inicio) / 1.4)
        vol = ratio ** 3
        mitad = min(340.0, w * 0.5 - 270.0)                      # sin tapar el estado (izq.) ni la camara (der.)
        _panel(buf, w * 0.5 - mitad, 18, w * 0.5 + mitad, 108, COL_CIAN)
        T.dibujar(buf, f"COMPARADOR A ESCALA REAL: {S.NOMBRES[a]} vs {S.NOMBRES[b]}", w * 0.5, 28, 19, COL_CIAN, True, "c")
        nom_may = a if ra >= rb else b
        nom_men = b if ra >= rb else a
        if ratio < 1.15:
            frase = f"¡{S.NOMBRES[a]} y {S.NOMBRES[b]} casi miden lo mismo! (×{valor:.2f})"
        else:
            frase = f"¡{_con_articulo(nom_may, True)} es {valor:.1f} veces más grande que {_con_articulo(nom_men)}!"
        T.dibujar(buf, frase, w * 0.5, 60, 17, COL_TEXTO, False, "c")
        vol_v = valor ** 3                                       # el volumen sube junto con la razón
        T.dibujar(buf, f"{_con_articulo(nom_may, True)} tiene ≈ {_miles(vol_v, 0 if vol >= 10 else 1)} veces "
                       f"el volumen de {_con_articulo(nom_men)}",
                  w * 0.5, 84, 13, COL_TENUE, False, "c")
        T.dibujar(buf, "Puño derecho: arrastra para girarlos  ·  Separa ambas manos abiertas para cerrar  ·  Teclas C y N",
                  w * 0.5, h - 26, 14, COL_TENUE, False, "c")


# ----------------------------------------------------------------------
# 2. "¿Cuánto pesarías aquí?" (Gravedad + pelota lanzada con la mano)
# ----------------------------------------------------------------------
class SimuladorGravedad:
    def __init__(self):
        self.activo = False
        self.peso_tierra = C.PESO_NINO_KG
        self.pelota_activa = False
        self.x = 0.0
        self.y = 0.0
        self.vy = 0.0
        self.suelo_y = 580.0
        self.px_por_metro = 75.0
        self._prev_y = {}
        self._ultimo_lanzamiento = -10.0

    def alternar(self, w, h, t):
        self.activo = not self.activo
        if self.activo:
            self.lanzar(w * 0.75, h * 0.52, h - 210, t)

    def cambiar_peso(self, delta):
        self.peso_tierra = float(np.clip(self.peso_tierra + delta, 10.0, 120.0))

    def lanzar(self, x, y, suelo_y, t, impulso_m_s=-6.8):
        self.pelota_activa = True
        self.x = float(x)
        self.y = float(y)
        self.suelo_y = float(suelo_y)
        self.vy = impulso_m_s * self.px_por_metro
        self._ultimo_lanzamiento = t

    def actualizar(self, est, foco, dt, t, w, h):
        if not self.activo or foco not in DATOS_EDU:
            self._prev_y.clear()
            return
        g = DATOS_EDU[foco]["gravedad"]

        # Detectar gesto de lanzar (movimiento rápido hacia arriba de cualquier mano)
        for m in est.manos:
            if not m.visible:
                continue
            yn = float(m.norm[9, 1])
            prev = self._prev_y.get(m.id)
            self._prev_y[m.id] = yn
            if prev is not None and dt > 1e-4:
                vy_norm = (prev - yn) / dt    # positivo = subiendo
                if vy_norm > C.LANZAR_V_ON and (t - self._ultimo_lanzamiento) > 0.65:
                    self.lanzar(np.clip(m.x, 120, w - 120), min(m.y, h - 230), h - 210, t)

        # Física de caída y rebote
        if self.pelota_activa:
            g_vis = min(g, 45.0)
            self.vy += g_vis * self.px_por_metro * dt
            self.y += self.vy * dt
            if self.y < 90.0:
                self.y = 90.0
                self.vy = max(self.vy, 0.0)
            if self.y >= self.suelo_y:
                self.y = self.suelo_y
                self.vy = -self.vy * 0.65
                if abs(self.vy) < 35.0:
                    self.vy = 0.0

    def dibujar(self, escena, foco):
        if not self.activo or foco not in DATOS_EDU:
            return
        buf, T = escena.buf, escena.texto
        w, h = escena.w, escena.h
        g = DATOS_EDU[foco]["gravedad"]
        factor = g / 9.81
        peso_aqui = self.peso_tierra * factor
        salto_aqui = C.SALTO_TIERRA_M / max(factor, 0.01)

        # Panel a la derecha (debajo de la miniatura de cámara)
        px0, py0, px1, py1 = w - 360, 240, w - 16, 425
        _panel(buf, px0, py0, px1, py1, COL_ACENTO)
        T.dibujar(buf, "¿CUÁNTO PESARÍAS AQUÍ?", px0 + 14, py0 + 12, 16, COL_ACENTO, True)
        T.dibujar(buf, f"En la Tierra pesas: {self.peso_tierra:.0f} kg",
                  px0 + 14, py0 + 40, 13, COL_TENUE)
        T.dibujar(buf, f"En {S.NOMBRES[foco]} pesarías: {peso_aqui:.1f} kg",
                  px0 + 14, py0 + 68, 17, COL_TEXTO, True)
        T.dibujar(buf, f"Tu salto sería de: {salto_aqui:.2f} m",
                  px0 + 14, py0 + 98, 16, COL_OK, True)
        T.dibujar(buf, f"Gravedad: {g:.2f} m/s² ({factor:.2f}× Tierra)",
                  px0 + 14, py0 + 126, 13, COL_TENUE)
        T.dibujar(buf, "¡Mueve tu mano hacia arriba para lanzar!",
                  px0 + 14, py0 + 154, 13, COL_CIAN)

        # Pelota y suelo de referencia
        if self.pelota_activa:
            sx, sy = int(self.x), int(self.suelo_y) + 18
            cv2.line(buf, (sx - 55, sy), (sx + 55, sy), (180, 180, 180), 2, cv2.LINE_AA)
            cv2.circle(buf, (int(self.x), int(self.y)), 18, COL_ACENTO, -1, cv2.LINE_AA)
            cv2.circle(buf, (int(self.x), int(self.y)), 18, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# 3. Corte transversal (separar las 2 manos abiertas sobre el planeta o tecla X)
# ----------------------------------------------------------------------
class CorteTransversal:
    def __init__(self):
        self.apertura = 0.0           # 0.0 cerrado -> 1.0 totalmente abierto
        self.manual = False

    def alternar_manual(self):
        self.manual = not self.manual

    def actualizar(self, est, modo, viaje):
        if modo != "astro" or viaje is not None:
            self.apertura = 0.0
            return
        objetivo = 1.0 if self.manual else 0.0

        # Gesto: 2 manos visibles y abiertas (ni puño ni OK) separándose horizontalmente
        if not self.manual:
            abiertas = [m for m in est.manos if m.visible and not m.puno and not m.ok]
            if len(abiertas) == 2:
                dist = abs(float(abiertas[0].norm[9, 0] - abiertas[1].norm[9, 0]))
                t_ap = (dist - C.CORTE_DIST_MIN) / max(C.CORTE_DIST_MAX - C.CORTE_DIST_MIN, 1e-3)
                objetivo = float(np.clip(t_ap, 0.0, 1.0))

        self.apertura = 0.82 * self.apertura + 0.18 * objetivo
        if self.apertura < 0.02 and objetivo == 0.0:
            self.apertura = 0.0

    def dibujar(self, escena, foco):
        if self.apertura <= 0.05 or foco not in DATOS_EDU:
            return
        obj = escena.astros.get("objetivos", {}).get(foco)
        if obj is None:
            return
        cx, cy, radio = obj
        if radio < 25:
            return

        buf, T = escena.buf, escena.texto
        cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(radio))
        capas = DATOS_EDU[foco]["capas"]
        ang_fin = int(-90 + 180 * self.apertura)      # se abre de arriba hacia abajo (cuña derecha)

        for frac, color_bgr, _ in capas:
            rc = max(6, int(round(r_i * frac)))
            cv2.ellipse(buf, (cx_i, cy_i), (rc, rc), 0, -90, ang_fin, color_bgr, -1, cv2.LINE_AA)
            cv2.ellipse(buf, (cx_i, cy_i), (rc, rc), 0, -90, ang_fin, (240, 240, 240), 1, cv2.LINE_AA)

        # Etiquetas de cada capa cuando ya está suficientemente abierto
        if self.apertura > 0.50:
            tx = min(cx_i + r_i + 34, escena.w - 290)
            y_base = max(cy_i - 55, 110)
            for idx, (frac, color_bgr, nombre) in enumerate(capas):
                ang = math.radians(-55 + idx * (110 / max(len(capas) - 1, 1)))
                r_mid = r_i * (frac * 0.82)
                px = int(cx_i + r_mid * math.cos(ang))
                py = int(cy_i + r_mid * math.sin(ang))
                ty = y_base + idx * 34
                cv2.line(buf, (px, py), (tx - 10, ty + 8), (225, 225, 225), 1, cv2.LINE_AA)
                cv2.circle(buf, (px, py), 4, color_bgr, -1, cv2.LINE_AA)
                T.dibujar(buf, nombre, tx, ty, 14, COL_TEXTO, idx == len(capas) - 1)


# ----------------------------------------------------------------------
# 4. Modo Quiz (responder señalando con el dedo índice o con clic)
# ----------------------------------------------------------------------
class ModoQuiz:
    def __init__(self):
        self.activo = False
        self.preguntas = []
        self.indice = 0
        self.puntaje = 0
        self.candidato = None
        self.inicio_hover = 0.0
        self.feedback = ""
        self.es_correcta = False
        self.feedback_hasta = 0.0
        self.puntero = None
        self.progreso = 0.0

    def iniciar(self):
        self.activo = True
        self.preguntas = random.sample(PREGUNTAS_QUIZ, len(PREGUNTAS_QUIZ))
        self.indice = 0
        self.puntaje = 0
        self.candidato = None
        self.feedback = ""
        self.feedback_hasta = 0.0
        self.progreso = 0.0

    def cerrar(self):
        self.activo = False
        self.candidato = None
        self.progreso = 0.0

    @property
    def actual(self):
        if self.indice >= len(self.preguntas):
            return None
        return self.preguntas[self.indice]

    def responder(self, astro_id, t):
        if not self.activo or self.actual is None or t < self.feedback_hasta:
            return
        _, correcta, explicacion = self.actual
        if astro_id == correcta:
            self.puntaje += 1
            self.es_correcta = True
            self.feedback = explicacion
        else:
            self.es_correcta = False
            self.feedback = f"¡Casi! La respuesta era {S.NOMBRES[correcta]}."
        self.indice += 1
        self.candidato = None
        self.progreso = 0.0
        self.feedback_hasta = t + C.QUIZ_FEEDBACK

    def actualizar(self, est, objetivos, w, h, t):
        if not self.activo:
            return
        self.puntero = None
        self.progreso = 0.0
        if t < self.feedback_hasta or self.actual is None:
            return

        # Buscar una mano señalando con el índice
        for m in est.manos:
            if m.visible and _esta_senalando(m):
                self.puntero = _punta_indice(m, w, h)
                break

        tocado = None
        if self.puntero is not None:
            dx, dy = self.puntero
            for nombre, (ox, oy, r) in objetivos.items():
                if math.hypot(dx - ox, dy - oy) <= max(r, 34.0) + 10.0:
                    tocado = nombre
                    break

        if tocado is None:
            self.candidato = None
            return
        if tocado != self.candidato:
            self.candidato = tocado
            self.inicio_hover = t
            return

        self.progreso = min(1.0, (t - self.inicio_hover) / C.QUIZ_SOSTENER)
        if self.progreso >= 1.0:
            self.responder(tocado, t)

    def dibujar(self, escena, t):
        if not self.activo:
            return
        buf, T = escena.buf, escena.texto
        w = escena.w

        mitad = min(390.0, w * 0.5 - 270.0)                      # sin tapar el estado (izq.) ni la camara (der.)
        _panel(buf, w * 0.5 - mitad, 14, w * 0.5 + mitad, 96, COL_OK)
        if t < self.feedback_hasta:
            col = COL_OK if self.es_correcta else COL_ERR
            T.dibujar(buf, self.feedback, w * 0.5, 42, 19, col, True, "c")
        elif self.actual is None:
            T.dibujar(buf, f"¡Reto completado! Aciertos: {self.puntaje} de {len(self.preguntas)}",
                      w * 0.5, 30, 21, COL_OK, True, "c")
            T.dibujar(buf, "Deja la palma sobre Quiz (o presiona K) para salir",
                      w * 0.5, 62, 14, COL_TENUE, False, "c")
        else:
            pregunta = self.actual[0]
            T.dibujar(buf, f"QUIZ ESPACIAL ({self.indice + 1}/{len(self.preguntas)})  ·  Puntos: {self.puntaje}",
                      w * 0.5, 22, 14, COL_OK, True, "c")
            T.dibujar(buf, pregunta, w * 0.5, 46, 19, COL_TEXTO, True, "c")
            T.dibujar(buf, "Señala un planeta con tu dedo índice (o haz clic)",
                      w * 0.5, 72, 13, COL_TENUE, False, "c")

        # Puntero y anillo de confirmación al señalar
        if self.puntero is not None:
            px, py = int(round(self.puntero[0])), int(round(self.puntero[1]))
            cv2.circle(buf, (px, py), 8, COL_OK, -1, cv2.LINE_AA)
            if self.progreso > 0.0:
                cv2.ellipse(buf, (px, py), (28, 28), -90, 0, int(360 * self.progreso), COL_OK, 4, cv2.LINE_AA)


# ----------------------------------------------------------------------
# 5. Menú flotante activable con la palma (sin tocar el teclado)
# ----------------------------------------------------------------------
class MenuGestos:
    """Botones flotantes en la esquina inferior derecha.
    En pausa, deja la palma encima 0.85 s (o haz clic) para activar/desactivar cada modo."""

    def __init__(self, dwell_s=0.85):
        self.dwell_s = dwell_s
        self.botones = []             # [(id, texto, activo, x0, y0, x1, y1)]
        self.hover_id = None
        self._inicio = 0.0
        self.progreso = 0.0
        self._cooldown_hasta = 0.0

    def _construir(self, w, h, modo, comparador, gravedad, corte, quiz):
        items = [
            ("sistema", "Planeta" if modo == "sistema" else "Sistema", modo == "sistema"),
            ("comparar", "Comparar", comparador.activo),
            ("quiz", "Quiz", quiz.activo),
        ]
        if modo == "astro" and not comparador.activo:
            items.insert(2, ("gravedad", "Gravedad", gravedad.activo))
            items.insert(3, ("corte", "Capas", corte.manual))

        bw, bh, gap = 112, 38, 10
        total_w = len(items) * bw + (len(items) - 1) * gap
        x_ini = w - total_w - 20
        y0 = h - 126
        botones = []
        for i, (bid, txt, act) in enumerate(items):
            x0 = x_ini + i * (bw + gap)
            botones.append((bid, txt, act, x0, y0, x0 + bw, y0 + bh))

        extras = []
        if comparador.activo:
            a, b = comparador.par or ("", "")
            extras = [("elegir_0", S.NOMBRES.get(a, "Astro 1"), comparador.eligiendo == 0),
                      ("elegir_1", S.NOMBRES.get(b, "Astro 2"), comparador.eligiendo == 1),
                      ("otro_par", "Otro par", False), ("comparar", "Cerrar", True)]
        elif gravedad.activo and modo == "astro":
            extras = [("peso_menos", "-5 kg", False), ("peso_mas", "+5 kg", False)]

        if extras:
            ew = 104
            ex_ini = w - (len(extras) * ew + (len(extras) - 1) * gap) - 20
            ey0 = y0 - bh - 10
            for i, (bid, txt, act) in enumerate(extras):
                x0 = ex_ini + i * (ew + gap)
                botones.append((bid, txt, act, x0, ey0, x0 + ew, ey0 + bh))

        if comparador.activo and comparador.eligiendo is not None and comparador.par:
            # selector: los 10 astros en una fila sobre los botones (el actual de esa posicion queda marcado)
            actual = comparador.par[comparador.eligiendo]
            pw, ph, pgap = C.COMPARADOR_SELECTOR_ANCHO, C.COMPARADOR_SELECTOR_ALTO, C.COMPARADOR_SELECTOR_GAP
            n = len(_ASTROS_COMP)
            px_ini = w - (n * pw + (n - 1) * pgap) - 20
            py0 = y0 - 2 * bh - 20 - (ph - bh)
            for i, nombre in enumerate(_ASTROS_COMP):
                x0 = px_ini + i * (pw + pgap)
                botones.append(("astro_" + nombre, S.NOMBRES[nombre], nombre == actual, x0, py0, x0 + pw, py0 + ph))

        self.botones = botones

    def boton_en(self, x, y):
        for bid, _, _, x0, y0, x1, y1 in self.botones:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return bid
        return None

    def actualizar(self, est, modo, comparador, gravedad, corte, quiz, viaje, w, h, t):
        """Devuelve el id de la acción activada o None."""
        if viaje is not None:
            self.botones = []
            self.hover_id = None
            self.progreso = 0.0
            return None

        self._construir(w, h, modo, comparador, gravedad, corte, quiz)

        # Solo responde a la mano cuando el control está en pausa (evita toques al girar/zoom)
        if est.activo or t < self._cooldown_hasta:
            self.hover_id = None
            self.progreso = 0.0
            return None

        tocado = None
        for m in est.manos:
            if not m.visible or m.puno or m.ok:
                continue
            bid = self.boton_en(m.x, m.y)
            if bid is not None:
                tocado = bid
                break

        if tocado is None:
            self.hover_id = None
            self.progreso = 0.0
            return None

        if tocado != self.hover_id:
            self.hover_id = tocado
            self._inicio = t
            self.progreso = 0.0
            return None

        self.progreso = min(1.0, (t - self._inicio) / self.dwell_s)
        if self.progreso >= 1.0:
            self.hover_id = None
            self.progreso = 0.0
            self._cooldown_hasta = t + 0.75
            return tocado
        return None

    def dibujar(self, escena):
        if not self.botones:
            return
        buf, T = escena.buf, escena.texto
        for bid, txt, activo, x0, y0, x1, y1 in self.botones:
            borde = COL_OK if activo else (COL_CIAN if bid == self.hover_id else (105, 95, 85))
            _panel(buf, x0, y0, x1, y1, borde)
            if bid == self.hover_id and self.progreso > 0.0:
                ancho_p = int((x1 - x0 - 4) * self.progreso)
                cv2.rectangle(buf, (int(x0) + 2, int(y1) - 6),
                              (int(x0) + 2 + ancho_p, int(y1) - 2), COL_OK, -1, cv2.LINE_AA)
            col_txt = COL_OK if activo else COL_TEXTO
            T.dibujar(buf, txt, (x0 + x1) / 2, y0 + 10, 14, col_txt, activo or bid == self.hover_id, "c")