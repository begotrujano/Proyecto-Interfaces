"""Control de la Tierra con gestos.

    Pausa  --(OK sostenido 0.3 s, cualquier mano)-->  Activo
    Activo --(OK otra vez  /  sin manos varios segundos)-->  Pausa

    Activo:
      Mano IZQUIERDA  -> zoom:  cierra el puno y sube (acercar) o baja (alejar).
                                Abre la mano para soltar y reacomodarla.
      Mano DERECHA    -> giro, como un globo terraqueo:
                         - swipe con la mano abierta: le da vuelo
                             horizontal -> gira sobre su eje (cambia la longitud)
                             vertical   -> se inclina (cambia la latitud, hasta los polos)
                         - puno: la agarras (se detiene) y la arrastras en cualquier
                           direccion; al soltar conserva el impulso
      El regreso de la mano despues de un swipe se ignora, como al girar un globo real.
      Cada swipe va en un solo eje (el dominante); los diagonales no cuentan.

Hilos: el hilo de deteccion llama actualizar(); el de dibujo llama avanzar() e
instantanea(). Todo pasa por un candado.
"""

import math
import threading
from collections import namedtuple

import numpy as np

from universo import config as C
from universo.manos import a_lienzo

ManoVista = namedtuple(
    "ManoVista",
    "id rol x y norm visible puno ok progreso_ok accion ratio_puno ratio_ok estirados")
Estado = namedtuple(
    "Estado", "activo log_escala giro omega manos avisos zoom_agarrado giro_agarrado swipe invertido "
              "inclinacion omega_v", defaults=(0.0, 0.0))


def _signo(x):
    return 1 if x > 0 else -1 if x < 0 else 0


class Control:
    def __init__(self, ancho=C.ANCHO, alto=C.ALTO):
        self.W, self.H = ancho, alto
        self.radio_base = C.RADIO_BASE_REL * alto
        self.lock = threading.Lock()
        self.invertido = C.INVERTIR_MANOS
        self.reiniciar()
        self.activo = False
        self.avisos = []                    # [(texto, t)]
        self._ok = {}                       # id -> {"desde", "ult", "bloq"}
        self._t_det = None
        self._t_render = None
        self._sin_manos_desde = None
        self._vistas = ()

    # ------------------------------------------------------------------
    def reiniciar(self):
        self.log_escala = 0.0
        self.giro = math.radians(-C.LONGITUD_INICIAL)   # giro positivo = la superficie va a la derecha
        self.omega = 0.0
        self.inclinacion = 0.0             # rad; positivo = la superficie de enfrente sube (se ve el sur)
        self.omega_v = 0.0                 # rad/s de la inclinacion
        self.zoom_mano = None
        self._y_prev = None
        self.giro_mano = None
        self._x_prev = None
        self._yg_prev = None
        self._vel_arrastre = 0.0
        self._vel_arrastre_v = 0.0
        self.swipe = "reposo"
        self._swipe_dir = 0                # +-1 horizontal, +-2 vertical
        self._swipe_fin = -1.0
        self._swipe_cand = 0
        self._swipe_dir_cand = 0
        self._swipe_pico = 0.0
        self._omega_inicio = 0.0

    def alternar_activo(self):
        """Respaldo con teclado (barra espaciadora) por si el OK falla en una demo."""
        with self.lock:
            self._activar(not self.activo, self._t_det or 0.0)

    def intercambiar_manos(self):
        with self.lock:
            self.invertido = not self.invertido
            self._soltar(0.0)
            self._avisar("Manos intercambiadas", self._t_det or 0.0)

    def colocar(self, giro, inclinacion):
        """Pone la camara en un punto de la orbita sin velocidad (al llegar a otro astro)."""
        with self.lock:
            self._soltar(self._t_det or 0.0)
            self.giro, self.inclinacion = float(giro), float(inclinacion)
            self.omega = self.omega_v = 0.0

    def avisar(self, texto):
        """Aviso en pantalla desde otro hilo (p. ej. al cambiar la exposicion con el teclado)."""
        with self.lock:
            self._avisar(texto, self._t_det or 0.0)

    def radio_px(self):
        return self.radio_base * math.exp(self.log_escala)

    def _radio_giro(self):
        return min(max(self.radio_px(), C.RADIO_GIRO_MIN), C.RADIO_GIRO_MAX)

    def _rol(self, m):
        if not self.invertido:
            return m.rol
        return "der" if m.rol == "izq" else "izq"

    def _avisar(self, texto, t):
        self.avisos = [a for a in self.avisos if t - a[1] < 2.0][-2:] + [(texto, t)]

    def _soltar_giro(self):
        """Al abrir la mano (o perderla) la Tierra conserva el impulso del arrastre."""
        if self.giro_mano is not None:
            if abs(self._vel_arrastre) > 0.3:
                self.omega = float(np.clip(self._vel_arrastre, -C.OMEGA_MAX, C.OMEGA_MAX))
            if abs(self._vel_arrastre_v) > 0.3:
                self.omega_v = float(np.clip(self._vel_arrastre_v, -C.OMEGA_V_MAX, C.OMEGA_V_MAX))
        self.giro_mano = None
        self.swipe = "reposo"

    def _inclinar(self, dang):
        """Suma a la inclinacion sin pasar de los polos (ahi se frena en seco)."""
        lim = math.radians(C.INCLINACION_MAX)
        nueva = self.inclinacion + dang
        if abs(nueva) >= lim:
            nueva = math.copysign(lim, nueva)
            if self.omega_v * nueva > 0:
                self.omega_v = 0.0
        self.inclinacion = nueva

    def _soltar(self, t):
        self._soltar_giro()
        self.zoom_mano = None

    def _activar(self, on, t, motivo=None):
        self.activo = on
        self._soltar(t)
        self._sin_manos_desde = None
        self._avisar(motivo or ("Control activado" if on else "Control en pausa"), t)

    # ------------------------------------------------------------------
    # Hilo de deteccion
    # ------------------------------------------------------------------
    def actualizar(self, manos, t):
        with self.lock:
            dt = 1 / 30 if self._t_det is None else min(max(t - self._t_det, 1e-3), 0.2)
            self._t_det = t
            vis = [m for m in manos.values() if m.visible]
            progreso = self._procesar_ok(vis, manos, t)
            acciones = {}

            if self.activo:
                if vis:
                    self._sin_manos_desde = None
                elif self._sin_manos_desde is None:
                    self._sin_manos_desde = t
                elif C.DESACTIVAR_SIN_MANOS > 0 and t - self._sin_manos_desde > C.DESACTIVAR_SIN_MANOS:
                    self._activar(False, t, "En pausa: no veo tus manos")

            if self.activo:
                izq = [m for m in vis if self._rol(m) == "izq"]
                der = [m for m in vis if self._rol(m) == "der"]
                self._zoom(izq[0] if izq else None, t, acciones)
                self._giro(der[0] if der else None, t, dt, acciones)

            self._vistas = tuple(self._vista(m, progreso.get(m.id, 0.0), acciones.get(m.id))
                                 for m in manos.values())

    def _procesar_ok(self, vis, manos, t):
        for hid in list(self._ok):
            if hid not in manos:
                del self._ok[hid]
        progreso = {}
        for m in vis:
            st = self._ok.setdefault(m.id, {"desde": None, "ult": -1.0, "bloq": False})
            if m.ok and m.edad(t) >= C.EDAD_MINIMA_MANO:
                st["ult"] = t
                if st["desde"] is None and not st["bloq"]:
                    st["desde"] = t
            else:
                if st["desde"] is not None and t - st["ult"] > C.OK_TOLERANCIA:
                    st["desde"] = None
                if t - st["ult"] > max(0.25, C.OK_TOLERANCIA + 0.05):
                    st["bloq"] = False      # hay que deshacer el OK antes de volver a usarlo
            if st["desde"] is not None:
                p = min((t - st["desde"]) / C.OK_SOSTENER, 1.0)
                progreso[m.id] = p
                if p >= 1.0 and m.ok:           # se completa solo con el OK presente
                    st["desde"], st["bloq"] = None, True
                    if not self.activo:
                        self._activar(True, t)
                    elif C.OK_TAMBIEN_PAUSA:
                        self._activar(False, t)
        return progreso

    def _zoom(self, m, t, acciones):
        if m is None or not m.puno or m.edad(t) < C.EDAD_MINIMA_MANO:
            self.zoom_mano = None
            return
        _, y = a_lienzo(m.palma, self.W, self.H)
        if self.zoom_mano != m.id or m.soltando:
            self.zoom_mano, self._y_prev = m.id, y        # agarra sin saltos / congela al soltar
        else:
            dlog = -C.GANANCIA_ZOOM * (y - self._y_prev) / self.H
            self.log_escala = float(np.clip(self.log_escala + dlog,
                                            math.log(C.ESCALA_MIN), math.log(C.ESCALA_MAX)))
            self._y_prev = y
        acciones[m.id] = "zoom"

    def _giro(self, m, t, dt, acciones):
        if m is None or m.edad(t) < C.EDAD_MINIMA_MANO:
            self._soltar_giro()
            return
        x, y = a_lienzo(m.palma, self.W, self.H)
        R = self._radio_giro()

        if m.puno:                                        # agarrar y arrastrar (en cualquier direccion)
            if self.giro_mano != m.id:
                self.giro_mano, self._x_prev, self._yg_prev = m.id, x, y
                self.omega = self.omega_v = 0.0           # atrapar la Tierra la detiene
                self._vel_arrastre = self._vel_arrastre_v = 0.0
            elif m.soltando:
                self._x_prev, self._yg_prev = x, y        # congela (conserva la velocidad para el lanzamiento)
            else:
                dang = (x - self._x_prev) / R
                dincl = -(y - self._yg_prev) / R           # subir la mano = la superficie sube
                self.giro += dang
                self._inclinar(dincl)
                self._vel_arrastre = 0.5 * self._vel_arrastre + 0.5 * dang / dt
                self._vel_arrastre_v = 0.5 * self._vel_arrastre_v + 0.5 * dincl / dt
                self._x_prev, self._yg_prev = x, y
            self.swipe = "reposo"
            acciones[m.id] = "agarre"
            return
        if self.giro_mano is not None:                    # acaba de abrir la mano
            self._soltar_giro()

        k = 1.0 / (1.0 - 2.0 * C.MARGEN_CAMARA)
        # velocidad en "anchos de la zona comoda por segundo" en los dos ejes: el vertical se
        # escala por el aspecto de la camara para que el mismo movimiento fisico valga lo mismo
        vel = (m.vel[0] * k, m.vel[1] * k * C.CAM_ALTO / C.CAM_ANCHO)
        eje = 0 if abs(vel[0]) >= abs(vel[1]) else 1      # eje dominante: 0 horizontal, 1 vertical
        v = abs(vel[eje])
        # cuenta solo si va claramente en un eje (los diagonales no giran)
        rapido = v > C.SWIPE_V_ON and v > C.SWIPE_DOMINANCIA * abs(vel[1 - eje])
        s = _signo(vel[eje]) * (eje + 1)                  # +-1 horizontal, +-2 vertical
        v_trazo = vel[abs(self._swipe_dir) - 1] if self._swipe_dir else 0.0  # en el eje del ultimo swipe
        lento = math.hypot(*vel) < C.SWIPE_V_OFF

        if max(abs(m.vel_cruda[0]), abs(m.vel_cruda[1])) * k > C.SWIPE_V_SALTO:
            self.swipe, self._swipe_cand = "salto", 0     # salto del rastreo, no un movimiento real
        if self.swipe == "refractario" and t - self._swipe_fin > C.SWIPE_REFRACTARIO:
            self.swipe = "reposo"

        if self.swipe == "salto":
            if lento:
                self.swipe = "reposo"
        elif self.swipe == "reposo":
            # 2 detecciones rapidas seguidas en el mismo sentido (evita disparos por ruido)
            self._swipe_cand = self._swipe_cand + 1 if (rapido and s == self._swipe_dir_cand) else int(rapido)
            self._swipe_dir_cand = s if rapido else 0
            if self._swipe_cand >= 2:
                self._iniciar_trazo(s, v)
        elif self.swipe == "refractario":
            if rapido:
                # mismo sentido, otro eje, o claramente mas fuerte que el anterior: nuevo swipe
                if (s == self._swipe_dir or abs(s) != abs(self._swipe_dir)
                        or v > C.SWIPE_REVERSA * self._swipe_pico):
                    self._iniciar_trazo(s, v)
                else:
                    self.swipe = "ignorando"
        elif self.swipe == "ignorando":                   # regreso de la mano: no cuenta...
            if rapido and v > C.SWIPE_REVERSA * self._swipe_pico:
                self._iniciar_trazo(s, v)                 # ...salvo que sea claramente mas fuerte
            elif lento:
                self.swipe, self._swipe_fin = "refractario", t
        elif self.swipe == "trazo":
            if abs(v_trazo) < C.SWIPE_V_OFF or _signo(v_trazo) != _signo(self._swipe_dir):
                self.swipe, self._swipe_fin = "refractario", t
            else:
                self._swipe_pico = max(self._swipe_pico, abs(v_trazo))

        if self.swipe == "trazo":
            v_trazo = vel[abs(self._swipe_dir) - 1]       # (el trazo pudo empezar en este cuadro)
            # cada swipe SUMA velocidad (como empujar un globo); en contra, la frena
            vertical = abs(self._swipe_dir) == 2
            maximo = C.OMEGA_V_MAX if vertical else C.OMEGA_MAX
            sentido = _signo(self._swipe_dir) * (-1 if vertical else 1)   # subir la mano = inclinacion +
            fuerza = (abs(v_trazo) - 0.5 * C.SWIPE_V_ON) / (C.SWIPE_V_FUERTE - 0.5 * C.SWIPE_V_ON)
            impulso = sentido * maximo * min(max(fuerza, 0.0), 1.0)
            deseado = float(np.clip(self._omega_inicio + impulso, -maximo, maximo))
            actual = self.omega_v if vertical else self.omega
            if (deseado - actual) * sentido > 0:
                actual += (deseado - actual) * (1.0 - math.exp(-dt / 0.05))
                if vertical:
                    self.omega_v = actual
                else:
                    self.omega = actual
            acciones[m.id] = "swipe_v" if vertical else "swipe"
        else:
            acciones[m.id] = "listo"

    def _iniciar_trazo(self, s, v):
        self.swipe, self._swipe_dir, self._swipe_pico = "trazo", s, v
        self._omega_inicio = self.omega_v if abs(s) == 2 else self.omega
        self._swipe_cand = 0

    def _vista(self, m, progreso, accion):
        x, y = a_lienzo(m.palma, self.W, self.H)
        return ManoVista(m.id, self._rol(m), x, y, m.norm.copy(), m.visible, m.puno, m.ok,
                         progreso, accion, m.ratio_puno, m.ratio_ok, m.estirados)

    # ------------------------------------------------------------------
    # Hilo de dibujo
    # ------------------------------------------------------------------
    def avanzar(self, t):
        """Inercia y friccion, a la frecuencia del dibujo (giro fluido aunque la camara vaya a 30)."""
        with self.lock:
            dt = 0.0 if self._t_render is None else min(max(t - self._t_render, 0.0), 0.1)
            self._t_render = t
            if self.giro_mano is not None:
                return
            self.giro += self.omega * dt
            self._inclinar(self.omega_v * dt)
            decae = 0.5 ** (dt / C.FRICCION_VIDA_MEDIA)
            en_trazo = self.activo and self.swipe == "trazo"
            if not self.activo:
                self.omega = C.ROTACION_REPOSO + (self.omega - C.ROTACION_REPOSO) * decae
            elif not (en_trazo and abs(self._swipe_dir) == 1):
                self.omega *= decae
                if abs(self.omega) < 0.005:
                    self.omega = 0.0
            if not (en_trazo and abs(self._swipe_dir) == 2):
                self.omega_v *= 0.5 ** (dt / C.FRICCION_V_VIDA_MEDIA)
                if abs(self.omega_v) < 0.005:
                    self.omega_v = 0.0

    def instantanea(self, t):
        with self.lock:
            self.avisos = [a for a in self.avisos if t - a[1] < 2.0]
            return Estado(self.activo, self.log_escala, self.giro, self.omega, self._vistas,
                          tuple(self.avisos), self.zoom_mano is not None,
                          self.giro_mano is not None, self.swipe, self.invertido,
                          self.inclinacion, self.omega_v)
