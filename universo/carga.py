"""Planetas que se cargan bajo demanda, para que agregar planetas no gaste memoria ni tiempo al arrancar.

Cada planeta tiene dos versiones:
  - CHICA (config.ICONO_ANCHO, ~0.5 MB): se construye al arrancar, en otro hilo. Sirve para los
    iconos de la vista del sistema y para cuando el planeta se ve pequeno (hasta ICONO_RADIO_MAX px).
  - GRANDE (la textura completa, ~80 MB con la luz horneada): se carga en otro hilo cuando la pide un
    viaje (al elegir el destino; el viaje dura VIAJE_DURACION s y cargar tarda ~0.2 s) o cuando el
    planeta se ve grande. Escena solo deja PLANETAS_EN_MEMORIA grandes; libera los menos usados.
Mientras una version no esta lista se dibuja la que haya (o un disco de su color): nunca se espera
en el hilo de dibujo.
"""

import threading
import time

import cv2
import numpy as np

from universo import config as C
from universo.tierra import Tierra


class AstroBajoDemanda:
    """Se usa igual que un Tierra (iluminar, dibujar) pero carga sus texturas cuando hacen falta."""

    def __init__(self, nombre, ruta, generar, color, ancho_respaldo=2048, al_pedir=None, **kw):
        """generar(ancho) -> textura procedural de respaldo; kw: argumentos de Tierra (atmosfera...)."""
        self.nombre, self.ruta, self.generar, self.color = nombre, ruta, generar, color
        self.ancho_respaldo, self.kw, self.al_pedir = ancho_respaldo, kw, al_pedir
        self.chico = None
        self.grande = None
        self.pedido = False                               # se pidio la grande (cargando o lista)
        self.quiere_planas = False                        # el comparador pidio las texturas sin luz (luz_vista)
        self.uso = 0.0                                    # ultima vez que se dibujo la grande (para liberar)
        self._sol = None                                  # ultima direccion del Sol (para hornear al cargar)
        self._lock = threading.Lock()
        self._generacion = 0                              # descarta una carga que termina despues de liberar
        self._hilo_chico = None
        self._hilo = None

    # ------------------------------------------------------------------
    @property
    def lista(self):
        return self.grande is not None

    @property
    def origen(self):
        t = self.grande or self.chico
        return t.origen if t is not None else "cargando"

    def preparar_chico(self, esperar=False):
        if self.chico is not None:
            return
        with self._lock:
            if self._hilo_chico is None:
                self._hilo_chico = threading.Thread(target=self._cargar_chico, daemon=True,
                                                    name=f"icono_{self.nombre}")
                self._hilo_chico.start()
        if esperar:
            self._hilo_chico.join()

    def _cargar_chico(self):
        a = C.ICONO_ANCHO
        t = Tierra(self.ruta, R=int(C.ICONO_RADIO_MAX) + 6, ancho_max=a,
                   respaldo=lambda: self.generar(a), **self.kw)
        if self._sol is not None:
            t.iluminar(self._sol, esperar=True)
        if self.quiere_planas:
            t.preparar_planas(esperar=True)
        self.chico = t

    def preparar(self, esperar=False):
        """Pide la version grande (en otro hilo). esperar=True: no regresa hasta que este lista."""
        with self._lock:
            nuevo = not self.pedido
            self.pedido = True
            gen = self._generacion
        if nuevo:
            self.uso = time.monotonic()
            if self.al_pedir is not None:
                self.al_pedir(self)                       # Escena libera otros si hay demasiados
            h = threading.Thread(target=self._cargar_grande, args=(gen,), daemon=True, name=f"carga_{self.nombre}")
            h.start()
            self._hilo = h
        if esperar and self._hilo is not None:
            self._hilo.join()

    def _cargar_grande(self, gen):
        t = Tierra(self.ruta, respaldo=lambda: self.generar(self.ancho_respaldo), **self.kw)
        if self._sol is not None:
            t.iluminar(self._sol, esperar=True)           # la luz se hornea aqui, no en el dibujo
        if self.quiere_planas:
            t.preparar_planas(esperar=True)               # (lo mismo para la luz de vista del comparador)
        with self._lock:
            if self.pedido and gen == self._generacion:
                self.grande = t

    def preparar_planas(self):
        """El comparador pide las texturas sin luz (luz_vista) de la version chica y la grande."""
        self.quiere_planas = True
        for t in (self.chico, self.grande):
            if t is not None:
                t.preparar_planas()

    def soltar_planas(self):
        self.quiere_planas = False
        for t in (self.chico, self.grande):
            if t is not None:
                t.soltar_planas()

    def listo_comparar(self):
        """La version grande y sus texturas sin luz estan listas (el comparador no dibuja antes)."""
        g = self.grande
        return g is not None and g.listo_comparar()

    def liberar(self):
        """Suelta la version grande (la chica se queda)."""
        with self._lock:
            self.grande = None
            self.pedido = False
            self._generacion += 1

    # ------------------------------------------------------------------
    def iluminar(self, sol_tierra, esperar=False):
        self._sol = np.asarray(sol_tierra, np.float64)
        if self.chico is None:
            self.preparar_chico(esperar)
        for t in (self.chico, self.grande):
            if t is not None:
                t.iluminar(self._sol, esperar)

    def dibujar(self, buf, cx, cy, radio, giro, inclinacion=0.0, sol=(-0.62, -0.38, 0.69), sol_tierra=None,
                rodar=0.0, luz_vista=False):
        if sol_tierra is not None:
            self._sol = np.asarray(sol_tierra, np.float64)
        grande = self.grande
        if radio > C.ICONO_RADIO_MAX:
            if grande is None:
                self.preparar()                           # se ve grande: que se cargue (sin esperar)
            else:
                self.uso = time.monotonic()
        t = grande if (grande is not None and radio > C.ICONO_RADIO_MAX) else (self.chico or grande)
        if t is None:
            self.preparar_chico()
            cv2.circle(buf, (int(round(cx * 16)), int(round(cy * 16))), int(max(radio, 1.0) * 16),
                       tuple(int(c * 0.6) for c in self.color), -1, cv2.LINE_AA, 4)
            return
        t.dibujar(buf, cx, cy, radio, giro, inclinacion, sol, sol_tierra, rodar, luz_vista)
