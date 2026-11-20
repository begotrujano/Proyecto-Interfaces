"""Universo en tus manos - prototipo: la Tierra como globo terraqueo controlado con gestos.

    1. Haz el signo de OK (una vez, con cualquier mano) para activar el control.
    2. Mano IZQUIERDA: cierra el puno y sube o baja -> zoom.
    3. Mano DERECHA: swipe horizontal -> la Tierra gira con inercia; puno -> la agarras.
    4. OK otra vez (o la barra espaciadora) -> pausa.

Arquitectura (3 hilos, para que nada se trabe):
    Captura    -> lee la camara sin parar y conserva solo el ultimo cuadro
    Deteccion  -> MediaPipe + filtros + gestos, siempre sobre el cuadro mas nuevo
    Principal  -> dibuja a 60 FPS con inercia y suavizado propios

Uso:   python main.py
       python main.py --camara 1 --dshow --pantalla-completa
"""

import argparse
import math
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # arranque rapido en Windows
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

import config as C
from control import Control
from manos import SeguidorManos
from render import Escena

TITULO = "Universo en tus manos"
AQUI = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Camara y detector
# ----------------------------------------------------------------------
def abrir_camara(indice, dshow):
    backend = cv2.CAP_DSHOW if (dshow and sys.platform.startswith("win")) else cv2.CAP_ANY
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_ALTO)
    cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if C.CAM_EXPOSICION is not None:
        fijar_exposicion(cap, C.CAM_EXPOSICION)
    return cap


def fijar_exposicion(cap, valor):
    """valor=None -> automatica. Si no, exposicion manual (DirectShow: log2 de segundos,
    -6 = 1/64 s, -7 = 1/128 s). Menos exposicion = menos barrido al mover rapido la mano
    = MediaPipe no la pierde (pero la imagen se oscurece: necesitas buena luz)."""
    if valor is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)      # DirectShow
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)         # MSMF / V4L2
        return
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(valor))


def buscar_modelo():
    for carpeta in (AQUI, AQUI.parent):
        ruta = carpeta / C.MODELO
        if ruta.is_file():
            return ruta
    return None


class DetectorMediaPipe:
    """detectar(frame_bgr, t_ms) -> [(norm(21,2), world(21,3)|None, 'Left'|'Right'|None, score)]

    Por defecto usa solo el HandLandmarker que viene DENTRO de gesture_recognizer.task
    (el .task es un zip): da los mismos puntos sin pagar el clasificador de gestos,
    que no usamos. Asi cada deteccion tarda menos y se pierden menos cuadros."""

    def __init__(self, ruta_modelo):
        import mediapipe as mp
        self.mp = mp
        vision = mp.tasks.vision
        comunes = dict(
            running_mode=vision.RunningMode.VIDEO,
            num_hands=C.NUM_MANOS,
            min_hand_detection_confidence=C.CONFIANZA_DETECCION,
            min_hand_presence_confidence=C.CONFIANZA_PRESENCIA,
            min_tracking_confidence=C.CONFIANZA_RASTREO,
        )
        # los modelos se pasan en memoria: evita problemas con rutas con acentos o espacios
        landmarker = extraer_landmarker(ruta_modelo) if C.SOLO_LANDMARKER else None
        if landmarker is not None:
            self.tipo = "HandLandmarker"
            self.modelo = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=landmarker), **comunes))
            self._correr = self.modelo.detect_for_video
        else:
            self.tipo = "GestureRecognizer"
            self.modelo = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(ruta_modelo).read_bytes()),
                **comunes))
            self._correr = self.modelo.recognize_for_video

    def detectar(self, frame_bgr, t_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._correr(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), t_ms)
        mundo = getattr(res, "hand_world_landmarks", None) or []
        lados = getattr(res, "handedness", None) or []
        salida = []
        for i, lm in enumerate(res.hand_landmarks):
            norm = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
            world = (np.array([[p.x, p.y, p.z] for p in mundo[i]], dtype=np.float64)
                     if i < len(mundo) else None)
            etiqueta, score = None, 0.5
            if i < len(lados) and lados[i]:
                etiqueta, score = lados[i][0].category_name, float(lados[i][0].score)
            salida.append((norm, world, etiqueta, score))
        return salida

    def cerrar(self):
        self.modelo.close()


def extraer_landmarker(ruta_modelo):
    """Saca hand_landmarker.task del bundle del gesture recognizer (o None si no esta)."""
    try:
        with zipfile.ZipFile(ruta_modelo) as z:
            return z.read("hand_landmarker.task")
    except (zipfile.BadZipFile, KeyError, OSError):
        return None


# ----------------------------------------------------------------------
# Hilos
# ----------------------------------------------------------------------
class Captura(threading.Thread):
    """Lee cuadros sin parar y conserva solo el ultimo: cero cola = cero retraso acumulado."""

    def __init__(self, lector, liberar=lambda: None, espejo=True, ajustar=None):
        super().__init__(daemon=True, name="captura")
        self.lector, self.liberar, self.espejo = lector, liberar, espejo
        self.ajustar = ajustar            # funcion(valor) que cambia la exposicion de la camara
        self.exposicion = C.CAM_EXPOSICION
        self._pendiente = False
        self.fps = 0.0
        self.cond = threading.Condition()
        self.frame, self.ts, self.seq = None, 0.0, 0
        self.activo = True
        self.fallos = 0

    def cambiar_exposicion(self, paso):
        """paso=None -> automatica; +1/-1 -> mas/menos tiempo de exposicion.
        Se aplica desde el propio hilo de captura (la camara no se toca desde dos hilos)."""
        if self.ajustar is None:
            return
        if paso is None:
            self.exposicion = None
        elif self.exposicion is None:
            self.exposicion = C.CAM_EXPOSICION_MANUAL
        else:
            self.exposicion = min(max(self.exposicion + paso, -13), 0)
        self._pendiente = True

    def run(self):
        t_prev = None
        while self.activo:
            if self._pendiente:
                self._pendiente = False
                self.ajustar(self.exposicion)
            ok, f = self.lector()
            if not ok or f is None:
                self.fallos += 1
                time.sleep(0.01)
                continue
            self.fallos = 0
            ts = time.perf_counter()
            if t_prev is not None:
                self.fps = 0.9 * self.fps + 0.1 / max(ts - t_prev, 1e-3)
            t_prev = ts
            if self.espejo:
                f = cv2.flip(f, 1)        # espejo: tu mano derecha aparece a la derecha
            with self.cond:
                self.frame, self.ts, self.seq = f, ts, self.seq + 1
                self.cond.notify_all()
        self.liberar()

    def esperar(self, ultimo_seq, timeout=0.5):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != ultimo_seq or not self.activo, timeout)
            return self.frame, self.ts, self.seq

    def detener(self):
        self.activo = False
        with self.cond:
            self.cond.notify_all()


class Deteccion(threading.Thread):
    def __init__(self, captura, detector, seguidor, control):
        super().__init__(daemon=True, name="deteccion")
        self.captura, self.detector = captura, detector
        self.seguidor, self.control = seguidor, control
        self.activo = True
        self.fps = 0.0
        self.lat_ms = 0.0
        self.error = None

    def run(self):
        seq, t_ms_prev, t_prev = -1, -1, None
        try:
            while self.activo:
                frame, ts, nuevo = self.captura.esperar(seq)
                if frame is None or nuevo == seq:
                    continue
                seq = nuevo
                alto, ancho = frame.shape[:2]
                t_ms = max(int(ts * 1000), t_ms_prev + 1)          # MediaPipe exige tiempos crecientes
                t_ms_prev = t_ms
                t0 = time.perf_counter()
                dets = self.detector.detectar(frame, t_ms)
                lat = (time.perf_counter() - t0) * 1000.0
                self.seguidor.actualizar(dets, ts, ancho, alto)
                self.control.actualizar(self.seguidor.manos, ts)
                self.lat_ms = 0.9 * self.lat_ms + 0.1 * lat if self.lat_ms else lat
                if t_prev is not None:
                    self.fps = 0.9 * self.fps + 0.1 / max(ts - t_prev, 1e-3)
                t_prev = ts
        except Exception as e:                                     # no morir en silencio
            self.error = e
            raise


class Suave:
    """Suavizado visual: el dibujo va a 60 FPS aunque la deteccion llegue a 30."""

    def __init__(self, tau):
        self.tau = tau
        self.log_escala = None
        self.giro = None

    def paso(self, log_escala, giro, dt):
        if self.log_escala is None or abs(giro - self.giro) > math.pi:
            self.log_escala, self.giro = log_escala, giro
            return
        a = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
        self.log_escala += (log_escala - self.log_escala) * a
        self.giro += (giro - self.giro) * a


# ----------------------------------------------------------------------
class Aplicacion:
    def __init__(self, captura, detector, ancho=C.ANCHO, alto=C.ALTO):
        self.captura, self.detector = captura, detector
        self.escena = Escena(ancho, alto)
        self.seguidor = SeguidorManos()
        self.control = Control(ancho, alto)
        self.deteccion = Deteccion(captura, detector, self.seguidor, self.control)
        self.suave = Suave(C.SUAVIZADO_TAU)
        self.ver_camara = True
        self.ver_info = True
        self.fps_render = 0.0
        self._t_prev = time.perf_counter()

    def iniciar(self):
        self.captura.start()
        self.deteccion.start()

    def detener(self):
        self.deteccion.activo = False
        self.captura.detener()
        self.deteccion.join(timeout=2)
        self.captura.join(timeout=2)
        if hasattr(self.detector, "cerrar"):
            self.detector.cerrar()

    def paso_render(self):
        t = time.perf_counter()
        dt = min(t - self._t_prev, 0.1)
        self._t_prev = t
        self.fps_render = 0.9 * self.fps_render + 0.1 / max(dt, 1e-3)

        self.control.avanzar(t)
        est = self.control.instantanea(t)
        self.suave.paso(est.log_escala, est.giro, dt)
        info = {"fps_render": self.fps_render, "fps_det": self.deteccion.fps, "lat_ms": self.deteccion.lat_ms,
                "fps_cam": getattr(self.captura, "fps", 0.0),
                "exposicion": getattr(self.captura, "exposicion", None)}
        return self.escena.componer(self.suave.log_escala, self.suave.giro, est, info,
                                    self.captura.frame, self.ver_camara, self.ver_info, t)

    def tecla(self, k):
        if k == ord(" "):
            self.control.alternar_activo()
        elif k == ord("v"):
            self.ver_camara = not self.ver_camara
        elif k == ord("h"):
            self.ver_info = not self.ver_info
        elif k == ord("r"):
            with self.control.lock:
                self.control.reiniciar()
        elif k == ord("i"):
            self.control.intercambiar_manos()
        elif k in (ord("["), ord("]"), ord("e")) and hasattr(self.captura, "cambiar_exposicion"):
            self.captura.cambiar_exposicion(None if k == ord("e") else (-1 if k == ord("[") else 1))
            exp = self.captura.exposicion
            texto = ("Exposición automática" if exp is None
                     else f"Exposición {exp:+.0f}  (1/{2 ** -exp:.0f} s)")
            self.control.avisar(texto)

    def correr(self, pantalla_completa=False):
        cv2.namedWindow(TITULO, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITULO, C.ANCHO, C.ALTO)
        completa = pantalla_completa
        if completa:
            cv2.setWindowProperty(TITULO, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        objetivo = 1.0 / C.FPS_RENDER
        while True:
            t0 = time.perf_counter()
            cv2.imshow(TITULO, self.paso_render())
            k = cv2.waitKey(max(1, int((objetivo - (time.perf_counter() - t0)) * 1000))) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("f"):
                completa = not completa
                cv2.setWindowProperty(TITULO, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if completa else cv2.WINDOW_NORMAL)
            elif k != 255:
                self.tecla(k)
            if cv2.getWindowProperty(TITULO, cv2.WND_PROP_VISIBLE) < 1:
                break
            if self.deteccion.error is not None:
                print("Error en el hilo de deteccion:", self.deteccion.error)
                break
            if self.captura.fallos > 100:
                print("La camara dejo de entregar imagen. Revisa que otra app no la este usando.")
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camara", type=int, default=C.CAM_INDICE, help="indice de la camara (0, 1, ...)")
    ap.add_argument("--dshow", action="store_true", help="Windows: usar DirectShow (a veces abre mas rapido)")
    ap.add_argument("--pantalla-completa", action="store_true")
    args = ap.parse_args()

    ruta = buscar_modelo()
    if ruta is None:
        print(f"Falta {C.MODELO}. Ponlo en {AQUI}.")
        return 1
    cap = abrir_camara(args.camara, args.dshow)
    if cap is None:
        print("No se pudo abrir la camara. Prueba --camara 1, o cierra otras apps que la usen.")
        return 1
    detector = DetectorMediaPipe(ruta)
    print(f"Camara: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS (pedido)   Modelo: {ruta.name} ({detector.tipo})")
    print("Cargando la Tierra...")
    captura = Captura(cap.read, cap.release, ajustar=lambda v: fijar_exposicion(cap, v))
    app = Aplicacion(captura, detector)
    print(f"Textura: {app.escena.tierra.origen}")
    app.iniciar()
    try:
        app.correr(args.pantalla_completa)
    finally:
        app.detener()
    return 0


if __name__ == "__main__":
    sys.exit(main())
