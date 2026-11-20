"""Pruebas sin camara: manos simuladas -> seguimiento -> control -> dibujo.

Ejecutar:  python test_proyecto.py      (guarda capturas en ./_pruebas)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

import config as C
from control import Control
from manos import OneEuro, SeguidorManos, medir_pose
from simulacion import mano

CAM_W, CAM_H = 640, 480
SALIDA = Path(__file__).with_name("_pruebas")
X_IZQ, X_DER, Y0 = 0.30, 0.70, 0.55


class Sim:
    """Deteccion a 30 Hz y dibujo a 60 Hz, deterministico (sin hilos)."""

    def __init__(self, activo=False, rng=None):
        self.seg, self.ctl = SeguidorManos(), Control()
        self.t = 100.0
        self.rng = rng
        if activo:
            self.ctl.activo = True

    def correr(self, guion, dur):
        t0 = self.t
        for i in range(int(round(dur * 60))):
            self.t = t0 + i / 60.0
            self.ctl.avanzar(self.t)
            if i % 2 == 0:
                dets = guion(self.t - t0)
                self.seg.actualizar(dets, self.t, CAM_W, CAM_H)
                self.ctl.actualizar(self.seg.manos, self.t)
        self.t = t0 + dur
        return self

    @property
    def escala(self):
        return math.exp(self.ctl.log_escala)


def lerp(a, b, k):
    k = min(max(k, 0.0), 1.0)
    return a + (b - a) * k


IZQ = lambda pose, x=X_IZQ, y=Y0: mano(pose, x, y, "Left")
DER = lambda pose, x=X_DER, y=Y0: mano(pose, x, y, "Right")


# ----------------------------------------------------------------------
def test_poses():
    r = {p: medir_pose(*mano(p, 0.5, 0.5)[:2]) for p in ("abierta", "puno", "ok", "relajada")}
    assert r["abierta"][0] > C.PUNO_OFF and r["abierta"][1] > C.OK_OFF
    assert r["puno"][0] < C.PUNO_ON and r["puno"][2] < C.OK_DEDOS_MIN
    assert r["ok"][1] < C.OK_ON and r["ok"][2] >= C.OK_DEDOS_MIN and r["ok"][0] > C.PUNO_OFF
    assert r["relajada"][1] > C.OK_OFF and r["relajada"][0] > C.PUNO_ON


def test_one_euro():
    rng = np.random.default_rng(1)
    f = OneEuro(C.OE_MINCUT, C.OE_BETA, C.OE_DCUT, vectorial=True)
    crudo, filt = [], []
    for i in range(400):
        x = np.array([[0.5, 0.5]]) + rng.normal(0, 0.003, (1, 2))
        crudo.append(x[0, 0]); filt.append(f(x, i / 30)[0, 0])
    r = np.std(crudo[60:]) / np.std(filt[60:])
    print(f"  temblor con la mano quieta reducido {r:.1f}x")
    assert r > 3


def test_roles():
    s = Sim().correr(lambda t: [IZQ("abierta"), DER("abierta")], 0.5)
    roles = {m.rol for m in s.seg.manos.values()}
    assert roles == {"izq", "der"}
    izq = [m for m in s.seg.manos.values() if m.rol == "izq"][0]
    assert izq.palma[0] < 0.5

    # MediaPipe dice que ambas son "Left": decide la posicion
    s = Sim().correr(lambda t: [mano("abierta", 0.3, Y0, "Left"), mano("abierta", 0.7, Y0, "Left")], 0.5)
    der = [m for m in s.seg.manos.values() if m.rol == "der"][0]
    assert der.palma[0] > 0.5

    # una sola mano con etiqueta que parpadea (25 % de errores): el rol no cambia
    rng = np.random.default_rng(4)
    roles = []

    def g(t):
        lado = "Right" if rng.random() > 0.25 else "Left"
        return [mano("abierta", 0.6, Y0, lado, score=0.8)]
    s = Sim()
    for _ in range(40):
        s.correr(g, 0.1)
        roles.append(list(s.seg.manos.values())[0].rol)
    print(f"  rol con etiqueta ruidosa: {set(roles)}")
    assert set(roles) == {"der"}


def test_ok_activa_y_pausa():
    s = Sim()
    s.correr(lambda t: [DER("abierta")], 0.5)
    s.correr(lambda t: [DER("ok")], 0.15)                      # OK muy corto: nada
    s.correr(lambda t: [DER("abierta")], 0.3)
    assert not s.ctl.activo, "un OK de 0.15 s no debe activar"
    s.correr(lambda t: [DER("ok")], 0.5)
    assert s.ctl.activo, "OK sostenido 0.3 s debe activar"
    s.correr(lambda t: [DER("ok")], 1.0)                       # seguir sosteniendo: no alterna
    assert s.ctl.activo, "sostener el OK no debe desactivar"
    s.correr(lambda t: [DER("abierta")], 0.5)
    s.correr(lambda t: [IZQ("ok"), DER("abierta")], 0.7)      # OK con la otra mano: pausa
    assert not s.ctl.activo


def test_pausa_no_mueve():
    s = Sim()
    s.correr(lambda t: [IZQ("puno", y=lerp(0.6, 0.35, t / 0.6)), DER("puno", x=lerp(0.6, 0.8, t / 0.6))], 0.8)
    assert abs(s.ctl.log_escala) < 1e-9, "en pausa no hay zoom"


def test_zoom_izquierda():
    s = Sim(activo=True)
    s.correr(lambda t: [IZQ("abierta", y=0.6)], 0.4)
    s.correr(lambda t: [IZQ("puno", y=0.6)], 0.3)
    s.correr(lambda t: [IZQ("puno", y=lerp(0.6, 0.4, t / 0.5))], 0.7)    # sube 0.2 = 25 % de pantalla
    e1 = s.escala
    esperado = math.exp(C.GANANCIA_ZOOM * 0.25)
    print(f"  subir el puno 1/4 de pantalla: x{e1:.2f} (esperado ~x{esperado:.2f})")
    assert abs(e1 - esperado) / esperado < 0.08
    s.correr(lambda t: [IZQ("abierta", y=lerp(0.4, 0.6, t / 0.5))], 0.8)  # baja con la mano abierta: nada
    assert abs(s.escala - e1) < 0.02, "con la mano abierta no debe cambiar el zoom"
    s.correr(lambda t: [IZQ("puno", y=0.6)], 0.3)
    s.correr(lambda t: [IZQ("puno", y=lerp(0.6, 0.8, t / 0.5))], 0.7)    # baja con puno: aleja
    assert s.escala < e1 * 0.7


def test_giro_agarre_e_inercia():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.6)], 0.4)
    s.correr(lambda t: [DER("puno", x=0.6)], 0.3)
    g0 = s.ctl.giro
    s.correr(lambda t: [DER("puno", x=lerp(0.6, 0.76, t / 0.5))], 0.6)    # 0.16 = 20 % de pantalla
    R = min(max(C.RADIO_BASE_REL * C.ALTO, C.RADIO_GIRO_MIN), C.RADIO_GIRO_MAX)
    esperado = 0.2 * C.ANCHO / R
    print(f"  arrastrar 20 % de pantalla: {s.ctl.giro - g0:.2f} rad (esperado ~{esperado:.2f})")
    assert abs((s.ctl.giro - g0) - esperado) < 0.15
    assert abs(s.ctl.omega) < 1e-9

    # lanzar: abrir la mano mientras se mueve
    s.correr(lambda t: [DER("puno", x=lerp(0.76, 0.6, t / 0.3))], 0.4)
    s.correr(lambda t: [DER("puno", x=lerp(0.6, 0.75, t / 0.2))], 0.15)
    s.correr(lambda t: [DER("abierta", x=0.75 + 0.1 * t)], 0.25)
    w0 = s.ctl.omega
    print(f"  velocidad al soltar: {w0:.2f} rad/s")
    assert w0 > 1.0
    g1 = s.ctl.giro
    s.correr(lambda t: [], 2.0)
    assert s.ctl.giro > g1 + 0.5 and 0 < s.ctl.omega < w0, "debe seguir girando y frenarse"


def swipe(x0, x1, dur, pose="abierta"):
    return lambda t: [DER(pose, x=lerp(x0, x1, t / dur))]


def test_swipe_y_regreso():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.35)], 0.5)
    s.correr(swipe(0.35, 0.75, 0.25), 0.3)                     # swipe a la derecha (2 anchos/s)
    w1 = s.ctl.omega
    s.correr(swipe(0.75, 0.35, 0.30), 0.35)                    # regreso inmediato, tambien rapido
    w2 = s.ctl.omega
    print(f"  swipe: {w1:.2f} rad/s  ->  tras el regreso de la mano: {w2:.2f} rad/s")
    assert w1 > 1.2, "el swipe debe darle vuelo"
    assert w2 > 0.6 * w1, "el regreso de la mano no debe frenarla ni invertirla"
    s.correr(lambda t: [DER("abierta", x=0.35)], 0.2)
    s.correr(swipe(0.35, 0.75, 0.25), 0.3)                     # otro swipe igual: SUMA velocidad
    w3 = s.ctl.omega
    print(f"  segundo swipe en el mismo sentido: {w3:.2f} rad/s")
    assert w3 > w2 + 0.8

    s.correr(lambda t: [DER("abierta", x=0.75)], 0.8)
    w4 = s.ctl.omega
    s.correr(swipe(0.75, 0.35, 0.25), 0.3)                     # swipe en contra: frena
    print(f"  swipe en contra: {w4:.2f} -> {s.ctl.omega:.2f} rad/s")
    assert s.ctl.omega < w4 - 1.0
    for _ in range(2):                                         # y si insistes, gira al reves
        s.correr(lambda t: [DER("abierta", x=lerp(0.35, 0.75, t / 0.8))], 0.8)
        s.correr(swipe(0.75, 0.35, 0.2), 0.25)
    print(f"  dos swipes mas a la izquierda: {s.ctl.omega:.2f} rad/s")
    assert s.ctl.omega < -1.0


def test_impulso_proporcional():
    res = []
    for dur in (0.45, 0.25, 0.12):                             # lento, medio, fuerte
        s = Sim(activo=True)
        s.correr(lambda t: [DER("abierta", x=0.3)], 0.5)
        s.correr(swipe(0.3, 0.75, dur), dur + 0.05)
        res.append(s.ctl.omega)
    print("  impulso segun la velocidad del swipe: " + ", ".join(f"{w:.2f}" for w in res) + " rad/s")
    assert res[0] < res[1] < res[2] <= C.OMEGA_MAX + 1e-6


def test_impulso_previo_no_bloquea():
    """Echar la mano hacia atras (rapido) justo antes de un swipe fuerte no debe bloquearlo."""
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.6)], 0.5)
    s.correr(swipe(0.6, 0.38, 0.17), 0.2)                      # impulso previo hacia la izquierda (~1.6/s)
    s.correr(swipe(0.38, 0.8, 0.12), 0.2)                      # swipe fuerte a la derecha (~4.4/s)
    s.correr(lambda t: [DER("abierta", x=0.8)], 0.3)
    print(f"  impulso previo + swipe fuerte: omega {s.ctl.omega:.2f}")
    assert s.ctl.omega > 1.0


def test_movimiento_lento_no_gira():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.3)], 0.5)
    s.correr(swipe(0.3, 0.7, 2.0), 2.2)
    print(f"  mano lenta: omega {s.ctl.omega:.3f}")
    assert abs(s.ctl.omega) < 0.05
    s.correr(lambda t: [DER("abierta", x=0.7, y=lerp(Y0, 0.3, t / 0.5))], 0.6)
    s.correr(lambda t: [DER("abierta", x=0.7, y=lerp(0.3, 0.8, t / 0.25))], 0.3)
    assert abs(s.ctl.omega) < 0.05, "un movimiento vertical rapido no debe girar"


def test_salto_de_rastreo_no_gira():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.7)], 0.5)
    s.correr(lambda t: [DER("abierta", x=0.35)], 0.6)         # la mano "se teletransporta"
    print(f"  salto de 0.35 en un cuadro: omega {s.ctl.omega:.2f}")
    assert abs(s.ctl.omega) < 0.3
    s.correr(swipe(0.35, 0.75, 0.25), 0.3)                     # y luego un swipe real si funciona
    assert s.ctl.omega > 2.0


def test_atrapar_detiene():
    s = Sim(activo=True)
    s.ctl.omega = 4.0
    s.correr(lambda t: [DER("abierta", x=0.6)], 0.3)
    s.correr(lambda t: [DER("puno", x=0.6)], 0.2)
    g = s.ctl.giro
    s.correr(lambda t: [DER("puno", x=0.6)], 0.5)
    assert s.ctl.omega == 0.0 and abs(s.ctl.giro - g) < 0.01


def test_dos_manos_a_la_vez():
    s = Sim(activo=True)
    s.correr(lambda t: [IZQ("abierta"), DER("abierta", x=0.35)], 0.4)
    s.correr(lambda t: [IZQ("puno"), DER("abierta", x=0.35)], 0.2)
    s.correr(lambda t: [IZQ("puno", y=lerp(Y0, 0.4, t / 0.3)), DER("abierta", x=lerp(0.35, 0.75, t / 0.25))], 0.35)
    assert s.escala > 1.3 and s.ctl.omega > 1.0


def test_pausa_automatica():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta")], 0.3)
    s.correr(lambda t: [], C.DESACTIVAR_SIN_MANOS + 0.5)
    assert not s.ctl.activo


def test_render():
    import cv2
    from control import Estado, ManoVista
    from render import Escena
    SALIDA.mkdir(exist_ok=True)
    esc = Escena()
    n, _, _, _ = mano("abierta", 0.3, 0.5, "Left")
    info = {"fps_render": 60, "fps_det": 30, "lat_ms": 14}
    cam = np.full((480, 640, 3), 40, np.uint8)
    mv = lambda i, rol, x, y, acc, p=0.0, puno=False, ok=False: ManoVista(
        i, rol, x, y, n, True, puno, ok, p, acc, 1.9 if not puno else 0.9, 0.2 if ok else 0.9, 3)
    casos = {
        "pausa_ok": Estado(False, 0.0, math.radians(99), 0.07, (mv(0, "der", 900, 380, None, 0.6, ok=True),),
                           (), False, False, "reposo", False),
        "activo": Estado(True, math.log(1.6), math.radians(80), 2.0,
                         (mv(0, "izq", 300, 300, "zoom", puno=True), mv(1, "der", 980, 420, "swipe")),
                         (("Control activado", 0.0),), True, False, "trazo", False),
    }
    tiempos = []
    for nombre, est in casos.items():
        for i in range(20):
            a = time.perf_counter()
            f = esc.componer(est.log_escala, est.giro + i * 0.02, est, info, cam, True, True, 0.5)
            tiempos.append(time.perf_counter() - a)
        cv2.imwrite(str(SALIDA / f"render_{nombre}.png"), f)
    med = np.median(tiempos) * 1000
    print(f"  cuadro completo (Tierra + HUD + camara): {med:.1f} ms")
    assert med < 25


# ----------------------------------------------------------------------
class FuenteFalsa:
    def leer(self):
        time.sleep(1 / 60)
        return True, np.full((CAM_H, CAM_W, 3), 25, np.uint8)


class DetectorGuion:
    """Reproduce en TIEMPO REAL un guion completo pasando por los 3 hilos."""

    def __init__(self):
        self.t0 = None
        self.rng = np.random.default_rng(5)

    def detectar(self, frame, t_ms):
        if self.t0 is None:
            self.t0 = t_ms
        t = (t_ms - self.t0) / 1000.0
        time.sleep(0.015)                              # costo de inferencia simulado
        if t < 0.4:
            d = []
        elif t < 0.8:
            d = [DER("abierta")]
        elif t < 1.4:
            d = [DER("ok")]                            # activar
        elif t < 1.8:
            d = [IZQ("abierta"), DER("abierta")]
            d[1] = DER("abierta")
        elif t < 2.1:
            d = [IZQ("puno"), DER("abierta")]
        elif t < 2.7:
            d = [IZQ("puno", y=lerp(Y0, 0.42, (t - 2.1) / 0.5)), DER("abierta")]   # zoom
        elif t < 3.1:
            d = [IZQ("abierta", y=0.42), DER("abierta", x=lerp(X_DER, 0.35, (t - 2.7) / 0.4))]
        elif t < 3.4:
            d = [IZQ("abierta", y=0.42), DER("abierta", x=lerp(0.35, 0.78, (t - 3.1) / 0.25))]  # swipe
        else:
            d = [IZQ("abierta", y=0.42), DER("abierta", x=0.78)]
        return [(n + self.rng.normal(0, 0.0012, n.shape), w, l, s) for n, w, l, s in d]


def test_integracion_hilos():
    import cv2
    from main import Aplicacion, Captura
    SALIDA.mkdir(exist_ok=True)
    app = Aplicacion(Captura(FuenteFalsa().leer), DetectorGuion())
    app.iniciar()
    t0 = time.perf_counter()
    fotos = {0.6: "1_pausa", 1.2: "2_ok_progreso", 2.5: "3_zoom", 3.35: "4_swipe", 4.2: "5_girando"}
    hechas, n, tiempos = set(), 0, []
    while time.perf_counter() - t0 < 4.4:
        a = time.perf_counter()
        f = app.paso_render()
        tiempos.append(time.perf_counter() - a)
        n += 1
        t = time.perf_counter() - t0
        for k, nombre in fotos.items():
            if t >= k and nombre not in hechas:
                cv2.imwrite(str(SALIDA / f"integ_{nombre}.png"), f)
                hechas.add(nombre)
        time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - a)))
    err = app.deteccion.error
    app.detener()
    est = app.control.instantanea(time.perf_counter())
    fps = n / (time.perf_counter() - t0)
    print(f"  dibujo {fps:.0f} fps (p95 {np.percentile(tiempos, 95) * 1000:.1f} ms) | deteccion "
          f"{app.deteccion.fps:.0f} fps | activo={est.activo} escala x{math.exp(est.log_escala):.2f} "
          f"omega {est.omega:.2f} rad/s")
    assert err is None, err
    assert est.activo
    assert math.exp(est.log_escala) > 1.4
    assert est.omega > 1.0
    assert fps > 40


if __name__ == "__main__":
    pruebas = [v for k, v in globals().items() if k.startswith("test_")]
    fallos = 0
    for p in pruebas:
        print(f"- {p.__name__}")
        try:
            p()
            print("  OK")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  FALLO:", e)
            fallos += 1
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas OK")
    sys.exit(1 if fallos else 0)
