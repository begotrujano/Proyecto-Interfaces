"""Pruebas sin camara: manos simuladas -> seguimiento -> control -> dibujo.

Ejecutar:  python pruebas/test_proyecto.py      (guarda capturas en pruebas/capturas)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # para importar universo y main

from universo import config as C
from universo.control import Control
from universo.manos import OneEuro, SeguidorManos, medir_pose
from simulacion import mano

CAM_W, CAM_H = 640, 480
SALIDA = Path(__file__).with_name("capturas")
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


def test_ok_realista():
    """OK como lo hace la gente (pulgar en la falange, dedos relajados, profundidad
    equivocada), con temblor y con cuadros donde MediaPipe lo pierde: debe activar."""
    _, ok, dedos = medir_pose(*mano("ok_real", 0.5, 0.5)[:2])
    print(f"  ok_real: aro {ok:.2f} (<{C.OK_ON})  dedos {dedos}/3")
    assert ok < C.OK_ON and dedos >= C.OK_DEDOS_MIN
    rng = np.random.default_rng(7)
    s = Sim()
    s.correr(lambda t: [DER("abierta")], 0.5)
    cuadro = [0]

    def g(t):
        cuadro[0] += 1                      # 1 de cada 4 detecciones el OK "se rompe"
        pose = "abierta" if cuadro[0] % 4 == 0 else "ok_real"
        return [mano(pose, X_DER, Y0, "Right", ruido=0.002, rng=rng)]
    s.correr(g, 0.8)
    assert s.ctl.activo, "un OK realista e intermitente debe activar"


def test_ok_en_movimiento_no_activa():
    s = Sim()
    s.correr(lambda t: [DER("abierta", x=0.3)], 0.5)
    for _ in range(3):                      # aro cerrado pero la mano barriendo de lado a lado
        s.correr(lambda t: [DER("ok", x=lerp(0.3, 0.8, t / 0.4))], 0.4)
        s.correr(lambda t: [DER("ok", x=lerp(0.8, 0.3, t / 0.4))], 0.4)
    assert not s.ctl.activo, "un OK en pleno movimiento no debe activar"


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


def swipe_v(y0, y1, dur, x=X_DER):
    return lambda t: [DER("abierta", x=x, y=lerp(y0, y1, t / dur))]


def test_swipe_vertical_inclina():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", y=0.75)], 0.5)
    s.correr(swipe_v(0.75, 0.30, 0.18), 0.22)                 # swipe hacia ARRIBA
    wv = s.ctl.omega_v
    print(f"  swipe arriba: omega_v {wv:.2f} rad/s, omega {s.ctl.omega:.2f}")
    assert wv > 1.0, "un swipe vertical debe inclinar"
    assert abs(s.ctl.omega) < 0.05, "un swipe vertical no debe girar"
    s.correr(swipe_v(0.30, 0.75, 0.30), 0.35)                 # regreso de la mano: se ignora
    print(f"  tras el regreso de la mano: omega_v {s.ctl.omega_v:.2f}")
    assert s.ctl.omega_v > 0.4 * wv, "el regreso de la mano no debe frenarla ni invertirla"
    s.correr(lambda t: [DER("abierta", y=0.75)], 1.5)
    print(f"  inclinacion final: {math.degrees(s.ctl.inclinacion):.0f} grados")
    assert s.ctl.inclinacion > 0.3, "subir la mano = la superficie sube (se ve el sur)"

    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", y=0.30)], 0.5)
    s.correr(swipe_v(0.30, 0.75, 0.25), 0.3)                  # swipe hacia ABAJO
    s.correr(lambda t: [DER("abierta", y=0.75)], 1.0)
    print(f"  swipe abajo: inclinacion {math.degrees(s.ctl.inclinacion):.0f} grados")
    assert s.ctl.inclinacion < -0.3, "bajar la mano = se ve el norte"


def test_swipe_horizontal_no_inclina():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.35)], 0.5)
    s.correr(lambda t: [DER("abierta", x=lerp(0.35, 0.75, t / 0.25), y=lerp(Y0, 0.50, t / 0.25))], 0.3)
    assert s.ctl.omega > 1.0 and abs(s.ctl.omega_v) < 0.05
    # diagonal a 45 grados (fisicos: la camara es 4:3): no es de ningun eje, no hace nada
    dy = 0.30 * CAM_W / CAM_H
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", x=0.40, y=0.30)], 0.5)
    s.correr(lambda t: [DER("abierta", x=lerp(0.40, 0.70, t / 0.2), y=lerp(0.30, 0.30 + dy, t / 0.2))], 0.25)
    print(f"  diagonal: omega {s.ctl.omega:.2f}  omega_v {s.ctl.omega_v:.2f}")
    assert abs(s.ctl.omega) < 0.05 and abs(s.ctl.omega_v) < 0.05


def test_agarre_inclina_y_tope_en_polos():
    s = Sim(activo=True)
    s.correr(lambda t: [DER("abierta", y=0.6)], 0.4)
    s.correr(lambda t: [DER("puno", y=0.6)], 0.3)
    s.correr(lambda t: [DER("puno", y=lerp(0.6, 0.45, t / 0.5))], 0.6)    # sube 0.15 = 18.75 % de alto
    R = min(max(C.RADIO_BASE_REL * C.ALTO, C.RADIO_GIRO_MIN), C.RADIO_GIRO_MAX)
    esperado = 0.15 / 0.8 * C.ALTO / R
    print(f"  arrastrar hacia arriba: {s.ctl.inclinacion:.2f} rad (esperado ~{esperado:.2f})")
    assert abs(s.ctl.inclinacion - esperado) < 0.12
    # lanzarla hacia arriba con fuerza: llega al polo y se detiene ahi
    s.correr(lambda t: [DER("puno", y=lerp(0.45, 0.85, t / 0.4))], 0.5)
    s.correr(lambda t: [DER("puno", y=lerp(0.85, 0.2, t / 0.2))], 0.15)
    s.correr(lambda t: [DER("abierta", y=0.2)], 0.2)
    s.correr(lambda t: [], 2.0)
    lim = math.radians(C.INCLINACION_MAX)
    print(f"  tras lanzarla: inclinacion {math.degrees(s.ctl.inclinacion):.0f} grados, omega_v {s.ctl.omega_v:.2f}")
    assert abs(s.ctl.inclinacion - lim) < 1e-6 and s.ctl.omega_v == 0.0


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


# ----------------------------------------------------------------------
# Espacio: efemerides, camara en orbita, dia/noche, Sol y Luna
# ----------------------------------------------------------------------
def _fecha(*a):
    from datetime import datetime, timezone
    return datetime(*a, tzinfo=timezone.utc)


def test_efemerides():
    from universo import astros as A
    # eclipse total del 8 de abril de 2024, maximo sobre Durango (25.29 N, 104.14 O)
    jd = A.dia_juliano(_fecha(2024, 4, 8, 18, 17, 20))
    sol, _ = A.sol(jd)
    luna = A.luna(jd)
    obs = A.punto_superficie(25.29, -104.14)
    hacia_luna = (luna - obs) / np.linalg.norm(luna - obs)
    sep = math.degrees(math.acos(float(np.clip(hacia_luna @ sol, -1, 1))))
    print(f"  eclipse 2024: Sol y Luna separados {sep:.2f} grados vistos desde Durango")
    assert sep < 0.35
    peri = A.sol(A.dia_juliano(_fecha(2025, 1, 4, 13, 28)))[1] / 1e6
    afel = A.sol(A.dia_juliano(_fecha(2025, 7, 3, 19, 55)))[1] / 1e6
    print(f"  Sol: perihelio {peri:.2f}, afelio {afel:.2f} millones de km")
    assert abs(peri - 147.10) < 0.05 and abs(afel - 152.09) < 0.05
    s = A.sol(A.dia_juliano(_fecha(2024, 6, 20, 20, 51)))[0]
    assert abs(math.degrees(math.asin(s[2])) - 23.44) < 0.05        # solsticio: Sol sobre el tropico
    dist = [np.linalg.norm(A.luna(A.dia_juliano(_fecha(2026, 11, d, 0)))) * A.RADIO_TIERRA_KM
            for d in range(1, 31)]
    print(f"  Luna en noviembre 2026: {min(dist):,.0f} a {max(dist):,.0f} km")
    assert 355_000 < min(dist) < 370_000 and 400_000 < max(dist) < 407_500


def test_camara_orbita():
    from universo import astros as A
    f = 1000.0
    giro, incl = math.radians(99), math.radians(-20)
    d = A.distancia_para_radio(173.0, f)
    cam = A.Camara(giro, incl, d, f, 640, 360)
    x, y, z = cam.proyectar_punto(np.zeros(3))
    assert abs(x - 640) < 1e-6 and abs(y - 360) < 1e-6 and abs(z - d) < 1e-9
    assert abs(A.radio_aparente_px(1.0, d, f) - 173.0) < 1e-6
    # la camara esta sobre lon = -giro, lat = -inclinacion (lo mismo que dibuja la Tierra y el HUD)
    centro = cam.a_camara(A.punto_superficie(20.0, -99.0))
    assert np.allclose(centro, [0, 0, 1], atol=1e-9)
    norte = cam.a_camara(A.punto_superficie(30.0, -99.0))
    este = cam.a_camara(A.punto_superficie(20.0, -89.0))
    assert norte[1] < 0 and este[0] > 0                             # norte arriba, este a la derecha


def test_estrellas_siguen_a_la_camara():
    """Al orbitar, la camara gira para seguir viendo la Tierra: el fondo se desliza al lado
    CONTRARIO de la superficie de enfrente (como al caminar alrededor de un arbol mirandolo)."""
    from universo import astros as A
    f = 1000.0
    d = A.distancia_para_radio(173.0, f)
    a = A.Camara(1.0, 0.0, d, f, 640, 360)
    b = A.Camara(1.05, 0.0, d, f, 640, 360)
    estrella = -a.M[2] + 0.1 * a.M[0]                             # algo detras de la Tierra
    estrella /= np.linalg.norm(estrella)
    xa, _, _ = a.proyectar_direccion(estrella)
    xb, _, _ = b.proyectar_direccion(estrella)
    p = A.punto_superficie(0.0, -math.degrees(1.0))               # el punto de la Tierra al centro
    sa, _, _ = a.proyectar_punto(p)
    sb, _, _ = b.proyectar_punto(p)
    print(f"  giro +0.05 rad: estrella {xb - xa:+.1f} px, superficie {sb - sa:+.1f} px")
    assert sb - sa > 5 and xb - xa < -5
    assert abs((xb - xa) + f * 0.05) < 2                           # las estrellas: rotacion pura de la vista


def test_dia_noche_y_luces():
    from universo import astros as A
    from universo.render import Escena
    esc = Escena()
    est = Control().instantanea(0.0)
    fecha = _fecha(2026, 11, 20, 19, 0)                            # 00:30 en la India, 13:00 en Mexico
    sol = A.sol(A.dia_juliano(fecha))[0]
    lat_s, lon_s = math.asin(sol[2]), math.atan2(sol[1], sol[0])

    def centro(giro, incl):
        img = esc.componer(0.0, giro, est, {}, None, False, False, 0.0, incl, fecha)
        return img[350:370, 630:650].astype(float).mean(), img

    dia, _ = centro(-lon_s, -lat_s)                                # camara sobre el punto subsolar
    noche, _ = centro(-lon_s + math.pi, lat_s)                     # sobre el antisolar
    print(f"  brillo al centro: dia {dia:.0f}, noche {noche:.0f}")
    assert dia > 70 and noche < 0.4 * dia
    assert esc.astros["sol_px"] is None, "con la Tierra enfrente, el Sol queda tapado"

    _, img = centro(math.radians(-78), math.radians(-22))          # India de noche
    disco = img[360 - 160:360 + 160, 640 - 160:640 + 160].astype(int)
    luces = int(((disco[..., 2] > 150) & (disco[..., 1] > 110) & (disco[..., 0] < disco[..., 2])).sum())
    print(f"  pixeles de luces de ciudades sobre la India de noche: {luces}")
    assert luces > 40


def test_seleccion_de_astro():
    from universo.viaje import Selector
    sel = Selector()
    obj = {"tierra": (640, 360, 170), "luna": (1000, 300, 4)}
    t, elegido, vista = 0.0, None, None
    while t < C.SELECCION_SOSTENER - 0.1:                         # palma sobre la Luna (pequena: margen)
        vista, elegido = sel.actualizar([(1, 1020, 310)], obj, "tierra", t)
        t += 1 / 60
    assert elegido is None and 0.8 < vista[2] < 1.0
    sel.actualizar([], obj, "tierra", t)                           # un parpadeo del rastreo no reinicia
    for _ in range(12):
        t += 1 / 60
        vista, e = sel.actualizar([(1, 1020, 310)], obj, "tierra", t)
        elegido = elegido or e
    assert elegido == "luna"
    sel = Selector()                                               # el astro donde ya estas no se elige
    for k in range(120):
        _, e = sel.actualizar([(1, 640, 360)], obj, "tierra", k / 60)
        assert e is None


def test_viaje_continuo():
    from universo.viaje import Viaje
    pos0, tierra, luna = np.array([5.8, 0, 0]), np.zeros(3), np.array([-40.0, 45.0, 3.0])
    v = Viaje("luna", 10.0, pos0, tierra, luna, 0.2727, 1.6)
    prev, hipers = None, []
    for k in range(int(C.VIAJE_DURACION * 60) + 1):
        p, objetivo, h, arriba = v.camara(10.0 + k / 60)
        hipers.append(h)
        assert abs(np.linalg.norm(arriba) - 1.0) < 1e-9
        if prev is not None:
            assert np.linalg.norm(p - prev) < 3.0, "sin saltos (no es teletransporte)"
        prev = p
    assert np.allclose(v.camara(10.0)[0], pos0)
    assert abs(np.linalg.norm(prev - luna) - 1.6) < 1e-6 and v.terminado(10.0 + C.VIAJE_DURACION)
    assert hipers[0] < 0.01 and hipers[-1] < 0.05 and max(hipers) > 0.95
    s = Sim()
    s.ctl.colocar(1.0, -0.4)
    assert s.ctl.giro == 1.0 and s.ctl.inclinacion == -0.4 and s.ctl.omega == 0.0


def test_marte_y_planetas():
    """Efemerides de los planetas: oposiciones y conjunciones reales de Marte, y el orden del sistema."""
    from universo import astros as A
    from universo import sistema as S

    def geo(*f):
        jd = A.dia_juliano(_fecha(*f))
        g = S.geocentrica("marte", jd)
        sol = A.sol_posicion(jd)
        elong = math.degrees(math.acos(float(g @ sol / np.linalg.norm(g) / np.linalg.norm(sol))))
        return np.linalg.norm(g) / S.UA_RT, elong
    d, el = geo(2025, 1, 16, 0)                                    # oposicion de Marte
    print(f"  oposicion 16-ene-2025: {d:.3f} UA, elongacion {el:.0f} grados (real: 0.64 UA)")
    assert abs(d - 0.642) < 0.01 and el > 170
    d, el = geo(2022, 12, 8, 0)
    assert abs(d - 0.545) < 0.01 and el > 170
    d, el = geo(2026, 1, 9, 0)                                     # conjuncion con el Sol
    print(f"  conjuncion 9-ene-2026: {d:.2f} UA, elongacion {el:.1f} grados")
    assert el < 3 and 2.3 < d < 2.5
    jd = A.dia_juliano(_fecha(2026, 11, 22, 0))
    radios = [np.linalg.norm(S.heliocentrica(n, jd)) for n in S.ORDEN]
    assert radios == sorted(radios), "los planetas estan en su orden: Mercurio ... Neptuno"
    assert abs(radios[2] - 1.0) < 0.02 and 1.38 < radios[3] < 1.67 and 29.5 < radios[7] < 30.3
    for n in S.ORDEN:                                              # la orbita pasa por el planeta
        o = S.orbita(n, jd, 720)
        assert np.min(np.linalg.norm(o - S.heliocentrica(n, jd), axis=1)) < 0.02 * np.linalg.norm(S.heliocentrica(n, jd))


def test_rotacion_de_astros():
    from universo import astros as A
    from universo import sistema as S
    jd = A.dia_juliano(_fecha(2026, 11, 22, 0))
    for n in ("sol", "marte"):
        R = S.rotacion(n, jd)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12
    # el polo de Marte (en el marco ecuatorial) esta donde dice la IAU: AR 317.7, Dec 52.9
    polo = A.ecuatorial_a_tierra(jd).T @ S.rotacion("marte", jd).T @ np.array([0.0, 0.0, 1.0])
    assert abs(math.degrees(math.asin(polo[2])) - 52.87) < 0.1
    assert abs(math.degrees(math.atan2(polo[1], polo[0])) % 360 - 317.65) < 0.1
    # Marte gira una vuelta (sidereal) en 24.623 h: tras ese tiempo el meridiano 0 vuelve a apuntar igual
    R1, R2 = S.rotacion("marte", jd), S.rotacion("marte", jd + 24.6229 / 24.0)
    cuerpo = lambda R, jd_: R @ A.ecuatorial_a_tierra(jd_) @ np.array([1.0, 0, 0])
    d = math.degrees(math.acos(float(np.clip(cuerpo(R1, jd) @ cuerpo(R2, jd + 24.6229 / 24.0), -1, 1))))
    print(f"  Marte tras 24.623 h: el mismo punto del cielo gira {d:.2f} grados")
    assert d < 0.2
    # la camara "en astro" deja el polo del astro arriba en la pantalla
    f = 1000.0
    cam = A.Camara.en_astro(0.7, -0.3, 6.0, f, 640, 360, np.array([100.0, 50.0, -20.0]), S.rotacion("marte", jd))
    polo_ecef = S.rotacion("marte", jd).T @ np.array([0.0, 0.0, 1.0])
    x, y, _ = cam.proyectar_punto(np.array([100.0, 50.0, -20.0]) + polo_ecef)
    assert abs(x - 640) < 1e-6 and y < 360 - 100
    g, i, rod = A.orientacion_vista(cam.M, S.rotacion("marte", jd))
    assert abs(g - 0.7) < 1e-9 and abs(i + 0.3) < 1e-9 and abs(rod) < 1e-9


def test_dibujo_con_camara_rodada():
    """Las normales de la esfera con (giro, inclinacion, rodar) coinciden con la geometria directa:
    asi un astro se ve bien orientado desde CUALQUIER camara, no solo con su norte arriba."""
    import cv2
    from universo import astros as A
    from universo.tierra import Tierra, tablas_esfera
    tabla, _, _ = tablas_esfera(200)
    normales = tuple(cv2.split(tabla))
    dentro = tabla[..., 2] > 0.3
    n = tabla[dentro].astype(np.float64)
    rng = np.random.default_rng(3)
    peor = 0.0
    for _ in range(8):
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        Q *= np.sign(np.linalg.det(Q))
        M = np.stack([Q[0], -Q[1], Q[2]])                          # ejes de camara: derecha, abajo, atras
        P = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        P *= np.sign(np.linalg.det(P))                             # rotacion marco Tierra -> astro
        g, i, rod = A.orientacion_vista(M, P)
        lon_cam, colat = Tierra._latlon(normales, i, rod)
        lon = (lon_cam[dentro] - g) % (2 * math.pi)
        nb = n @ (M @ P.T)                                         # normal de cada pixel, en el marco del astro
        lon_esp = np.arctan2(nb[:, 1], nb[:, 0]) % (2 * math.pi)
        col_esp = np.arccos(np.clip(nb[:, 2], -1, 1))
        d_lon = np.abs((lon - lon_esp + math.pi) % (2 * math.pi) - math.pi) * np.sin(col_esp)
        peor = max(peor, float(d_lon.max()), float(np.abs(colat[dentro] - col_esp).max()))
    print(f"  error maximo de orientacion: {peor:.1e} rad")
    assert peor < 2e-3


def test_sol_y_marte_en_escena():
    import cv2
    from universo import astros as A
    from universo.render import Escena
    esc = Escena()
    est = Control().instantanea(0.0)
    fecha = _fecha(2026, 11, 22, 0, 33)
    time.sleep(1.5)                                                # termina de hornear la luz de Marte
    def cuadro(foco, giro=1.0, incl=0.0, le=0.0):
        for k in range(2):
            img = esc.componer(le, giro, est, {}, None, False, False, 0.0, incl, fecha, foco)
        return img
    # desde la Tierra: Marte es un punto en su direccion real, a la distancia real
    cuadro("tierra", math.radians(99))
    a = esc.astros
    jd = A.dia_juliano(fecha)
    marte = A.np.asarray(__import__("universo.sistema", fromlist=["x"]).geocentrica("marte", jd))
    print(f"  Marte visto desde la Tierra: {a['dist_km']['marte'] / 1e6:.0f} millones de km")
    assert abs(a["dist_km"]["marte"] - np.linalg.norm(marte) * A.RADIO_TIERRA_KM) < 10
    assert 150e6 < a["dist_km"]["marte"] < 400e6 and "marte" in a["objetivos"]
    # orbitando Marte y el Sol: son esferas de verdad (con textura), centradas y del tamano pedido
    for foco in ("marte", "sol"):
        img = cuadro(foco, math.radians(40), math.radians(-25))
        a = esc.astros
        x, y, r = a["objetivos"][foco]
        assert abs(x - 640) < 1 and abs(y - 360) < 1 and abs(r - esc.radio_base) < 1.0
        disco = img[int(y - r * 0.6):int(y + r * 0.6), int(x - r * 0.6):int(x + r * 0.6)].astype(float)
        print(f"  {foco}: brillo medio del disco {disco.mean():.0f}, variacion {disco.std():.0f} (textura)")
        assert disco.max() > 60 and disco.std() > 3
        cv2.imwrite(str(SALIDA / f"render_{foco}.png"), img)
        assert a["altitud_km"] > 0
    # el Sol desde la Tierra mide lo mismo que el Sol real (0.267 grados de radio)
    s = A.sol(jd)[0]                                               # camara casi opuesta al Sol: queda detras, al lado de la Tierra
    e = np.array([0.0, 0.0, 1.0]) - s[2] * s
    c = -s * math.cos(math.radians(15)) + e / np.linalg.norm(e) * math.sin(math.radians(15))
    g, i = A.giro_inclinacion(c)
    cuadro("tierra", g, i)
    _, _, r_sol = esc.astros["objetivos"].get("sol", (0, 0, 0))
    esperado = esc.focal * math.tan(math.radians(C.SOL_ANGULO_RADIO_GRADOS))
    print(f"  radio aparente del Sol desde la Tierra: {r_sol:.2f} px (esperado {esperado:.2f})")
    assert abs(r_sol - esperado) < 0.4
    cv2.imwrite(str(SALIDA / "render_tierra_con_marte.png"), cuadro("tierra", math.radians(99)))


def test_vista_sistema_solar():
    import cv2
    from universo import astros as A
    from universo import sistema as S
    from universo.render import Escena
    esc = Escena()
    est = Control().instantanea(0.0)
    fecha = _fecha(2026, 11, 22, 0, 33)
    jd = A.dia_juliano(fecha)
    # desde arriba (inclinacion de la vista = 0): las distancias en pantalla son las del plano de la eclíptica
    incl = -math.radians(C.SISTEMA_INCLINACION)
    for k in range(3):                                             # (los primeros cuadros preparan tablas y luz)
        img = esc.componer(0.0, 0.0, est, {}, None, False, False, 0.0, incl, fecha, "tierra", None, None, "sistema")
    t0 = time.perf_counter()
    for k in range(5):
        img = esc.componer(0.0, 0.0, est, {}, None, False, False, 0.0, incl, fecha, "tierra", None, None, "sistema")
    ms = (time.perf_counter() - t0) / 5 * 1000
    a = esc.astros
    assert a["modo"] == "sistema" and set(a["objetivos"]) == {"sol", "tierra", "marte"}
    px_ua = a["px_ua"]
    sx, sy, _ = a["objetivos"]["sol"]
    for n in ("tierra", "marte"):
        x, y, _ = a["objetivos"][n]
        P = S.heliocentrica(n, jd)
        print(f"  {n}: {math.hypot(x - sx, y - sy) / px_ua:.3f} UA en pantalla (real {np.hypot(P[0], P[1]):.3f})")
        assert abs(math.hypot(x - sx, y - sy) / px_ua - np.hypot(P[0], P[1])) < 0.01
        assert abs(math.atan2(-(y - sy), x - sx) - math.atan2(P[1], P[0])) < 0.01    # y tambien el angulo: el norte eclíptico arriba
    # el zoom por omision encuadra la orbita del planeta activo mas lejano (Marte, afelio 1.67 UA)
    assert abs(px_ua * 1.666 - C.SISTEMA_RADIO_REL * esc.h) < 1.0
    # las orbitas dibujadas pasan por los planetas: hay pixeles de color de Marte bajo su icono
    print(f"  vista del sistema: {ms:.1f} ms por cuadro")
    assert ms < 40
    cv2.imwrite(str(SALIDA / "render_sistema_solar.png"), img)
    img = esc.componer(math.log(0.25), 0.8, est, {}, None, False, False, 9.0, 0.3, fecha, "tierra", None, None, "sistema")
    cv2.imwrite(str(SALIDA / "render_sistema_solar_lejos.png"), img)


def test_cambio_de_vista_en_la_aplicacion():
    """Tecla S: entra y sale de la vista del sistema conservando donde estabas; elegir Marte desde el
    sistema viaja a Marte y llega orbitandolo con el norte arriba y el zoom x1."""
    from main import Aplicacion, Captura
    app = Aplicacion(Captura(FuenteFalsa().leer), DetectorGuion())
    app.control.colocar(1.234, -0.3)
    app.control.fijar_escala(math.log(2.0))
    app.paso_render()
    app.alternar_sistema()
    assert app.modo == "sistema" and abs(math.exp(app.control.log_escala) - 1.0) < 1e-6
    app.paso_render()
    assert app.escena.astros["modo"] == "sistema"
    app.alternar_sistema()
    assert app.modo == "astro" and abs(app.control.giro - 1.234) < 1e-9 and abs(math.exp(app.control.log_escala) - 2.0) < 1e-6
    app.alternar_sistema()
    app.paso_render()
    t = time.perf_counter()
    app._viajar("marte", t)
    assert app.modo == "astro" and app.viaje is not None and app.viaje.nombre == "marte"
    app.paso_render()
    app.viaje.t0 = t - C.VIAJE_DURACION - 1                       # ya termino
    app.paso_render()
    assert app.foco == "marte" and app.viaje is None
    from universo import astros as A
    f = app.escena.camara(app.suave.log_escala, app.suave.giro, app.suave.inclinacion, "marte",
                          A.dia_juliano(__import__("universo.render", fromlist=["x"]).fecha_actual()))
    x, y, r = app.escena.astros["objetivos"]["marte"]
    print(f"  llegada a Marte: centro ({x:.0f}, {y:.0f}) radio {r:.0f}px")
    assert abs(x - 640) < 3 and abs(y - 360) < 3 and abs(r - app.escena.radio_base) < 3
    app.tecla(ord("r"))
    assert app.foco == "tierra" and app.modo == "astro"


def test_raton_control():
    """Raton/touchpad: rueda = zoom logaritmico, arrastrar = la superficie sigue al cursor, inercia al soltar."""
    ctl = Control()
    e0 = ctl.log_escala
    ctl.zoom_raton(+3)
    assert abs(ctl.log_escala - e0 - 3 * C.RATON_ZOOM_PASO) < 1e-9
    ctl.zoom_raton(-100)
    assert abs(ctl.log_escala - math.log(C.ESCALA_MIN)) < 1e-9          # tope
    ctl.zoom_raton(+100)
    assert abs(ctl.log_escala - math.log(C.ESCALA_MAX)) < 1e-9
    ctl = Control()
    g0, r = ctl.giro, ctl._radio_giro()
    ctl.raton_agarrar()
    ctl.raton_arrastrar(60.0, -30.0, 1 / 60)                              # cursor a la derecha y arriba
    assert abs(ctl.giro - g0 - 60.0 / r) < 1e-9                          # la superficie sigue al cursor (como el puno)
    assert abs(ctl.inclinacion - 30.0 / r) < 1e-9
    ctl.avanzar(1.0)
    ctl.avanzar(1.5)
    assert abs(ctl.giro - g0 - 60.0 / r) < 1e-9, "arrastrando no hay inercia encima"
    ctl.raton_soltar()
    ctl.avanzar(1.6)
    assert ctl.omega > 0.1 and ctl.giro > g0 + 60.0 / r                  # conserva el impulso
    ctl2 = Control()
    ctl2.raton_agarrar()
    ctl2.raton_arrastrar(50, 0, 1 / 60)
    ctl2.raton_soltar(quieto=True)
    assert ctl2.omega == 0.0                                              # si estaba quieto, se queda
    ctl2.raton_agarrar()
    for _ in range(30):
        ctl2.raton_arrastrar(0, -400, 1 / 60)
    assert abs(ctl2.inclinacion - math.radians(C.INCLINACION_MAX)) < 1e-9  # tope en el polo
    # funciona aunque el control por manos este en pausa
    assert not ctl2.activo


def test_raton_en_la_aplicacion():
    """Clic sobre un astro viaja a el (tambien desde la vista del sistema), rueda = zoom, y un cambio de vista
    ocurrido DENTRO de paso_render ya no deja el suavizado vacio (era el crash del HUD)."""
    import cv2
    from main import Aplicacion, Captura
    app = Aplicacion(Captura(FuenteFalsa().leer), DetectorGuion())
    time.sleep(1.0)
    app.paso_render()
    app.alternar_sistema()
    app.paso_render()
    objetivos = app.escena.astros["objetivos"]
    xm, ym, _ = objetivos["marte"]
    # arrastrar gira la vista; la rueda cambia la escala (px por UA)
    # OpenCV llama al callback con 5 argumentos (evento, x, y, flags, param)
    app.raton(cv2.EVENT_LBUTTONDOWN, 600, 300, 0, None)
    g0 = app.control.giro
    app.raton(cv2.EVENT_MOUSEMOVE, 680, 300, cv2.EVENT_FLAG_LBUTTON)
    app.raton(cv2.EVENT_LBUTTONUP, 680, 300, 0)
    assert abs(app.control.giro - g0) > 0.1 and app.viaje is None
    e0 = app.control.log_escala
    app.raton(cv2.EVENT_MOUSEWHEEL, 640, 360, 120 << 16)
    assert app.control.log_escala > e0
    app.paso_render()
    xm, ym, _ = app.escena.astros["objetivos"]["marte"]
    app.paso_render()
    app.raton(cv2.EVENT_MOUSEMOVE, xm, ym, 0)
    app.paso_render()
    assert app.escena.resaltado is not None and app.escena.resaltado[3] == "marte"
    xm, ym, _ = app.escena.astros["objetivos"]["marte"]
    app.raton(cv2.EVENT_LBUTTONDOWN, xm, ym, 0)
    app.raton(cv2.EVENT_LBUTTONUP, xm, ym, 0)                            # clic sobre Marte
    assert app.viaje is not None and app.viaje.nombre == "marte" and app.modo == "astro"
    app.paso_render()
    # regresion: viajar desde la vista del sistema DENTRO de paso_render (lo hace la palma sobre un astro)
    app2 = Aplicacion(Captura(FuenteFalsa().leer), DetectorGuion())
    app2.paso_render()
    app2.alternar_sistema()
    app2.paso_render()
    app2._elegir_astro = lambda est, t: (app2._viajar("marte", t), None)[1]
    app2.paso_render()
    assert app2.suave.log_escala is not None and app2.viaje is not None
    app2._elegir_astro = lambda est, t: (app2.alternar_sistema(t), None)[1]    # y salir con la palma sobre el astro actual
    app2.viaje = None
    app2.modo = "sistema"
    app2._guardado = (0.0, 1.0, 0.0)
    app2.paso_render()
    assert app2.modo == "astro" and app2.suave.log_escala is not None
    # clic en el vacio no hace nada
    app.viaje = None
    app.modo = "astro"
    app.clic(5, 5)
    assert app.viaje is None


def test_render():
    import cv2
    from universo.control import Estado, ManoVista
    from universo.render import Escena
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
        "inclinada": Estado(True, math.log(1.3), math.radians(99), 0.0,
                            (mv(1, "der", 980, 300, "swipe_v"),), (), False, False, "trazo", False,
                            math.radians(-50), -1.5),
    }
    tiempos = []
    for nombre, est in casos.items():
        for i in range(20):
            a = time.perf_counter()
            f = esc.componer(est.log_escala, est.giro + i * 0.02, est, info, cam, True, True, 0.5)
            tiempos.append(time.perf_counter() - a)
        cv2.imwrite(str(SALIDA / f"render_{nombre}.png"), f)
    med = np.median(tiempos) * 1000
    print(f"  cuadro completo (cielo + Sol + Luna + Tierra + HUD + camara): {med:.1f} ms")
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
