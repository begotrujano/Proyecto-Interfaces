"""Explosion de un astro (demostracion): sirve para cualquiera porque trabaja sobre lo que ya se dibujo.

Al empezar se toma una "foto" del astro tal como se ve (el parche de pantalla y su mascara: lo que cambio
al dibujarlo, asi entran los anillos de Saturno y el halo de la atmosfera). La foto se parte en celdas de
Voronoi con bordes irregulares (coordenadas deformadas con ruido suave) y cada celda es un pedazo.

Linea de tiempo:
  carga    (EXPLOSION_CARGA s): el astro se sigue dibujando, se agrieta (bordes de los pedazos al rojo) y brilla.
  estallido: el astro ya no se dibuja; destello blanco, bola de fuego, onda expansiva inclinada, chispas,
             y los pedazos salen girando, al rojo vivo al principio, frenando (exponencial) y derivando.
  reconstruccion (EXPLOSION_RECONSTRUIR_*): la misma animacion hacia atras, hasta que vuelve el astro real.

Los pedazos se guardan en el tamano de la foto (radio r0) y se dibujan relativos al centro y radio actuales
del astro: si la camara se mueve o hace zoom, los escombros la siguen. Solo numpy y OpenCV; nada por pixel en
Python (un bucle por pedazo, ~40)."""

import math

import cv2
import numpy as np

from universo import config as C

CALIENTE = np.array([40.0, 150.0, 255.0], np.float32)      # BGR naranja incandescente
GRIETA = np.array([30.0, 120.0, 255.0], np.float32)
ONDA = (255, 236, 214)


def _suave(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _resplandor(n=129):
    """Mancha radial (0..1) para la bola de fuego y el brillo de la carga."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    d2 = x * x + y * y
    return (np.exp(-d2 * 5.0) * (d2 < 1.0)).astype(np.float32)


def _sumar(buf, img, x0, y0):
    """buf[y0:, x0:] += img (float32 BGR o 1 canal ya multiplicado por color), recortado a la pantalla."""
    h, w = buf.shape[:2]
    ih, iw = img.shape[:2]
    a0, b0 = max(0, -x0), max(0, -y0)
    a1, b1 = min(iw, w - x0), min(ih, h - y0)
    if a1 <= a0 or b1 <= b0:
        return
    roi = buf[y0 + b0:y0 + b1, x0 + a0:x0 + a1]
    cv2.add(roi, img[b0:b1, a0:a1], dst=roi, dtype=cv2.CV_8U)


class Explosion:
    def __init__(self, semilla=None):
        self.rng = np.random.default_rng(semilla)
        self._glow = _resplandor()
        self.reiniciar()

    def reiniciar(self):
        self.astro = None          # astro que explota (o None)
        self.fase = None           # "espera" (falta la foto) | "explota" | "reconstruye"
        self.t0 = 0.0              # cuando se tomo la foto (empieza la carga)
        self.t_rec = 0.0
        self.q_rec = 0.0
        self.pedazos = []
        self.grietas = None

    # ------------------------------------------------------------------
    @property
    def activa(self):
        return self.astro is not None

    def iniciar(self, astro):
        self.reiniciar()
        self.astro, self.fase = astro, "espera"

    def restaurar(self, t):
        """Vuelve a armar el astro (la animacion al reves). Si aun no estallaba, simplemente se cancela."""
        if self.fase == "explota" and t - self.t0 > C.EXPLOSION_CARGA:
            self.q_rec = self._q_explota(t)
            self.fase, self.t_rec = "reconstruye", t
        elif self.fase != "reconstruye":
            self.reiniciar()

    def actualizar(self, t):
        """Reconstruccion automatica y fin de la animacion."""
        if self.fase == "explota" and C.EXPLOSION_RECONSTRUIR_S is not None \
                and t - self.t0 > C.EXPLOSION_CARGA + C.EXPLOSION_RECONSTRUIR_S:
            self.restaurar(t)
        if self.fase == "reconstruye" and t - self.t_rec >= C.EXPLOSION_RECONSTRUIR_DURACION:
            self.reiniciar()

    def necesita_captura(self, nombre):
        return self.fase == "espera" and nombre == self.astro

    def oculta(self, nombre, t):
        """True si el astro real no se debe dibujar (ya estallo)."""
        if nombre != self.astro:
            return False
        return self.fase == "reconstruye" or (self.fase == "explota" and t - self.t0 >= C.EXPLOSION_CARGA)

    def estallo(self, t):
        return self.fase in ("explota", "reconstruye") and self.oculta(self.astro, t)

    # ------------------------------------------------------------------
    def _q_explota(self, t):
        """Avance de los pedazos (0 = en su lugar; ~1 = ya frenaron)."""
        s = max(t - self.t0 - C.EXPLOSION_CARGA, 0.0)
        return 1.0 - math.exp(-s / C.EXPLOSION_FRENADO) + C.EXPLOSION_DERIVA * s

    def _q(self, t):
        if self.fase == "reconstruye":
            return self.q_rec * (1.0 - _suave((t - self.t_rec) / C.EXPLOSION_RECONSTRUIR_DURACION))
        return self._q_explota(t)

    # ------------------------------------------------------------------
    def capturar(self, parche, mascara, cx, cy, r0, t):
        """parche: BGR uint8 del astro recien dibujado; mascara: bool (pixeles del astro); (cx, cy): centro del
        astro en el parche; r0: su radio en px."""
        self.t0, self.fase = t, "explota"
        rng = self.rng
        H, W = mascara.shape
        r0 = max(float(r0), 4.0)
        # Voronoi en una malla reducida, con las coordenadas deformadas por ruido suave (bordes de roca)
        f = max(1.0, max(H, W) / 320.0)
        hg, wg = max(int(round(H / f)), 2), max(int(round(W / f)), 2)
        yy, xx = np.mgrid[0:hg, 0:wg].astype(np.float32)
        xx = xx * (W / wg) - cx
        yy = yy * (H / hg) - cy
        for celdas, amp in ((5, 0.06 * r0), (17, 0.012 * r0)):        # ondulado + rugoso
            ruido = rng.standard_normal((2, celdas, celdas)).astype(np.float32)
            xx += cv2.resize(ruido[0], (wg, hg), interpolation=cv2.INTER_CUBIC) * amp
            yy += cv2.resize(ruido[1], (wg, hg), interpolation=cv2.INTER_CUBIC) * amp
        n = C.EXPLOSION_FRAGMENTOS
        rad = r0 * 0.98 * np.sqrt(rng.random(n))
        ang = rng.random(n) * 2.0 * math.pi
        semillas = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1).astype(np.float32)
        d = (xx[..., None] - semillas[:, 0]) ** 2 + (yy[..., None] - semillas[:, 1]) ** 2
        etiquetas_g = np.argmin(d, axis=2).astype(np.int32)
        etiquetas = cv2.resize(etiquetas_g, (W, H), interpolation=cv2.INTER_NEAREST)

        # grietas: bordes entre pedazos (brillan durante la carga)
        borde = np.zeros((H, W), np.uint8)
        borde[:, 1:] |= (etiquetas[:, 1:] != etiquetas[:, :-1])
        borde[1:, :] |= (etiquetas[1:, :] != etiquetas[:-1, :])
        borde &= mascara
        k = max(1, int(round(r0 / 160.0)))
        borde = cv2.dilate(borde * 255, np.ones((k, k), np.uint8)) if k > 1 else borde * 255
        self.grietas = (cv2.GaussianBlur(borde, (0, 0), 0.5 + k * 0.4).astype(np.float32) / 255.0, cx, cy, r0)

        self.pedazos = []
        m8 = mascara.astype(np.uint8)
        for i in range(n):
            ys, xs = np.nonzero(etiquetas_g == i)
            if len(xs) == 0:
                continue
            x0 = max(int(xs.min() * W / wg) - 2, 0)
            x1 = min(int((xs.max() + 1) * W / wg) + 2, W)
            y0 = max(int(ys.min() * H / hg) - 2, 0)
            y1 = min(int((ys.max() + 1) * H / hg) + 2, H)
            alfa = ((etiquetas[y0:y1, x0:x1] == i) & (m8[y0:y1, x0:x1] > 0)).astype(np.float32)
            if alfa.sum() < 6:
                continue
            alfa = cv2.GaussianBlur(alfa, (0, 0), 0.7)                   # borde suave (sin escalones)
            spr = (np.ascontiguousarray(parche[y0:y1, x0:x1]), alfa)
            sx, sy = (x1 - x0) / 2.0, (y1 - y0) / 2.0                    # centro del sprite
            ox, oy = (x0 + sx - cx) / r0, (y0 + sy - cy) / r0            # su posicion, en radios
            rho = math.hypot(ox, oy)
            dirx, diry = (ox / rho, oy / rho) if rho > 0.08 else (math.cos(ang[i]), math.sin(ang[i]))
            giro = rng.normal(0.0, 0.35)
            c, s = math.cos(giro), math.sin(giro)
            dirx, diry = dirx * c - diry * s, dirx * s + diry * c
            vmin, vmax = C.EXPLOSION_VELOCIDAD
            vel = rng.uniform(vmin, vmax) * (0.7 + 0.5 * min(rho, 1.2))
            self.pedazos.append((spr, sx, sy, ox, oy, dirx * vel, diry * vel,
                                 rng.uniform(-0.3, 0.7),                 # se acerca/aleja (escala)
                                 rng.uniform(-160.0, 160.0)))            # grados de giro por unidad de avance

        m = C.EXPLOSION_CHISPAS
        rr = np.sqrt(rng.random(m)) * 0.95
        aa = rng.random(m) * 2.0 * math.pi
        self.ch_p0 = np.stack([rr * np.cos(aa), rr * np.sin(aa)], 1).astype(np.float32)
        va = aa + rng.normal(0.0, 0.5, m)
        vv = rng.uniform(1.2, 5.5, m) * (0.4 + rr)
        self.ch_v = np.stack([vv * np.cos(va), vv * np.sin(va)], 1).astype(np.float32)
        mezcla = rng.random(m)[:, None].astype(np.float32)               # blanco -> amarillo -> naranja
        self.ch_col = (np.array([235, 250, 255], np.float32) * (1 - mezcla)
                       + np.array([30, 140, 255], np.float32) * mezcla)
        self.inclinacion_onda = rng.uniform(-25.0, 25.0)

    # ------------------------------------------------------------------
    def dibujar(self, buf, nombre, x, y, r, t):
        """Lo que toca en este cuadro, con el astro en (x, y) de radio r px."""
        if nombre != self.astro or self.fase not in ("explota", "reconstruye"):
            return
        s = r / self.grietas[3]                                          # escala respecto a la foto
        tau = t - self.t0
        if self.fase == "explota" and tau < C.EXPLOSION_CARGA:
            self._carga(buf, x, y, r, s, tau / C.EXPLOSION_CARGA)
            return
        q = self._q(t)
        self._pedazos(buf, x, y, r, s, q)
        self._chispas(buf, x, y, r, q)
        if self.fase == "explota":
            self._estallido(buf, x, y, r, tau - C.EXPLOSION_CARGA)

    def _carga(self, buf, x, y, r, s, u):
        g, cx, cy, r0 = self.grietas
        c = u * u
        if s != 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            cx, cy = cx * s, cy * s
        img = g[..., None] * (GRIETA * min(1.4 * c, 1.0))
        _sumar(buf, img, int(round(x - cx)), int(round(y - cy)))
        self._mancha(buf, x, y, r * (1.0 + 0.5 * c), CALIENTE * (0.9 * c))
        if u > 0.6:                                                       # tiembla un poco antes de estallar
            k = int(round(r * 0.012 * (u - 0.6) / 0.4 * math.sin(u * 90.0)))
            if k:
                h, w = buf.shape[:2]
                x0, x1 = max(int(x - r * 1.2), 0), min(int(x + r * 1.2), w)
                y0, y1 = max(int(y - r * 1.2), 0), min(int(y + r * 1.2), h)
                if x1 - x0 > abs(k) * 2:
                    buf[y0:y1, x0:x1] = np.roll(buf[y0:y1, x0:x1], k, axis=1)

    def _mancha(self, buf, x, y, radio, color):
        """Resplandor aditivo: se arma chico (1/4) solo en la parte visible y se escala (es suave, no se nota)."""
        h, w = buf.shape[:2]
        d = 2.0 * radio
        if d < 3 or float(np.max(color)) < 1.0:
            return
        x0, y0 = max(int(x - radio), 0), max(int(y - radio), 0)
        x1, y1 = min(int(math.ceil(x + radio)), w), min(int(math.ceil(y + radio)), h)
        if x1 <= x0 or y1 <= y0:
            return
        f = 4.0
        gw, gh = max(int((x1 - x0) / f), 2), max(int((y1 - y0) / f), 2)
        n = self._glow.shape[0] - 1                                      # pixeles de la pantalla -> de la mancha
        M = np.array([[n / d * (x1 - x0) / gw, 0, (x0 - (x - radio)) * n / d],
                      [0, n / d * (y1 - y0) / gh, (y0 - (y - radio)) * n / d]], np.float32)
        g = cv2.warpAffine(self._glow, M, (gw, gh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        chica = cv2.merge([g * float(c) for c in color])
        chica = np.clip(chica, 0, 255).astype(np.uint8)
        grande = cv2.resize(chica, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        roi = buf[y0:y1, x0:x1]
        cv2.add(roi, grande, dst=roi)

    def _pedazos(self, buf, x, y, r, s, q):
        h, w = buf.shape[:2]
        calor = min(max(1.0 - q / 0.55, 0.0), 1.0) ** 1.5
        opac = 1.0 - 0.5 * min(max((q - 0.5) / 0.6, 0.0), 1.0)
        caliente = None
        for (bgr, alfa), sx, sy, ox, oy, vx, vy, vz, wgir in self.pedazos:
            px = x + r * (ox + vx * q)
            py = y + r * (oy + vy * q)
            esc = s * (1.0 + vz * q)
            if esc <= 0.02:
                continue
            radio_spr = math.hypot(sx, sy) * esc
            if px + radio_spr < 0 or py + radio_spr < 0 or px - radio_spr >= w or py - radio_spr >= h:
                continue
            if calor > 0.0:                                              # al rojo vivo (en el sprite chico)
                if caliente is None or caliente.shape != bgr.shape:
                    caliente = np.empty_like(bgr)
                    caliente[:] = (CALIENTE * 0.8 * calor).astype(np.uint8)
                bgr = cv2.addWeighted(bgr, 1.0 - 0.6 * calor, caliente, 1.0, 0.0)
            M = cv2.getRotationMatrix2D((sx, sy), wgir * q, esc)       # sprite -> pantalla
            M[0, 2] += px - sx
            M[1, 2] += py - sy
            hs, ws = alfa.shape
            esq = M[:, :2] @ np.array([[0, ws, 0, ws], [0, 0, hs, hs]], np.float64) + M[:, 2:]
            bx0, by0 = max(int(esq[0].min()) - 1, 0), max(int(esq[1].min()) - 1, 0)   # caja ajustada al giro
            bx1, by1 = min(int(esq[0].max()) + 2, w), min(int(esq[1].max()) + 2, h)
            if bx1 <= bx0 or by1 <= by0:
                continue
            M[0, 2] -= bx0
            M[1, 2] -= by0
            tam = (bx1 - bx0, by1 - by0)
            col = cv2.warpAffine(bgr, M, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            a = cv2.warpAffine(alfa, M, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if opac < 1.0:
                a *= opac
            roi = buf[by0:by1, bx0:bx1]
            cv2.blendLinear(col, roi, a, 1.0 - a, dst=roi)

    def _chispas(self, buf, x, y, r, q):
        brillo = min(max(1.2 - q / 0.9, 0.0), 1.0)
        if brillo <= 0.02:
            return
        p = self.ch_p0 + self.ch_v * (q * (1.0 + 0.4 * q))
        xs = np.round(x + p[:, 0] * r).astype(np.int32)
        ys = np.round(y + p[:, 1] * r).astype(np.int32)
        h, w = buf.shape[:2]
        col = (self.ch_col * brillo).astype(np.uint8)
        tam = 2 if r > 60 else 1
        for dx in range(tam):
            for dy in range(tam):
                xx, yy = xs + dx, ys + dy
                ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                buf[yy[ok], xx[ok]] = np.maximum(buf[yy[ok], xx[ok]], col[ok])

    def _estallido(self, buf, x, y, r, s):
        """Destello, bola de fuego y onda expansiva (solo al estallar, no al reconstruirse). s: s desde el estallido."""
        bola = math.exp(-s / 0.7) * 1.3
        if bola > 0.02:
            self._mancha(buf, x, y, r * (1.3 + 2.4 * (1.0 - math.exp(-s / 0.35))),
                         np.array([170.0, 225.0, 255.0], np.float32) * bola)
        onda = 0.8 * math.exp(-s / 0.7)
        if onda > 0.03:
            R = r * (1.1 + C.EXPLOSION_ONDA_VEL * s)
            if R < 4.0 * max(buf.shape[:2]):
                grosor = max(1, int(r * 0.015 * (1.0 + 1.5 * s)))
                col = tuple(int(c * onda) for c in ONDA)
                ejes = (int(R * 16), int(R * 0.3 * 16))
                centro = (int(round(x * 16)), int(round(y * 16)))
                cv2.ellipse(buf, centro, ejes, self.inclinacion_onda, 0, 360, col, grosor, cv2.LINE_AA, 4)
                col2 = tuple(int(c * onda * 0.45) for c in ONDA)
                ejes2 = (int(R * 0.82 * 16), int(R * 0.82 * 0.3 * 16))
                cv2.ellipse(buf, centro, ejes2, self.inclinacion_onda, 0, 360, col2, max(1, grosor // 2),
                            cv2.LINE_AA, 4)
        destello = C.EXPLOSION_DESTELLO * math.exp(-s / 0.14)
        if destello > 0.02:
            cv2.convertScaleAbs(buf, dst=buf, alpha=1.0 - destello, beta=255.0 * destello)
