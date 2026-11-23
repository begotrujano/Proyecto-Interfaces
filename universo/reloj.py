"""La fecha del cielo, que puede correr mas rapido, hacia atras o detenerse.

Solo depende de la libreria estandar. Con el reloj acelerado se ven moverse los planetas en sus
orbitas, cambiar la fase de Venus, girar sus nubes, el dia y la noche pasar sobre la Tierra...

    Teclas (main.py):  .  mas rapido (hacia adelante)   ,  mas lento / hacia atras   T  volver al ahora
"""

from datetime import datetime, timedelta, timezone

from universo import config as C


def fecha_inicial():
    """config.FECHA si esta fijada; si no, ahora mismo (UTC)."""
    if C.FECHA:
        f = datetime.fromisoformat(C.FECHA)
        return f if f.tzinfo else f.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


_UNIDADES = ((30 * 86400, "mes", "meses"), (7 * 86400, "semana", "semanas"), (86400, "día", "días"),
             (3600, "hora", "horas"), (60, "minuto", "minutos"), (1, "segundo", "segundos"))


def texto_velocidad(v):
    """'tiempo real', '1 día por segundo, hacia atrás', 'tiempo detenido'..."""
    if v == 0:
        return "tiempo detenido"
    if v == 1:
        return "tiempo real"
    a = abs(v)
    for seg, uno, varios in _UNIDADES:
        if a >= seg:
            n = a / seg
            txt = f"{n:g} {uno if n == 1 else varios} por segundo"
            break
    return txt + ("" if v > 0 else ", hacia atrás")


class Reloj:
    def __init__(self, t=0.0, fecha=None):
        self._fecha0 = fecha or fecha_inicial()          # fecha en el instante t0 (ancla)
        self._t0 = t
        self.velocidad = 1.0                              # segundos simulados por segundo real
        self.congelado = False                            # (durante un viaje)
        self._indice = C.TIEMPO_PASOS.index(1)

    def fecha(self, t):
        if self.congelado:
            return self._fecha0
        return self._fecha0 + timedelta(seconds=(t - self._t0) * self.velocidad)

    def _anclar(self, t):
        self._fecha0, self._t0 = self.fecha(t), t

    @property
    def en_tiempo_real(self):
        return self.velocidad == 1 and not C.FECHA and abs(
            (self.fecha(self._t0) - datetime.now(timezone.utc)).total_seconds()) < 2 and not self.congelado

    def fijar_velocidad(self, v, t):
        self._anclar(t)
        self.velocidad = float(v)
        if v in C.TIEMPO_PASOS:
            self._indice = C.TIEMPO_PASOS.index(v)

    def paso(self, direccion, t):
        """direccion +1: un escalon mas rapido hacia adelante; -1: hacia atras. Devuelve el texto."""
        self._indice = min(max(self._indice + direccion, 0), len(C.TIEMPO_PASOS) - 1)
        self.fijar_velocidad(C.TIEMPO_PASOS[self._indice], t)
        return texto_velocidad(self.velocidad)

    def ahora(self, t):
        """Vuelve a la fecha inicial (ahora mismo, o config.FECHA) en tiempo real."""
        self._fecha0, self._t0 = fecha_inicial(), t
        self.fijar_velocidad(1, t)

    def congelar(self, t, si=True):
        if si and not self.congelado:
            self._anclar(t)
            self.congelado = True
        elif not si and self.congelado:
            self.congelado = False
            self._t0 = t                                   # sigue desde donde se quedo
