"""Comandos por voz (en espanol, sin internet): microfono -> Vosk -> texto -> comando.

Dos partes:
  - `interpretar(texto)`: convierte una frase en un comando (accion, argumento). Solo biblioteca estandar:
    se prueba sin microfono (pruebas/test_proyecto.py).
  - `Escucha`: hilo que graba del microfono (sounddevice) y reconoce con Vosk, restringido a la gramatica de
    `frases_gramatica()` (reconoce mucho mejor que con vocabulario libre). Si faltan las librerias o el
    modelo, el programa sigue funcionando solo con gestos (`disponible()` dice por que).

Instalar:  pip install vosk sounddevice   y   python herramientas/descargar_voz.py   (baja el modelo en espanol).

Comandos (todos en `FRASES_FIJAS` y las plantillas de abajo; `interpretar` es mas flexible que la gramatica):
  ve a Marte / llevame a la Luna        -> viajar            Marte (a secas)    -> viajar (o responde el quiz)
  compara la Tierra con Marte           -> comparador        cambia el segundo a Venus / otro par / cierra
  quiz / termina el quiz                -> quiz              (con el quiz abierto, di el planeta = respuesta)
  sistema solar / vuelve al planeta     -> vista             gravedad / mas peso / capas / ficha
  mas rapido / hacia atras / ahora      -> tiempo            acercate / alejate / gira a la derecha / detente
  activa el control / pausa el control  -> gestos            silencio / activa la voz / ayuda
  explota la Tierra / destruye el planeta -> explotar         reconstruye la Tierra / repara el planeta
"""

import json
import queue
import re
import threading
import unicodedata
from pathlib import Path

from universo import config as C
from universo import sistema as S

ASTROS = ("sol", "mercurio", "venus", "tierra", "luna", "marte", "jupiter", "saturno", "urano", "neptuno")
_ALIAS = {"yupiter": "jupiter"}                                  # como a veces se transcribe

# palabras (sin acentos ni mayusculas) que usa `interpretar`
_IR = {"ve", "ir", "vamos", "llevame", "lleva", "llevanos", "viaja", "viajar", "visita", "visitar", "muestrame",
       "muestra", "vuelve", "regresa", "volver", "regresar", "acercate"}
_APAGAR = {"cierra", "cerrar", "quita", "quitar", "apaga", "apagar", "oculta", "ocultar", "esconde", "termina",
           "terminar", "desactiva", "desactivar", "sal", "salir", "finaliza", "detener", "deten"}
_COMPARAR = {"compara", "comparar", "comparacion", "comparame", "comparemos"}
_ELEGIR = {"cambia", "cambiar", "pon", "poner", "elige", "usa", "reemplaza", "sustituye"}
_PRIMERO, _SEGUNDO = {"primero", "primer", "izquierdo"}, {"segundo", "derecho"}
_GRAVEDAD = {"gravedad", "peso", "pesaria", "pesarias", "pesar"}
_CAPAS = {"capas", "corte", "transversal", "interior"}
_TIEMPO_ADELANTE = {"acelera", "acelerar", "rapido", "adelanta", "avanza"}
_TIEMPO_ATRAS = {"lento", "despacio", "atras", "retrocede", "retroceder", "desacelera"}
_TIEMPO_PARAR = {"pausa", "pausar", "para", "detente", "deten", "detener", "congela", "congelar"}
_GIRAR = {"gira", "girar", "rota", "rotar", "mueve", "mover", "inclina", "inclinar", "desplaza"}
_DIRECCION = {"izquierda": (-1, 0), "derecha": (1, 0), "arriba": (0, 1), "abajo": (0, -1)}
_ACERCAR = {"acerca", "acercate", "acercar", "cerca", "agranda"}
_ALEJAR = {"aleja", "alejate", "alejar", "lejos", "achica"}
_CONTROL = {"control", "gestos", "gesto", "manos"}
_CONTROL_ON = {"activa", "activar", "enciende", "reanuda", "empieza", "prende"}
_CONTROL_OFF = {"pausa", "pausar", "desactiva", "apaga", "deten", "detener", "para"}
_EXPLOTAR = {"explota", "explotar", "explote", "destruye", "destruir", "revienta", "reventar", "detona", "detonar",
             "boom", "bum", "kaboom"}
_RECONSTRUIR = {"reconstruye", "reconstruir", "restaura", "restaurar", "repara", "reparar", "arregla", "arreglar",
                "rearma", "recupera"}


def normalizar(texto):
    """Minusculas, sin acentos ni signos: lista de palabras."""
    s = unicodedata.normalize("NFKD", texto.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


def _astros_en(palabras):
    """Astros nombrados, en el orden en que se dijeron (sin repetir)."""
    vistos = []
    for p in palabras:
        a = _ALIAS.get(p, p)
        if a in ASTROS and a not in vistos:
            vistos.append(a)
    return vistos


def interpretar(texto):
    """Frase -> (accion, argumento) o None si no es un comando. Acciones:
    voz(bool) ayuda reiniciar ficha(bool) quiz(bool) otro_par cerrar_comparador comparar(par|None)
    comparar_con(astro) elegir((pos, astro)) gravedad(bool) peso(+1|-1) capas(bool)
    tiempo('mas'|'menos'|'ahora'|'parar') control(bool) camara(bool) vista('sistema'|'astro') viajar(astro)
    zoom(+1|-1) girar((dx, dy)) detener astro(astro) cerrar explotar(astro|None) reconstruir(astro|None)"""
    p = normalizar(texto)
    if not p:
        return None
    t = set(p)
    astros = _astros_en(p)
    apagar = bool(t & _APAGAR)

    # --- la propia voz
    if "silencio" in t or ("deja" in t and "escuchar" in t) or ("no" in t and "escuches" in t) \
            or ("voz" in t and apagar):
        return ("voz", False)
    if ("voz" in t and t & {"activa", "enciende", "escucha", "despierta", "prende"}) \
            or ("aster" in t and t & {"escucha", "escuchar", "despierta"}):
        return ("voz", True)
    if t & {"ayuda", "comandos"} or {"que", "puedo", "decir"} <= t:
        return ("ayuda", None)
    if t & {"reinicia", "reiniciar", "reinicio", "reset"} or ("empezar" in t and "nuevo" in t):
        return ("reiniciar", None)

    # --- explosion (demostracion): solo por voz o con el boton rojo escondido
    if t & _EXPLOTAR:
        return ("explotar", astros[0] if astros else None)
    if t & _RECONSTRUIR:
        return ("reconstruir", astros[0] if astros else None)

    # --- ficha educativa
    if t & {"ficha", "datos"}:
        return ("ficha", not apagar)

    # --- quiz
    if t & {"quiz", "cuestionario", "trivia", "examen", "preguntas"}:
        return ("quiz", not apagar)

    # --- comparador
    if (t & {"otro", "otra", "siguiente"}) and (t & {"par", "comparacion"}):
        return ("otro_par", None)
    if t & _COMPARAR:
        if apagar:
            return ("cerrar_comparador", None)
        if len(astros) >= 2:
            return ("comparar", (astros[0], astros[1]))
        if len(astros) == 1:
            return ("comparar_con", astros[0])
        return ("comparar", None)
    if astros and (t & _ELEGIR) and (t & (_PRIMERO | _SEGUNDO)):
        return ("elegir", (0 if t & _PRIMERO else 1, astros[-1]))

    # --- gravedad y capas
    if t & _GRAVEDAD:
        if "peso" in t and t & {"mas", "sube", "aumenta", "aumentar"}:
            return ("peso", +1)
        if "peso" in t and t & {"menos", "baja", "disminuye", "bajar", "reduce"}:
            return ("peso", -1)
        return ("gravedad", not apagar)
    if t & _CAPAS:
        return ("capas", not apagar)

    # --- tiempo
    if "tiempo" in t and t & _TIEMPO_PARAR:
        return ("tiempo", "parar")
    if t & {"ahora", "hoy"} or ("real" in t and "tiempo" in t):
        return ("tiempo", "ahora")
    if t & _TIEMPO_ADELANTE:
        return ("tiempo", "mas")
    if t & _TIEMPO_ATRAS:
        return ("tiempo", "menos")

    # --- control por gestos y camara
    if t & _CONTROL:
        if t & _CONTROL_ON:
            return ("control", True)
        if t & _CONTROL_OFF:
            return ("control", False)
    if "camara" in t:
        return ("camara", not apagar)

    # --- vistas
    if "sistema" in t:
        return ("vista", "astro" if t & {"sal", "salir", "fuera"} else "sistema")
    if "planeta" in t and not astros:
        return ("vista", "astro")

    # --- viajar
    if astros and (t & _IR):
        return ("viajar", astros[0])

    # --- moverse con la voz
    if t & _ACERCAR:
        return ("zoom", +1)
    if t & _ALEJAR:
        return ("zoom", -1)
    if t & {"detente", "quieto", "alto"} or ("girar" in t and t & {"deja", "para", "no"}):
        return ("detener", None)
    if (t & _GIRAR) or (t & set(_DIRECCION)):
        dx = sum(_DIRECCION[w][0] for w in p if w in _DIRECCION)
        dy = sum(_DIRECCION[w][1] for w in p if w in _DIRECCION)
        return ("girar", (dx, dy) if (dx or dy) else (1, 0))

    # --- un astro a secas: ir a el (o responder el quiz)
    if astros:
        return ("astro", astros[0])
    if apagar:
        return ("cerrar", None)
    return None


# ----------------------------------------------------------------------
# Gramatica para Vosk: todas las frases que se esperan (lo demas sale como [unk] y se ignora)
# ----------------------------------------------------------------------
FRASES_FIJAS = [
    "sistema solar", "vista del sistema solar", "muestra el sistema solar", "muéstrame el sistema solar",
    "sal del sistema solar", "vuelve al planeta", "regresa al planeta", "vista de planeta",
    "quiz", "inicia el quiz", "empieza el quiz", "termina el quiz", "cierra el quiz", "salir del quiz",
    "gravedad", "muestra la gravedad", "cuánto pesaría aquí", "quita la gravedad", "cierra la gravedad",
    "más peso", "menos peso", "sube el peso", "baja el peso",
    "capas", "muestra las capas", "corte transversal", "quita las capas", "oculta las capas",
    "más rápido", "acelera el tiempo", "más lento", "más despacio", "hacia atrás", "retrocede el tiempo",
    "tiempo real", "ahora", "pausa el tiempo", "congela el tiempo", "fecha de hoy",
    "acércate", "más cerca", "aléjate", "más lejos", "acerca", "aleja",
    "gira", "gira a la izquierda", "gira a la derecha", "gira hacia arriba", "gira hacia abajo",
    "inclina hacia arriba", "inclina hacia abajo", "detente", "quieto", "deja de girar",
    "ficha", "muestra la ficha", "oculta la ficha", "datos del planeta",
    "activa el control", "activa los gestos", "pausa el control", "pausa los gestos", "activa las manos",
    "reinicia", "empezar de nuevo", "muestra la cámara", "oculta la cámara",
    "ayuda", "qué puedo decir", "comandos", "silencio", "deja de escuchar", "activa la voz", "áster escucha",
    "otro par", "siguiente par", "otra comparación", "cierra la comparación", "cierra", "cerrar", "compara",
    "explota el planeta", "destruye el planeta", "haz explotar el planeta", "boom",
    "reconstruye el planeta", "repara el planeta", "restaura el planeta",
]


def _con_articulo(a):
    art = S.ARTICULOS.get(a, "")
    return f"{art} {S.NOMBRES[a].lower()}".strip()


def _hacia(a):
    art = S.ARTICULOS.get(a, "")
    n = S.NOMBRES[a].lower()
    return f"al {n}" if art == "el" else (f"a {art} {n}" if art else f"a {n}")


def frases_gramatica():
    """Lista de frases para el reconocedor (con acentos, como las escribe el modelo)."""
    f = list(FRASES_FIJAS)
    for a in ASTROS:
        n, na = S.NOMBRES[a].lower(), _con_articulo(a)
        f += [n, na, f"es {na}", f"la respuesta es {na}", f"creo que es {na}", f"responde {na}"]
        f += [f"{v} {_hacia(a)}" for v in ("ve", "vamos", "llévame", "viaja", "vuelve", "regresa", "acércate")]
        f += [f"ir {_hacia(a)}", f"visita {na}", f"muéstrame {na}", f"compara {na}"]
        f += [f"explota {na}", f"destruye {na}", f"haz explotar {na}", f"reconstruye {na}", f"repara {na}"]
        f += [f"cambia el primero a {na}", f"cambia el segundo a {na}", f"pon {na} en el primero",
              f"pon {na} en el segundo", f"cambia el segundo por {na}"]
        for b in ASTROS:
            if a != b:
                f += [f"compara {na} con {_con_articulo(b)}", f"compara {na} y {_con_articulo(b)}"]
    unicas = list(dict.fromkeys(f))
    return unicas + ["[unk]"]


# ----------------------------------------------------------------------
# Escucha del microfono
# ----------------------------------------------------------------------
def ruta_modelo():
    """Carpeta del modelo. Si el zip se descomprimio con una carpeta de mas (modelos/vosk-es/vosk-model-.../am), la baja."""
    base = Path(C.RAIZ) / C.VOZ_MODELO
    if not (base / "am").is_dir() and base.is_dir():
        for hija in base.iterdir():
            if (hija / "am").is_dir():
                return hija
    return base


def disponible():
    """(True, '') si hay librerias y modelo; si no (False, motivo en espanol)."""
    try:
        import sounddevice  # noqa: F401
        import vosk  # noqa: F401
    except ImportError as e:
        return False, f"falta instalar {e.name} (pip install vosk sounddevice)"
    if not (ruta_modelo() / "am").is_dir():
        return False, (f"no encuentro el modelo de voz en {ruta_modelo()} (debe tener las carpetas am, conf, graph...; "
                       f"corre herramientas/descargar_voz.py)")
    return True, ""


def listar_microfonos():
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


class Escucha(threading.Thread):
    """Hilo: microfono -> Vosk. Las frases reconocidas quedan en `cola` (la lee el hilo principal)."""

    def __init__(self, microfono=None):
        super().__init__(daemon=True, name="voz")
        self.microfono = microfono
        self.cola = queue.Queue()
        self.estado = "iniciando"            # iniciando | escuchando | error
        self.error = None
        self.nivel = 0.0                     # 0..1: volumen del microfono (para ver si de verdad te oye)
        self.oido = ""                       # ultima frase reconocida
        self._parar = threading.Event()

    def detener(self):
        self._parar.set()

    def frases(self):
        """Frases reconocidas desde la ultima vez (no bloquea)."""
        out = []
        while True:
            try:
                out.append(self.cola.get_nowait())
            except queue.Empty:
                return out

    def run(self):
        try:
            import numpy as np
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)
            modelo = Model(str(ruta_modelo()))
            audio = queue.Queue()

            def al_grabar(datos, cuadros, tiempo, estado):
                b = bytes(datos)
                audio.put(b)
                x = np.frombuffer(b, np.int16)
                if x.size:
                    self.nivel = min(1.0, float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) / 3000.0)

            def abrir(frecuencia):
                return sd.RawInputStream(samplerate=frecuencia, blocksize=int(frecuencia / 4), device=self.microfono,
                                         dtype="int16", channels=1, callback=al_grabar)

            frecuencia = C.VOZ_FRECUENCIA
            try:
                flujo = abrir(frecuencia)
            except Exception:                                    # el microfono no acepta 16 kHz: usa el suyo
                frecuencia = int(sd.query_devices(self.microfono, "input")["default_samplerate"])
                flujo = abrir(frecuencia)
            if C.VOZ_GRAMATICA:
                rec = KaldiRecognizer(modelo, frecuencia, json.dumps(frases_gramatica(), ensure_ascii=False))
            else:
                rec = KaldiRecognizer(modelo, frecuencia)
            with flujo:
                self.estado = "escuchando"
                while not self._parar.is_set():
                    try:
                        datos = audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(datos):
                        texto = json.loads(rec.Result()).get("text", "").replace("[unk]", " ").strip()
                        texto = " ".join(texto.split())
                        if texto:
                            self.oido = texto
                            self.cola.put(texto)
        except Exception as e:                                   # sin microfono, modelo danado, etc.
            self.error = f"{type(e).__name__}: {e}"
            self.estado = "error"
