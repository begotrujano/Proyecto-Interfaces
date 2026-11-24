"""Diagnostico de la voz, sin abrir el programa: muestra tu microfono, el volumen, lo que oye y el comando que saca.

    .venv\\Scripts\\python.exe herramientas\\probar_voz.py [--microfono N] [--libre]

--buscar: prueba cada microfono 2 s y dice cual te oye (habla sin parar mientras corre, unos 20 s).
--libre: vocabulario libre en vez de la gramatica (prueba si el problema es la gramatica).
Habla ("ve a Marte") y mira: la barra de volumen debe moverse; luego aparece "oi: ..." y el comando.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from universo import config as C
from universo import voz as V


def buscar():
    """Graba 2 s en cada entrada y muestra el volumen: el que sube cuando hablas es el bueno."""
    import numpy as np
    import sounddevice as sd
    print("HABLA SIN PARAR (cuenta del 1 al 20, por ejemplo) mientras se prueba cada microfono...\n")
    resultados = []
    for i, nombre in V.listar_microfonos():
        if "altavoz" in nombre.lower() or "speaker" in nombre.lower():
            continue                                         # son salidas de audio, no microfonos
        try:
            info = sd.query_devices(i, "input")
            frec = int(info["default_samplerate"])
            x = sd.rec(int(2 * frec), samplerate=frec, channels=1, dtype="int16", device=i)
            sd.wait()
            nivel = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
            resultados.append((nivel, i, nombre))
            print(f"  {i:>2}: volumen {nivel:7.0f} {'#' * min(40, int(nivel / 100))}  {nombre}")
        except Exception as e:
            print(f"  {i:>2}: no se pudo abrir ({type(e).__name__})  {nombre}")
    if resultados:
        nivel, i, nombre = max(resultados)
        print(f"\nEl que mas te oyo: {i} ({nombre}). Prueba:  python herramientas\\probar_voz.py --microfono {i}")
        if nivel < 150:
            print("Ninguno oyo nada: revisa Configuracion de Windows > Privacidad > Microfono (permitir apps de escritorio) "
                  "y que el microfono no este silenciado.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--microfono", type=int, default=None)
    ap.add_argument("--libre", action="store_true")
    ap.add_argument("--buscar", action="store_true")
    args = ap.parse_args()
    if args.buscar:
        return buscar()
    ok, motivo = V.disponible()
    print("Librerias y modelo:", "OK" if ok else f"PROBLEMA -> {motivo}")
    if not ok:
        return 1
    print("Modelo:", V.ruta_modelo())
    print("Microfonos de entrada (usa --microfono N para elegir):")
    for i, nombre in V.listar_microfonos():
        print(f"   {i}: {nombre}")
    if args.libre:
        C.VOZ_GRAMATICA = False
    print(f"Modo: {'gramatica' if C.VOZ_GRAMATICA else 'vocabulario libre'}. Cargando el modelo...")
    e = V.Escucha(args.microfono)
    e.start()
    t0 = time.time()
    while e.estado == "iniciando" and time.time() - t0 < 60:
        time.sleep(0.1)
    if e.estado != "escuchando":
        print("ERROR al abrir el microfono o el modelo:", e.error)
        return 1
    print("Escuchando. Di por ejemplo «ve a Marte». Ctrl+C para salir.\n")
    try:
        while True:
            barra = "#" * int(40 * e.nivel)
            print(f"\r volumen [{barra:<40}]", end="", flush=True)
            for frase in e.frases():
                print(f"\n oi: «{frase}»   ->   {V.interpretar(frase)}")
            if e.estado == "error":
                print("\nERROR:", e.error)
                return 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        e.detener()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
