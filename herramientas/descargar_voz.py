"""Descarga el modelo de voz en espanol (Vosk, 39 MB, licencia Apache 2.0) a modelos/vosk-es (una vez, con internet).

    .venv\\Scripts\\python.exe herramientas\\descargar_voz.py

Antes instala las librerias:  .venv\\Scripts\\python.exe -m pip install vosk sounddevice
Fuente: https://alphacephei.com/vosk/models  (vosk-model-small-es-0.42). Para mas precision puedes bajar el
grande (vosk-model-es-0.42, 1.4 GB) y ponerlo en la misma carpeta.
"""

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
DESTINO = Path(__file__).resolve().parent.parent / "modelos" / "vosk-es"


def main():
    if (DESTINO / "am").is_dir():
        print(f"Ya esta el modelo en {DESTINO}")
        return 0
    DESTINO.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_ruta = Path(tmp) / "modelo.zip"
        print(f"Descargando {URL} ...")
        urllib.request.urlretrieve(URL, zip_ruta)
        with zipfile.ZipFile(zip_ruta) as z:
            z.extractall(tmp)
        carpetas = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if not carpetas:
            print("El zip no trajo ninguna carpeta.")
            return 1
        if DESTINO.exists():
            shutil.rmtree(DESTINO)
        shutil.move(str(carpetas[0]), str(DESTINO))
    print(f"Listo: {DESTINO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
