"""Descarga las texturas reales de la NASA a assets/ (hazlo una vez, con internet).

    .venv\Scripts\python.exe herramientas\descargar_texturas.py

Marte: "Mars" de NASA 3D Resources (https://science.nasa.gov/3d-resources/mars), mosaico de las
imagenes de Viking procesadas por el USGS (NASA/JPL-Caltech, dominio publico). Se guarda como
assets/marte_nasa_viking.jpg y el programa la usa en lugar de la textura procedural.

Si al verla los rasgos quedan corridos 180 grados (la longitud 0 no esta al centro de la imagen),
pon DESFASE_TEXTURA_MARTE = 180.0 en universo/config.py.

Sol: no hay un mapa de toda la superficie con licencia clara en un solo archivo; si consigues uno
equirectangular (2:1) de SDO/NASA, guardalo como assets/sol_nasa_sdo.jpg. Mientras tanto se usa la procedural.
"""

import sys
import urllib.request
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DESCARGAS = {
    "marte_nasa_viking.jpg": "https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/image/mars/Mars.jpg",
}


def main():
    ASSETS.mkdir(exist_ok=True)
    for nombre, url in DESCARGAS.items():
        destino = ASSETS / nombre
        print(f"Descargando {nombre} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Universo en tus manos)"})
            datos = urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:
            print(f"  No se pudo: {e}\n  Bajala a mano de {url} y guardala como {destino}")
            continue
        destino.write_bytes(datos)
        try:                                           # comprobacion rapida: debe ser 2:1
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
            alto, ancho = img.shape[:2]
            print(f"  listo: {ancho}x{alto}" + ("" if ancho == 2 * alto else "  (no es 2:1; se reescalara al cargar)"))
        except Exception:
            print("  listo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
