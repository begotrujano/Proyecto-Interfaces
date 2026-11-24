"""Descarga la tipografia de la interfaz, Exo 2 (Google Fonts, licencia SIL Open Font License), a assets/fuentes.

    .venv\\Scripts\\python.exe herramientas\\descargar_fuentes.py

Sin ella la interfaz usa Segoe UI (Light / Semibold) de Windows, que se parece pero es menos "espacial".
"""

import sys
import urllib.request
from pathlib import Path

DESTINO = Path(__file__).resolve().parent.parent / "assets" / "fuentes"
ARCHIVOS = {
    "Exo2[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/exo2/Exo2%5Bwght%5D.ttf",
    "OFL.txt": "https://raw.githubusercontent.com/google/fonts/main/ofl/exo2/OFL.txt",
}


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, url in ARCHIVOS.items():
        ruta = DESTINO / nombre
        if ruta.exists():
            print(f"Ya esta: {ruta}")
            continue
        print(f"Descargando {nombre} ...")
        urllib.request.urlretrieve(url, ruta)
    print("Listo. Vuelve a abrir el programa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
