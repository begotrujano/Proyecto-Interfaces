"""Descarga las texturas reales a assets/ (hazlo una vez, con internet).

    .venv\\Scripts\\python.exe herramientas\\descargar_texturas.py

Marte: "Mars" de NASA 3D Resources (https://science.nasa.gov/3d-resources/mars), mosaico de las
imagenes de Viking procesadas por el USGS (NASA/JPL-Caltech, dominio publico). Se guarda como
assets/marte_nasa_viking.jpg y el programa la usa en lugar de la textura procedural.
Si al verla los rasgos quedan corridos 180 grados, pon DESFASE_TEXTURA_MARTE = 180.0 en universo/config.py.

Mercurio: mosaico global de MESSENGER (PIA15063, NASA/JHU APL/Carnegie Institution of Washington,
https://photojournal.jpl.nasa.gov/catalog/PIA15063). Viene de 0 a 360 grados este (la longitud 0 en el
borde); aqui se recentra para que la longitud 0 quede al centro, como espera el programa.

Venus: desde el espacio solo se ven las nubes. Se usa la textura "Venus Atmosphere" de Solar System
Scope (https://www.solarsystemscope.com/textures, basada en datos de la NASA, licencia CC BY 4.0: hay
que dar credito, ya esta en docs/LEEME.txt). La superficie de radar de Magellan esta en
https://science.nasa.gov/3d-resources/venus (guardala como assets/venus_nasa_magellan.jpg y ponla
primero en config.TEXTURA_VENUS si quieres ver el suelo en vez de las nubes).

Jupiter: "Jupiter" de Solar System Scope (misma fuente y licencia CC BY 4.0 que Venus). Se guarda como
assets/jupiter_nubes.jpg; mientras no este, se usa assets/jupiter_procedural.jpg (python -m universo.planetas jupiter).

Saturno: "Saturn" de Solar System Scope (CC BY 4.0), assets/saturno_nubes.jpg; los anillos no usan imagen: se
calculan (universo/anillos.py). Sin la imagen se usa assets/saturno_procedural.jpg (python -m universo.planetas saturno).

Sol: no hay un mapa de toda la superficie con licencia clara en un solo archivo; si consigues uno
equirectangular (2:1) de SDO/NASA, guardalo como assets/sol_nasa_sdo.jpg. Mientras tanto se usa la procedural.
"""

import sys
import urllib.request
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
# nombre -> (url, grados que se gira para dejar la longitud 0 al centro)
DESCARGAS = {
    "marte_nasa_viking.jpg": ("https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/image/mars/Mars.jpg", 0.0),
    "mercurio_nasa_messenger.jpg": ("https://photojournal.jpl.nasa.gov/jpeg/PIA15063.jpg", 180.0),
    "venus_nubes.jpg": ("https://www.solarsystemscope.com/textures/download/2k_venus_atmosphere.jpg", 0.0),
    "jupiter_nubes.jpg": ("https://www.solarsystemscope.com/textures/download/2k_jupiter.jpg", 0.0),
    "neptuno_nubes.jpg": ("https://www.solarsystemscope.com/textures/download/2k_neptune.jpg", 0.0),
    "urano_nubes.jpg": ("https://www.solarsystemscope.com/textures/download/2k_uranus.jpg", 0.0),
    "saturno_nubes.jpg": ("https://www.solarsystemscope.com/textures/download/2k_saturn.jpg", 0.0),
}


def main():
    ASSETS.mkdir(exist_ok=True)
    for nombre, (url, recentrar) in DESCARGAS.items():
        destino = ASSETS / nombre
        if destino.exists():
            print(f"{nombre}: ya esta, se omite (borrala para bajarla de nuevo)")
            continue
        print(f"Descargando {nombre} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Universo en tus manos)"})
            datos = urllib.request.urlopen(req, timeout=90).read()
        except Exception as e:
            print(f"  No se pudo: {e}\n  Bajala a mano de {url} y guardala como {destino}")
            continue
        try:                                           # comprobacion rapida (debe ser 2:1) y recentrado
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
            alto, ancho = img.shape[:2]
            if recentrar:
                img = np.roll(img, int(round(recentrar / 360.0 * ancho)), axis=1)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 94])
                datos = buf.tobytes()
            print(f"  listo: {ancho}x{alto}" + ("" if ancho == 2 * alto else "  (no es 2:1; se reescalara al cargar)")
                  + (f", recentrada {recentrar:.0f} grados" if recentrar else ""))
        except Exception:
            print("  listo")
        destino.write_bytes(datos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
