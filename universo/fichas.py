"""Fichas educativas: datos reales de cada astro, que aparecen al llegar a el (tecla D: fijar/ocultar).

Fuente: NASA, Planetary Fact Sheet (nssdc.gsfc.nasa.gov/planetary/factsheet) y Sun Fact Sheet.
Los textos son de la interfaz: con acentos.
"""

FICHAS = {
    "sol": dict(
        tipo="Estrella (enana amarilla, tipo G2)",
        filas=[("Diámetro", "1.39 millones de km (109 Tierras)"),
               ("Temperatura", "5 500 °C en la superficie; 15 millones en el centro"),
               ("Edad", "unos 4 600 millones de años"),
               ("Luz hasta la Tierra", "8 min 20 s")],
        dato="Tiene el 99.8 % de toda la masa del Sistema Solar: todo lo demás junto es menos del 0.2 %."),
    "mercurio": dict(
        tipo="Planeta rocoso · el más cercano al Sol",
        filas=[("Diámetro", "4 879 km (0.38 Tierras)"),
               ("Distancia al Sol", "57.9 millones de km · 0.39 UA"),
               ("Año", "88 días"),
               ("Gira sobre sí", "en 58.6 días"),
               ("Temperatura", "de −180 °C de noche a 430 °C de día"),
               ("Lunas", "ninguna")],
        dato="Un día solar (de un amanecer al siguiente) dura 176 días terrestres: dos de sus años."),
    "venus": dict(
        tipo="Planeta rocoso · cubierto de nubes de ácido sulfúrico",
        filas=[("Diámetro", "12 104 km (0.95 Tierras)"),
               ("Distancia al Sol", "108.2 millones de km · 0.72 UA"),
               ("Año", "224.7 días"),
               ("Gira sobre sí", "en 243 días, al revés"),
               ("Temperatura", "464 °C (la más alta del Sistema Solar)"),
               ("Lunas", "ninguna")],
        dato="Su día (243 días) es más largo que su año (225). Sus nubes, en cambio, le dan la vuelta en solo 4 días."),
    "tierra": dict(
        tipo="Planeta rocoso · nuestro hogar",
        filas=[("Diámetro", "12 756 km"),
               ("Distancia al Sol", "149.6 millones de km · 1 UA"),
               ("Año", "365.25 días"),
               ("Gira sobre sí", "en 23 h 56 min"),
               ("Temperatura media", "15 °C"),
               ("Lunas", "1")],
        dato="El único lugar conocido con vida. Su eje inclinado 23.4° es lo que causa las estaciones."),
    "luna": dict(
        tipo="Satélite natural de la Tierra",
        filas=[("Diámetro", "3 475 km (0.27 Tierras)"),
               ("Distancia a la Tierra", "384 400 km en promedio"),
               ("Vuelta a la Tierra", "27.3 días"),
               ("Temperatura", "de −173 °C a 127 °C")],
        dato="Siempre nos muestra la misma cara: tarda lo mismo en girar sobre sí que en darle la vuelta a la Tierra."),
    "marte": dict(
        tipo="Planeta rocoso · el planeta rojo",
        filas=[("Diámetro", "6 792 km (0.53 Tierras)"),
               ("Distancia al Sol", "228 millones de km · 1.52 UA"),
               ("Año", "687 días"),
               ("Gira sobre sí", "en 24 h 37 min"),
               ("Temperatura media", "−65 °C"),
               ("Lunas", "2 (Fobos y Deimos)")],
        dato="Tiene el volcán más alto del Sistema Solar, Olympus Mons: unos 22 km, casi 2.5 veces el Everest."),
}
