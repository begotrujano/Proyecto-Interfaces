"""Fichas educativas: datos reales explicados para niños (tecla D: fijar/ocultar).

Fuente: NASA, Planetary Fact Sheet (nssdc.gsfc.nasa.gov/planetary/factsheet) y Sun Fact Sheet.
Los textos son de la interfaz: con acentos.
"""

FICHAS = {
    "sol": dict(
        tipo="Nuestra estrella gigante",
        filas=[
            ("Tamaño", "¡Enorme! Caben más de un millón de Tierras dentro"),
            ("Calor", "5 500 °C por fuera; ¡15 millones °C en el centro!"),
            ("Edad", "Tiene unos 4 600 millones de años, bastante viejo"),
            ("Su luz", "Tarda solo 8 minutos en llegar hasta tus ojos"),
        ],
        dato="Es tan pesado que tiene casi toda la masa del Sistema Solar (99.8 %). Sin su luz todo estaría congelado.",
    ),
    "mercurio": dict(
        tipo="El planeta más chiquito y veloz",
        filas=[
            ("Tamaño", "Es menos de la mitad de grande que la Tierra"),
            ("Cercanía", "Es el vecino más cercano al Sol"),
            ("Su año", "Dura solo 88 días: ¡cumplirías años rapidísimo!"),
            ("Clima", "Muy caliente de día (430 °C) y helado de noche (−180 °C)"),
            ("Lunas", "No tiene ninguna"),
        ],
        dato="Está lleno de cráteres como la Luna. Ahí, un día completo dura más que todo su año.",
    ),
    "venus": dict(
        tipo="El planeta más caluroso y brillante",
        filas=[
            ("Tamaño", "Casi del mismo tamaño que la Tierra"),
            ("Clima", "464 °C: sus nubes atrapan el calor como un horno gigante"),
            ("Su giro", "¡Gira al revés que casi todos los demás planetas!"),
            ("Su año", "225 días terrestres"),
            ("Lunas", "No tiene ninguna"),
        ],
        dato="Es tan brillante que puedes verlo desde la Tierra al amanecer o al atardecer como una estrella gigante.",
    ),
    "jupiter": dict(
        tipo="El gigante del Sistema Solar",
        filas=[
            ("Tamaño", "Más de 1 300 Tierras cabrían dentro de él"),
            ("Qué es", "Una bola de gas: no tiene suelo donde pisar"),
            ("Su día", "Solo 10 horas: ¡es el que gira más rápido!"),
            ("Su año", "12 años terrestres para rodear al Sol"),
            ("Lunas", "Más de 90; las 4 grandes: Ío, Europa, Ganímedes y Calisto"),
        ],
        dato="Su Gran Mancha Roja es una tormenta más grande que la Tierra que lleva siglos girando.",
    ),
    "urano": dict(
        tipo="El gigante de hielo que rueda de lado",
        filas=[
            ("Tamaño", "4 veces más ancho que la Tierra"),
            ("Inclinación", "Su eje está tumbado casi 98°: gira rodando"),
            ("Temperatura", "−224 °C: la atmósfera más fría del Sistema Solar"),
            ("Su año", "84 años terrestres para rodear al Sol"),
            ("Lunas", "28 conocidas, con nombres de obras de Shakespeare"),
        ],
        dato="Cada polo pasa unos 42 años seguidos de pleno día y otros 42 de noche total.",
    ),
    "neptuno": dict(
        tipo="El gigante azul de los vientos",
        filas=[
            ("Tamaño", "4 veces más ancho que la Tierra"),
            ("Vientos", "Los más rápidos: hasta 2 000 km/h"),
            ("Distancia", "30 veces más lejos del Sol que la Tierra"),
            ("Su año", "165 años terrestres para rodear al Sol"),
            ("Lunas", "16 conocidas; Tritón gira al revés y tiene géiseres"),
        ],
        dato="Fue el primer planeta descubierto con matemáticas: se calculó su lugar antes de verlo (1846).",
    ),
    "saturno": dict(
        tipo="El señor de los anillos",
        filas=[
            ("Tamaño", "9 veces más ancho que la Tierra"),
            ("Anillos", "Hielo y roca: de granitos de polvo a rocas como casas"),
            ("Ligereza", "Flotaría en agua si hubiera una bañera gigante"),
            ("Su año", "29 años terrestres para rodear al Sol"),
            ("Lunas", "Más de 270; la mayor, Titán, tiene atmósfera"),
        ],
        dato="Sus anillos miden cientos de miles de km de ancho pero casi nada de grosor: en algunos sitios, unos 10 metros.",
    ),
    "tierra": dict(
        tipo="Nuestro hogar en el espacio",
        filas=[
            ("Tamaño", "12 756 km de ancho (¡nuestra casa completa!)"),
            ("Clima", "15 °C en promedio, perfecto para el agua líquida"),
            ("Su año", "365 días para dar una vuelta completa al Sol"),
            ("Su día", "24 horas para girar sobre sí misma"),
            ("Lunas", "1 (nuestra Luna)"),
        ],
        dato="Es el único planeta conocido con océanos, plantas, animales y aire respirable. Su inclinación nos da las estaciones.",
    ),
    "luna": dict(
        tipo="Nuestra vecina brillante",
        filas=[
            ("Tamaño", "Es unas 4 veces más pequeña que la Tierra"),
            ("Distancia", "A 384 400 km (unos 3 días viajando en cohete)"),
            ("Su vuelta", "Tarda 27 días en rodear a la Tierra"),
            ("Gravedad", "6 veces menor: ¡ahí darías saltos gigantes!"),
        ],
        dato="Siempre nos muestra la misma cara y las huellas de los astronautas siguen ahí porque no hay viento que las borre.",
    ),
    "marte": dict(
        tipo="El planeta rojo",
        filas=[
            ("Tamaño", "Un poco más de la mitad del tamaño de la Tierra"),
            ("Color", "Es rojo porque su polvo tiene mucho hierro oxidado"),
            ("Clima", "Hace mucho frío: −65 °C en promedio"),
            ("Su año", "687 días (casi el doble que aquí)"),
            ("Lunas", "2 chiquitas: Fobos y Deimos"),
        ],
        dato="Tiene el volcán más alto del Sistema Solar (Monte Olimpo) y allá viven robots exploradores tomándole fotos.",
    ),
}