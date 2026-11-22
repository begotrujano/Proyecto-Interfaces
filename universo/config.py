"""Todos los parametros ajustables, en un solo lugar.

Guia rapida para afinar (mira los numeros de depuracion en pantalla, tecla H):
  - El OK no activa:                 mira "ok" y "dedos" (tecla H); sube OK_ON (0.40 -> 0.48)
                                     o baja OK_DEDOS_MIN a 1
  - El OK se activa solo:            baja OK_ON o sube OK_SOSTENER
  - El puno no "agarra":             sube PUNO_ON (1.25 -> 1.35)
  - Suelta el puno sin querer:       sube PUNO_OFF
  - El swipe no gira:                baja SWIPE_V_ON
  - Gira solo al mover la mano:      sube SWIPE_V_ON
  - Un swipe horizontal inclina (o al reves): sube SWIPE_DOMINANCIA
  - Un swipe vertical inclina mucho/poco: FRICCION_V_VIDA_MEDIA (y OMEGA_V_MAX)
  - El zoom va muy rapido/lento:     GANANCIA_ZOOM
  - Tiembla / se siente con retraso: OE_MINCUT / OE_BETA y SUAVIZADO_TAU
"""

# ----------------------------------------------------------------------
# Ventana y render
# ----------------------------------------------------------------------
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent   # carpeta del proyecto (la de main.py)

ANCHO, ALTO = 1280, 720            # resolucion interna del lienzo
FPS_RENDER = 60
RADIO_BASE_REL = 0.24              # radio de la Tierra a escala 1 (fraccion del alto)
TEXTURA = "assets/tierra_nasa_bluemarble.jpg"   # relativo a RAIZ; si falta, se genera una procedural
LONGITUD_INICIAL = -99.0           # grados; -99 = Mexico al centro
SUAVIZADO_TAU = 0.045              # s; suavizado visual (independiente de la camara)

# ----------------------------------------------------------------------
# Espacio: camara en orbita, Sol, Luna y estrellas (posiciones reales)
# ----------------------------------------------------------------------
# Los gestos mueven la CAMARA alrededor de la Tierra (giro/inclinacion = donde esta la
# camara; zoom = que tan cerca). El Sol, la Luna y las estrellas estan donde estan de
# verdad en este momento, asi que el lado de dia y el de noche son los reales.
FECHA = None                       # None = ahora mismo; o un texto ISO, p. ej. "2024-04-08T18:17:00+00:00"
CAMPO_VISION_V = 40.0              # grados de campo de vision vertical de la camara
TEXTURA_NOCHE = "assets/tierra_nasa_blackmarble.jpg"   # luces de ciudades (NASA Black Marble 2016)
BRILLO_LUCES = 1.25                # intensidad del lado de noche (luces y tierra a la luz de la Luna)
SOL_REHORNEAR_GRADOS = 0.08        # la luz del dia/noche se recalcula (en otro hilo) cuando el Sol
                                   # se mueve esto respecto a la Tierra (0.25 grados por minuto)
BRILLO_ESTRELLAS = 650.0           # brillo de una estrella de magnitud 1
CONTRASTE_ESTRELLAS = 0.45         # <1 = las debiles se ven mas (1 = brillo fisico, casi no se ven)
BRILLO_VIA_LACTEA = 0.26
SOL_ANGULO_RADIO_GRADOS = 0.267    # radio aparente real del Sol visto desde la Tierra
SOL_RESPLANDOR_REL = 0.42          # tamano del resplandor (fraccion del alto de la pantalla)
BRILLO_LUNA = 0.85
LUZ_TIERRA_EN_LUNA = 0.03          # "luz cenicienta": la Tierra ilumina un poco el lado oscuro
TEXTURA_LUNA = "assets/luna_nasa_lroc.jpg"   # NASA SVS CGI Moon Kit (LRO), 4096x2048
ETIQUETAS_ASTROS = True

# ----------------------------------------------------------------------
# Sol y planetas (esferas con textura, en su posicion real)
# ----------------------------------------------------------------------
# Cada textura es una lista: se usa la primera que exista. Pon la imagen de la NASA con el
# nombre de la primera (mira herramientas/descargar_texturas.py); si no esta, se usa la procedural.
TEXTURA_MARTE = ["assets/marte_nasa_viking.jpg", "assets/marte_procedural.jpg"]
TEXTURA_SOL = ["assets/sol_nasa_sdo.jpg", "assets/sol_procedural.jpg"]
TEXTURA_MERCURIO = ["assets/mercurio_nasa_messenger.jpg", "assets/mercurio_procedural.jpg"]
# Venus: desde el espacio solo se ven sus nubes (acido sulfurico). Si prefieres ver la superficie
# (radar de Magellan), bajala de science.nasa.gov/3d-resources/venus como "assets/venus_nasa_magellan.jpg"
# y ponla primero en esta lista. venus_nubes.jpg la baja herramientas/descargar_texturas.py.
TEXTURA_VENUS = ["assets/venus_nubes.jpg", "assets/venus_procedural.jpg"]
DESFASE_TEXTURA_MARTE = 0.0        # grados; si la imagen de Marte trae la longitud 0 en un borde y no al centro, usa 180
DESFASE_TEXTURA_SOL = 0.0
DESFASE_TEXTURA_MERCURIO = 0.0     # igual que el de Marte (la procedural ya trae la longitud 0 al centro)
DESFASE_TEXTURA_VENUS = 0.0
COLOR_ATM_VENUS = (0.55, 0.86, 1.0)  # BGR del halo de Venus: bruma amarilla palida (a contraluz se enciende el anillo)
BRILLO_SOL = 1.25                  # el Sol emite: >1 satura el centro a blanco; el borde queda anaranjado
COLOR_HALO_SOL = (0.40, 0.75, 1.0) # BGR del halo junto al disco del Sol
PLANETAS_ACTIVOS = ("mercurio", "venus", "tierra", "marte")   # los que ya se dibujan como esfera. Agrega uno al implementarlo.
PUNTO_ASTRO_MIN = 1.2              # radio (px) bajo el cual un planeta se dibuja como un punto brillante

# ----------------------------------------------------------------------
# Vista del Sistema Solar completo (tecla S, o palma sobre el astro donde estas)
# ----------------------------------------------------------------------
SISTEMA_RADIO_REL = 0.40           # a zoom x1 la orbita del planeta activo mas lejano ocupa esta fraccion del alto
SISTEMA_INCLINACION = 52.0         # grados de inclinacion inicial de la vista (90 = de canto, 0 = desde arriba)
SISTEMA_GIRO_INICIAL = 35.0        # grados
SISTEMA_ICONO_PLANETA = 12.0       # radio (px) de un planeta a zoom x1 (crece con el zoom, suave)
SISTEMA_ICONO_SOL = 19.0
SISTEMA_MOSTRAR_PENDIENTES = True  # orbitas y puntos de los planetas que aun no se implementan
SISTEMA_TRANSICION = 0.55          # s del fundido al entrar o salir de la vista

# ----------------------------------------------------------------------
# Cambiar de astro (solo con el control EN PAUSA): deja la palma sobre un astro
# ----------------------------------------------------------------------
SELECCION_SOSTENER = 1.2           # s con la palma encima para elegirlo (anillo blanco)
SELECCION_TOLERANCIA = 0.25        # s que la palma puede salirse sin perder el progreso
SELECCION_MARGEN = 30.0            # px extra alrededor del astro que cuentan como "encima"
SELECCION_RADIO_MIN = 18.0         # px: los astros pequenos se pueden elegir igual
VIAJE_DURACION = 3.0               # s del viaje completo
VIAJE_ARCO = 155.0                 # grados que rodeas el destino (el origen queda a un costado)
VIAJE_GIRO = 0.25                  # fraccion del viaje para voltear hacia el destino            # nombre y distancia junto al Sol y la Luna

# ----------------------------------------------------------------------
# Raton / touchpad (alternativa a las manos; siempre activo, aunque el control este en pausa)
#   arrastrar con el boton izquierdo = girar e inclinar | rueda (o dos dedos en el pad) = zoom
#   clic sobre un astro = viajar a el (en la vista del sistema, sobre el astro actual = volver)
# ----------------------------------------------------------------------
RATON_ZOOM_PASO = 0.18             # cambio del logaritmo del zoom por cada "muesca" de la rueda
RATON_CLIC_PX = 6.0                # movimiento maximo (px) para que un arrastre cuente como clic
RATON_GANANCIA = 1.0               # 1 = la superficie sigue al cursor; >1 = gira mas rapido
RATON_INERCIA = True               # al soltar el arrastre conserva el impulso (como el puno)

# ----------------------------------------------------------------------
# Camara
# ----------------------------------------------------------------------
CAM_INDICE = 0
CAM_ANCHO, CAM_ALTO, CAM_FPS = 640, 480, 60
# Zona comoda: el 80 % central de la camara se estira a toda la pantalla,
# asi llegas a las orillas sin sacar la mano del cuadro.
MARGEN_CAMARA = 0.10
# Exposicion. None = automatica. Con poca luz la camara alarga la exposicion: la mano
# sale barrida al moverla rapido, MediaPipe la pierde y la camara baja a 15-30 FPS.
# Un valor manual corto (DirectShow: -6 = 1/64 s, -7 = 1/128 s) congela mejor la mano,
# pero oscurece la imagen. Pruebalo en vivo con [ y ] (E = volver a automatica)
# y mira "cámara N fps" en la info (tecla H).
CAM_EXPOSICION = None
CAM_EXPOSICION_MANUAL = -6     # desde donde empiezan [ y ] si estaba en automatica

# ----------------------------------------------------------------------
# Deteccion (MediaPipe)
# ----------------------------------------------------------------------
MODELO = "modelos/gesture_recognizer.task"   # relativo a RAIZ
NUM_MANOS = 2
SOLO_LANDMARKER = True         # solo el detector de puntos (mas rapido); False = GestureRecognizer completo
CONFIANZA_DETECCION = 0.45     # para encontrar una mano nueva (palma)
CONFIANZA_PRESENCIA = 0.40     # para seguir creyendo que hay mano (mas bajo = la pierde menos)
CONFIANZA_RASTREO = 0.40       # para seguirla desde el cuadro anterior sin volver a buscarla

# Filtro One Euro sobre los puntos de la mano (coordenadas 0..1)
OE_MINCUT = 0.6       # menos = mas estable con la mano quieta
OE_BETA = 8.0         # mas = menos retraso al moverse rapido
OE_DCUT = 0.6

# ----------------------------------------------------------------------
# Seguimiento y roles (izquierda = zoom, derecha = giro)
# ----------------------------------------------------------------------
GRACIA_PERDIDA = 0.25          # s que se conserva una mano si el rastreo la pierde
MATCH_MAX_REL = 0.35
INVERTIR_MANOS = False         # True = derecha hace zoom e izquierda gira (zurdos). Tecla I.

# ----------------------------------------------------------------------
# Poses. Todo se mide en coordenadas 3D de MediaPipe ("world"), asi que no
# depende de la distancia a la camara ni de como gires la mano.
# ----------------------------------------------------------------------
# Puno: distancia punta-muneca / nudillo-muneca (~2.0 abierta, ~1.0 cerrada)
PUNO_ON = 1.30
PUNO_OFF = 1.55
PUNO_DEBOUNCE_ON = 0.05
PUNO_DEBOUNCE_OFF = 0.08

# OK (aro pulgar-indice + al menos 2 de los otros 3 dedos estirados)
#   El aro se mide como la distancia de la punta del pulgar a la ultima falange del
#   indice (y viceversa), en 3D y en la imagen; se usa la menor.
OK_ON = 0.40                   # separacion del aro / tamano de la palma para empezar el OK
OK_OFF = 0.58                  # ...y para soltarlo (histeresis)
OK_PESO_2D = 1.2               # castigo a la medida en la imagen (>1 = confia mas en la 3D)
OK_RECTITUD = 0.72             # un dedo cuenta como estirado si esta asi de recto...
OK_EXTENSION = 1.55            # ...o si su punta esta asi de lejos de la muneca (relajado)
OK_DEDOS_MIN = 2
OK_V_MAX = 0.8                 # la mano debe estar casi quieta para empezar el OK (anchos de camara/s)
OK_SOSTENER = 0.30             # s que hay que sostener el OK (se ve un anillo de progreso)
OK_TOLERANCIA = 0.20           # s que puede "parpadear" el OK sin perder el progreso
OK_TAMBIEN_PAUSA = True        # hacer OK otra vez pausa el control

# ----------------------------------------------------------------------
# Control
# ----------------------------------------------------------------------
DESACTIVAR_SIN_MANOS = 6.0     # s sin manos -> se pausa solo (0 = nunca)
EDAD_MINIMA_MANO = 0.20        # s; ignora movimientos de una mano recien aparecida

# Zoom (mano izquierda): puno + subir/bajar
GANANCIA_ZOOM = 2.5            # subir media pantalla = x3.5
ESCALA_MIN, ESCALA_MAX = 0.08, 6.0

# Giro (mano derecha)
#   puno = agarrar la Tierra (la superficie sigue a tu mano, en 2D); al soltar conserva el impulso
#   swipe con la mano abierta = darle vuelo (cada swipe suma; en contra, frena)
#     horizontal = girar sobre su eje;  vertical = inclinar (ver el norte o el sur)
RADIO_GIRO_MIN, RADIO_GIRO_MAX = 150.0, 450.0   # px; radio usado para convertir mano -> angulo
SWIPE_V_ON = 1.5               # anchos (o altos) de pantalla por segundo para que cuente como swipe
SWIPE_V_OFF = 0.6
SWIPE_V_FUERTE = 3.5           # a esta velocidad (o mas) el swipe da el impulso maximo
SWIPE_V_SALTO = 9.0            # mas rapido que esto (sin filtrar) = salto del rastreo, se ignora
SWIPE_DOMINANCIA = 1.3         # el eje del swipe debe ser al menos esto * el otro (movimiento fisico)
SWIPE_REFRACTARIO = 0.45       # s tras un swipe en que se ignora el regreso de la mano...
SWIPE_REVERSA = 1.6            # ...salvo que sea 1.6 veces mas rapido que el swipe anterior
OMEGA_MAX = 4.5                # rad/s (~0.7 vueltas por segundo)
OMEGA_V_MAX = 2.5              # rad/s de inclinacion (de ecuador a polo en ~0.6 s)
INCLINACION_MAX = 90.0         # grados; 90 = puedes ver el polo de frente
# La inclinacion frena mas rapido que el giro: recorre ~velocidad * 1.44 * vida media.
# Con 0.5 s un swipe suave inclina ~30 grados y uno fuerte llega al polo.
FRICCION_V_VIDA_MEDIA = 0.5
FRICCION_VIDA_MEDIA = 1.6      # s en que la velocidad de giro cae a la mitad
ROTACION_REPOSO = 0.07         # rad/s de giro suave cuando el control esta en pausa
