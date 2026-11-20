"""Todos los parametros ajustables, en un solo lugar.

Guia rapida para afinar (mira los numeros de depuracion en pantalla, tecla H):
  - El OK no activa:                 sube OK_ON (0.33 -> 0.40)
  - El OK se activa solo:            baja OK_ON o sube OK_SOSTENER
  - El puno no "agarra":             sube PUNO_ON (1.25 -> 1.35)
  - Suelta el puno sin querer:       sube PUNO_OFF
  - El swipe no gira:                baja SWIPE_V_ON
  - Gira solo al mover la mano:      sube SWIPE_V_ON
  - El zoom va muy rapido/lento:     GANANCIA_ZOOM
  - Tiembla / se siente con retraso: OE_MINCUT / OE_BETA y SUAVIZADO_TAU
"""

# ----------------------------------------------------------------------
# Ventana y render
# ----------------------------------------------------------------------
ANCHO, ALTO = 1280, 720            # resolucion interna del lienzo
FPS_RENDER = 60
RADIO_BASE_REL = 0.24              # radio de la Tierra a escala 1 (fraccion del alto)
TEXTURA = "assets/tierra_nasa_bluemarble.jpg"   # si falta, se genera una procedural
LONGITUD_INICIAL = -99.0           # grados; -99 = Mexico al centro
SUAVIZADO_TAU = 0.045              # s; suavizado visual (independiente de la camara)

# ----------------------------------------------------------------------
# Camara
# ----------------------------------------------------------------------
CAM_INDICE = 0
CAM_ANCHO, CAM_ALTO, CAM_FPS = 640, 480, 60
# Zona comoda: el 80 % central de la camara se estira a toda la pantalla,
# asi llegas a las orillas sin sacar la mano del cuadro.
MARGEN_CAMARA = 0.10

# ----------------------------------------------------------------------
# Deteccion (MediaPipe)
# ----------------------------------------------------------------------
MODELO = "gesture_recognizer.task"
NUM_MANOS = 2
CONFIANZA_MIN = 0.45

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

# OK (pulgar con indice + al menos 2 de los otros 3 dedos estirados)
OK_ON = 0.33                   # distancia pulgar-indice / tamano de la palma
OK_OFF = 0.48
OK_RECTITUD = 0.78             # que tan recto debe estar un dedo para contar como estirado
OK_DEDOS_MIN = 2
OK_SOSTENER = 0.30             # s que hay que sostener el OK (se ve un anillo de progreso)
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
#   puno = agarrar la Tierra (la superficie sigue a tu mano); al soltar conserva el impulso
#   swipe con la mano abierta = darle vuelo (cada swipe suma; en contra, frena)
RADIO_GIRO_MIN, RADIO_GIRO_MAX = 150.0, 450.0   # px; radio usado para convertir mano -> angulo
SWIPE_V_ON = 1.5               # anchos de pantalla por segundo para que cuente como swipe
SWIPE_V_OFF = 0.6
SWIPE_V_FUERTE = 3.5           # a esta velocidad (o mas) el swipe da el impulso maximo
SWIPE_V_SALTO = 9.0            # mas rapido que esto (sin filtrar) = salto del rastreo, se ignora
SWIPE_HORIZONTAL = 1.3         # |vx| debe ser al menos esto * |vy|
SWIPE_REFRACTARIO = 0.45       # s tras un swipe en que se ignora el regreso de la mano...
SWIPE_REVERSA = 1.6            # ...salvo que sea 1.6 veces mas rapido que el swipe anterior
OMEGA_MAX = 4.5                # rad/s (~0.7 vueltas por segundo)
FRICCION_VIDA_MEDIA = 1.6      # s en que la velocidad de giro cae a la mitad
ROTACION_REPOSO = 0.07         # rad/s de giro suave cuando el control esta en pausa
