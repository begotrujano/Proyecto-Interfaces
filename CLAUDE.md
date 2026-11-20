# Universo en tus manos

Proyecto de Interfaces Avanzadas (UV, 5.º semestre). Es un explorador educativo del espacio que se controla con **gestos de las manos frente a la webcam**. Hoy existe el **prototipo 2: la Tierra vista desde una cámara en órbita**, con el Sol, la Luna y las estrellas en su posición real del momento. La meta a largo plazo es ir de la Tierra a la Luna, luego al Sistema Solar, a cometas que el usuario desvía (estilo Solar Smash pero educativo), a la galaxia y al universo observable, con un easter egg de multiversos. Habrá una regla de escala visible y datos de NASA/ESA. Prioridad del autor: **fluidez, alta calidad visual y gestos cómodos**.

## Entorno
- Windows, VS Code, PowerShell, **Python 3.14**, entorno virtual en `.venv` (ya instalado).
- `requirements.txt`: opencv-python (5.1), mediapipe (1.2.3), numpy (2.5), pillow (13).
- Modelo `modelos/gesture_recognizer.task`. Se usan sus landmarks 2D/3D y la etiqueta de mano; las poses se calculan con geometría propia.
- Ejecutar: `.venv\Scripts\python.exe main.py` (opciones: `--camara 1 --dshow --pantalla-completa`)
- Pruebas sin cámara: `.venv\Scripts\python.exe pruebas\test_proyecto.py` (28 pruebas; deben pasar siempre. `test_integracion_hilos` exige >40 FPS y puede fallar si la laptop esta en ahorro de energia).

## Estructura de carpetas
- `main.py` (entrada) · `requirements.txt` · `CLAUDE.md` (debe estar en la raíz: Claude Code solo lo lee ahí).
- `universo/`: el programa, como paquete (`from universo import config as C`). Aquí están `config.py`, `manos.py`, `control.py`, `astros.py`, `cielo.py`, `viaje.py`, `tierra.py` y `render.py`.
- `pruebas/`: `test_proyecto.py`, `simulacion.py` y `capturas/` (PNG que generan las pruebas).
- `assets/`: imágenes (Tierra de día `tierra_nasa_bluemarble.jpg` y de noche `tierra_nasa_blackmarble.jpg`, ambas 4096×2048, NASA, dominio público). `modelos/`: modelos de MediaPipe.
- `docs/`: `CONTEXTO_PROYECTO.md` (contexto completo) y `LEEME.txt` (guía de uso).
- Las rutas a assets y modelos son relativas a `config.RAIZ` (la carpeta de `main.py`).

## Archivos
- `main.py`: ventana, teclas y 3 hilos. **Captura** guarda solo el último cuadro y lo voltea en espejo. **Detección** corre MediaPipe → `SeguidorManos` → `Control.actualizar`. **Dibujo** va a 60 FPS con `Control.avanzar` (inercia) y `Suave`. `DetectorMediaPipe` extrae `hand_landmarker.task` del bundle (zip) y usa solo HandLandmarker (`SOLO_LANDMARKER`), en memoria con `model_asset_buffer`. La captura mide los FPS reales y permite exposicion manual en vivo.
- `universo/config.py`: **todos** los parámetros, con guía de calibración.
- `universo/manos.py` (**solo numpy**):
  - Filtro One Euro, seguimiento de 2 manos (por posición predicha) y rol izq/der (etiqueta suavizada con histéresis; en conflicto decide la posición).
  - Poses en coordenadas 3D world. **Puño:** ratio punta/nudillo < 1.30 (suelta > 1.55). **OK:** aro pulgar-índice (punta a última falange, el menor entre 3D e imagen) / palma < 0.40, ≥ 2 dedos estirados (rectos o extendidos) y la mano casi quieta (`OK_V_MAX`).
  - `soltando` congela las acciones mientras el puño se abre. Dos velocidades: filtrada (swipes) y cruda (saltos del rastreo).
- `universo/control.py` (**solo numpy**, con lock):
  - OK sostenido 0.3 s activa o pausa. Pausa automática tras 6 s sin manos.
  - **Izquierda:** puño + subir/bajar = zoom logarítmico (×0.08–×6).
  - **Derecha:** swipe con la mano abierta da impulso proporcional y aditivo (en contra frena). Horizontal = gira (`giro`, `omega`); vertical = inclina (`inclinacion`, `omega_v`; tope ±90° en los polos; fricción propia más fuerte, `FRICCION_V_VIDA_MEDIA`). Cada swipe va en un solo eje: el dominante, medido en movimiento físico (`SWIPE_DOMINANCIA`, corrige el aspecto 4:3 de la cámara); los diagonales no cuentan. `_swipe_dir` = ±1 horizontal, ±2 vertical. Máquina `reposo/trazo/refractario/ignorando/salto`: ignora el regreso de la mano salvo que sea 1.6× más rápido, e ignora los saltos del rastreo. Con puño se agarra y arrastra en 2D; al soltar hereda las dos velocidades.
  - Expone `instantanea()` → `Estado`/`ManoVista`.
- **Cámara, no astros:** `giro`/`inclinacion` de `control.py` ya no giran la Tierra: dicen dónde está la cámara (sobre lon = −giro, lat = −inclinación) y el zoom, a qué distancia (`distancia_para_radio`, campo de visión `CAMPO_VISION_V`). Los gestos no cambiaron. Al orbitar, las estrellas se deslizan al lado contrario de la superficie de enfrente (es lo físicamente correcto).
- `universo/astros.py` (**solo numpy**): efemérides de baja precisión del Astronomical Almanac (Sol ~0.01°, Luna ~0.3° y ~0.2 % en distancia), tiempo sidéreo, marco de la Tierra (x = Greenwich, z = norte), clase `Camara` (filas de `M` = derecha, abajo, atrás) y catálogo de estrellas (~100 brillantes reales + ~9000 débiles estadísticas, más densas en la Vía Láctea). `config.FECHA` fija la fecha (None = ahora).
- `universo/cielo.py`: Vía Láctea (textura en coordenadas galácticas reales, muestreada a 1/16), estrellas (manchas 5×5 sumadas directo sobre sus píxeles), sprite del Sol (disco real de 0.267°, resplandor y 4 rayos) y Luna (Lommel-Seeliger, fase real, cara visible hacia la Tierra, mares procedurales).
- **Cambiar de astro** (`universo/viaje.py`, solo numpy; lo usa `main.py`): con el control EN PAUSA, la palma sobre un astro llena un anillo blanco semitransparente (`SELECCION_SOSTENER`); al completarse, `Viaje` mueve la cámara: voltea al destino, avanza (la intensidad `hiper` solo oculta las etiquetas; ya no hay rayas de estrellas) y llega rodeándolo `VIAJE_ARCO` grados para que el origen quede visible y se pueda volver. Al llegar, `Control.colocar` fija giro/inclinación en el marco del nuevo astro. `Escena._cuerpos` lista los astros (centro, radio, dibujante, rotación al marco del astro); `Escena.camara` orbita el astro en `foco`. Agregar un astro = agregarlo ahí.
- La Luna usa el mismo dibujante que la Tierra (`Tierra(..., atmosfera=False)`) con `assets/luna_nasa_lroc.jpg` (NASA SVS CGI Moon Kit, LRO); su longitud 0 mira a la Tierra.
- `universo/tierra.py`: esfera 3D en CPU (sirve para cualquier astro). Tabla precalculada de normales 3D (planos x, y, z sueltos: `split` es caro).
  - Inclinar = rotar las normales (2 `addWeighted`) y pasarlas a lon/colatitud (`cv2.phase`, `cv2.magnitude`); se cachea mientras la inclinación no cambie. Girar = desplazar la longitud.
  - Por cuadro: `addWeighted` (giro) → `cv2.remap` (BORDER_WRAP, mapas x/y separados). Con zoom ×6: ~16 ms girando y ~26 ms mientras se inclina (dividir en hilos no ayuda).
  - **Luz horneada en la textura:** `iluminar_textura` combina día (Blue Marble), ocaso naranja y noche (Black Marble, con luces de ciudades) según el Sol real, en el marco de la Tierra. No depende de la cámara: se rehace en un hilo cuando el Sol se mueve `SOL_REHORNEAR_GRADOS` (~cada 20 s); la primera vez es síncrona. El canal A = dónde puede reflejar el mar.
  - Por cuadro, solo lo que depende de la vista, a media resolución: reflejo del Sol en el mar (n·h) y atmósfera (n·sol, se enciende a contraluz). Si la Tierra llena la región, no se mezcla con el fondo.
  - Textura `assets/tierra_nasa_bluemarble.jpg` (4096×2048, NASA, dominio público). Arranca con longitud −99° (México).
- `universo/render.py`: orden de dibujo cielo → Sol (tapado por la Tierra según su disco) → Luna detrás → Tierra → Luna delante; etiquetas "Sol"/"Luna" con distancia; HUD con altitud de la cámara; cursores (naranja = zoom, azul = giro), barra de zoom, estado, avisos, info de depuración (tecla H) y miniatura de cámara. Textos con Pillow en caché, para que salgan los acentos.
- `pruebas/simulacion.py`: manos 3D sintéticas en formato MediaPipe. `pruebas/test_proyecto.py`: pruebas unitarias e integración con hilos.

## Teclas
Espacio activar/pausar · R reiniciar · I intercambiar manos · [ ] exposición (E = automática) · V cámara · H info · F pantalla completa · Q/Esc salir.

## Estado y pendientes
- Pruebas simuladas: 28/28 (conectada a la corriente; con batería `test_integracion_hilos` puede fallar por FPS). **Falta validar con la cámara real** y calibrar OK_ON, CAM_EXPOSICION, PUNO_ON, SWIPE_V_ON, GANANCIA_ZOOM y la inclinación (OMEGA_V_MAX, FRICCION_V_VIDA_MEDIA). También falta confirmar que la mano izquierda/derecha sale correcta (si no, tecla I o `INVERTIR_MANOS`).
- Por decidir:
  - si el OK también pausa;
  - la pausa automática;
  - un gesto para volver a la Tierra;
  - un gesto de traslado.
- Siguiente: Luna, luego el Sistema Solar con escala logarítmica, luego la misión del cometa. Antes de crecer, evaluar si se migra el render a GPU (Three.js o Unity), reutilizando `manos.py` y `control.py`.

## Reglas
- Todo en español: código, nombres e interfaz. En los comentarios, sin acentos; en la interfaz, sí.
- `manos.py` y `control.py` solo dependen de numpy.
- Todo parámetro nuevo va en `config.py`.
- Todo cambio de gestos o control lleva su prueba en `pruebas/test_proyecto.py`.
- Mantener ~60 FPS: precalcular, usar caché y operaciones de OpenCV/numpy, nunca bucles en Python por píxel.
- Después de cambiar algo, correr `pruebas/test_proyecto.py`.
- No tocar la carpeta `..\imAgen`.
