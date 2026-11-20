# Universo en tus manos

Proyecto de Interfaces Avanzadas (UV, 5.º semestre). Es un explorador educativo del espacio que se controla con **gestos de las manos frente a la webcam**. Hoy existe el **prototipo 2: la Tierra como globo terráqueo**. La meta a largo plazo es ir de la Tierra a la Luna, luego al Sistema Solar, a cometas que el usuario desvía (estilo Solar Smash pero educativo), a la galaxia y al universo observable, con un easter egg de multiversos. Habrá una regla de escala visible y datos de NASA/ESA. Prioridad del autor: **fluidez, alta calidad visual y gestos cómodos**.

## Entorno
- Windows, VS Code, PowerShell, **Python 3.14**, entorno virtual en `.venv` (ya instalado).
- `requirements.txt`: opencv-python (5.1), mediapipe (1.2.3), numpy (2.5), pillow (13).
- Modelo `gesture_recognizer.task` en la raíz. Se usan sus landmarks 2D/3D y la etiqueta de mano; las poses se calculan con geometría propia.
- Ejecutar: `.venv\Scripts\python.exe main.py` (opciones: `--camara 1 --dshow --pantalla-completa`)
- Pruebas sin cámara: `.venv\Scripts\python.exe test_proyecto.py` (17 pruebas; deben pasar siempre).

## Archivos
- `main.py`: ventana, teclas y 3 hilos. **Captura** guarda solo el último cuadro y lo voltea en espejo. **Detección** corre MediaPipe → `SeguidorManos` → `Control.actualizar`. **Dibujo** va a 60 FPS con `Control.avanzar` (inercia) y `Suave`. `DetectorMediaPipe` carga el modelo con `model_asset_buffer`.
- `config.py`: **todos** los parámetros, con guía de calibración.
- `manos.py` (**solo numpy**):
  - Filtro One Euro, seguimiento de 2 manos (por posición predicha) y rol izq/der (etiqueta suavizada con histéresis; en conflicto decide la posición).
  - Poses en coordenadas 3D world. **Puño:** ratio punta/nudillo < 1.30 (suelta > 1.55). **OK:** pulgar-índice/palma < 0.33 y ≥ 2 dedos estirados.
  - `soltando` congela las acciones mientras el puño se abre. Dos velocidades: filtrada (swipes) y cruda (saltos del rastreo).
- `control.py` (**solo numpy**, con lock):
  - OK sostenido 0.3 s activa o pausa. Pausa automática tras 6 s sin manos.
  - **Izquierda:** puño + subir/bajar = zoom logarítmico (×0.08–×6).
  - **Derecha:** swipe horizontal con la mano abierta da impulso proporcional y aditivo (en contra frena). Máquina `reposo/trazo/refractario/ignorando/salto`: ignora el regreso de la mano salvo que sea 1.6× más rápido, e ignora los saltos del rastreo. Con puño se agarra y arrastra; al soltar hereda la velocidad.
  - Expone `instantanea()` → `Estado`/`ManoVista`.
- `tierra.py`: esfera 3D en CPU. Tabla precalculada de lat/lon; girar = desplazar la longitud.
  - Por cuadro: `warpAffine` → `cv2.transform` → `cv2.remap` (BORDER_WRAP).
  - Iluminación precalculada (día/noche, reflejo en el mar, atmósfera vía LUT), mipmaps, y caché de lo que no cambia al girar.
  - Textura `assets/tierra_nasa_bluemarble.jpg` (4096×2048, NASA, dominio público). Arranca con longitud −99° (México).
- `render.py`: estrellas, Tierra, cursores (naranja = zoom, azul = giro), barra de zoom, estado, avisos, info de depuración (tecla H) y miniatura de cámara. Textos con Pillow en caché, para que salgan los acentos.
- `simulacion.py`: manos 3D sintéticas en formato MediaPipe. `test_proyecto.py`: pruebas unitarias e integración con hilos.

## Teclas
Espacio activar/pausar · R reiniciar · I intercambiar manos · V cámara · H info · F pantalla completa · Q/Esc salir.

## Estado y pendientes
- Pruebas simuladas: 17/17 pasan. **Falta validar con la cámara real** y calibrar OK_ON, PUNO_ON, SWIPE_V_ON y GANANCIA_ZOOM. También falta confirmar que la mano izquierda/derecha sale correcta (si no, tecla I o `INVERTIR_MANOS`).
- Por decidir:
  - si el OK también pausa;
  - la pausa automática;
  - inclinar con un swipe vertical;
  - un gesto para volver a la Tierra;
  - un gesto de traslado.
- Siguiente: Luna, luego el Sistema Solar con escala logarítmica, luego la misión del cometa. Antes de crecer, evaluar si se migra el render a GPU (Three.js o Unity), reutilizando `manos.py` y `control.py`.

## Reglas
- Todo en español: código, nombres e interfaz. En los comentarios, sin acentos; en la interfaz, sí.
- `manos.py` y `control.py` solo dependen de numpy.
- Todo parámetro nuevo va en `config.py`.
- Todo cambio de gestos o control lleva su prueba en `test_proyecto.py`.
- Mantener ~60 FPS: precalcular, usar caché y operaciones de OpenCV/numpy, nunca bucles en Python por píxel.
- Después de cambiar algo, correr `test_proyecto.py`.
- No tocar la carpeta `..\imAgen`.
