# Universo en tus manos — Contexto completo del proyecto

> Documento de contexto para continuar el desarrollo. Describe **qué queremos construir**, **por qué**, y **qué existe ya** (código, decisiones, entorno y pendientes). Actualizado: 20 de noviembre de 2026.

---

## 1. Origen y contexto académico

- **Materia:** Interfaces Avanzadas, 5.º semestre, Universidad Veracruzana (UV).
- **Autor / líder:** Alfonso Mendoza, que trabaja con un equipo.
- **Encargo:** proponer y prototipar una **interfaz natural**: gestos, voz o realidad aumentada. Debe ser útil, innovadora y demostrable; puede ser un juego o una actividad divertida.
- **Cómo se eligió la idea:** el equipo exploró salud/rehabilitación, educación, una feria en RA, un juego de hechizos con gestos ("Mage Arena con las manos") y un "Jarvis" para controlar la PC con gestos. Estas dos últimas quedaron **guardadas como ideas alternativas**. La elegida fue un **explorador educativo del espacio controlado con las manos**, porque fue la que más le gustó al autor.
- **Experiencia previa del autor (carpeta `imAgen`):** prácticas con OpenCV y MediaPipe, entre ellas una pizarra virtual con gestos (`pizarra_virtual.py`, actividad 8), un Space Invaders con gestos y un sandbox de física con manos-guante (`prueba3.py`). En ese sandbox se agarran bolas con el puño, con seguimiento de 2 manos, identidad estable y "mano fantasma" si se pierde el rastreo. De ahí vienen varias técnicas que se reutilizan en este proyecto.

## 2. Visión del producto

Un software educativo del espacio "en tus manos": la cámara web ve tus manos y con gestos controlas el universo.

**El viaje completo, a largo plazo:**
1. **La Tierra** como globo terráqueo: lo giras y le haces zoom.
2. **Tierra y Luna**: cómo se relacionan (órbita, fases, mareas).
3. Expandir con la mano hasta el **Sistema Solar**: los planetas.
4. **Cometas, nebulosas y cúmulos de sistemas**. El usuario se siente un "ente poderoso" que puede **detener o desviar cometas**, como en *Solar Smash* pero **educativo**. Por ejemplo: "desvía el cometa antes de que llegue a la Tierra" con física real, donde un empujón pequeño y temprano funciona mejor que uno grande y tardío.
5. **La galaxia**, luego el **grupo de galaxias** y al final el **universo observable**.
6. **Easter egg:** entrar a "otros multiversos", marcado claramente como **hipótesis, no ciencia comprobada**.

**Principios de diseño acordados:**
- **Zoom continuo de escala**, de la Tierra al universo, siempre con una **regla de escala visible** (km, UA, años luz, tiempo-luz) para enseñar la magnitud.
- **Misiones en vez de solo mirar.**
- **Pocos gestos y con significado.**
- **Fichas cortas** al tocar un objeto: qué es, tamaño y un dato curioso.
- **Datos correctos**, de NASA/ESA (datos abiertos).
- **Prioridad absoluta del autor:** que **no se vea de baja calidad**, que los gestos **no sean cortados** y que **no sea difícil de controlar**. Todo se optimiza desde el inicio.

**Por qué es diferente de lo que existe:** Solar System Scope, Star Walk o Universe Sandbox se controlan con mouse o pantalla táctil, y Solar Smash es solo destrucción. Nadie combina bien **gestos con cámara + escala completa + enseñanza**.

## 3. Estado actual (lo que ya existe y funciona)

**Prototipo 2: "la Tierra como globo terráqueo"**, en `...\INTERFACES AVANZADAS\Proyecto Final\`.

### 3.1 Qué hace hoy
- Fondo negro con estrellas (generadas, con un velo tenue de vía láctea).
- **La Tierra en 3D que gira sobre su eje**, como un globo real, con la textura **NASA Blue Marble** (dominio público). Tiene iluminación de día y noche con terminador suave, reflejo del sol solo en el mar y atmósfera azul en el borde. Arranca con **México al centro** (longitud −99°).
- **Control por gestos con las dos manos**, con un esquema de activación (ver 3.2).
- Interfaz en pantalla:
  - indicador "CONTROL ACTIVO / EN PAUSA";
  - escala y longitud del centro (p. ej. "Centro 99° O");
  - barra de zoom a la izquierda;
  - cursores por mano con color por rol (naranja = zoom, azul = giro);
  - anillo de progreso del OK y estela del swipe;
  - avisos ("Control activado");
  - instrucciones abajo;
  - miniatura de la cámara con esqueleto (arriba a la derecha);
  - información de depuración con números en vivo para calibrar (tecla H).
- Textos con **Pillow** (fuente Segoe UI en Windows), así que sí muestra acentos y "ñ". El texto de OpenCV no puede.

### 3.2 Esquema de interacción vigente

Lo pidió el autor después de probar la versión anterior. En esa versión había que pellizcar con las dos manos a la vez para escalar y rotar, y el OK/pellizco resultaba "muy específico e incómodo".

| Acción | Gesto |
|---|---|
| **Activar** | Signo de **OK una sola vez**, con cualquier mano, sostenido 0.3 s (se ve un anillo verde que se llena). Respaldo: barra espaciadora. |
| **Zoom** (mano **izquierda**) | **Cerrar el puño** y **subir** (acercar) o **bajar** (alejar). Se abre la mano para soltar y reacomodar, como un trinquete. |
| **Girar** (mano **derecha**) | **Swipe horizontal con la mano abierta**: le da vuelo con inercia. Cada swipe **suma** velocidad y uno en contra **frena**. El **regreso de la mano se ignora**. |
| **Agarrar la Tierra** (derecha) | **Puño**: la detiene y la arrastras; si abres la mano mientras la mueves, conserva el impulso. |
| **Pausar** | OK otra vez, barra espaciadora, o quitar las manos 6 s. |
| En pausa | La Tierra gira sola, despacio (modo de exhibición). Los gestos no la mueven. |

**Teclas:** Espacio activar/pausar · R reiniciar · I intercambiar manos (para zurdos, o si la detección izquierda/derecha sale al revés) · V cámara · H info · F pantalla completa · Q/Esc salir.

### 3.3 Estado de validación
- **17 pruebas automáticas sin cámara pasan** (`test_proyecto.py`). Usan manos 3D sintéticas y cubren poses, roles, activación, zoom, agarre e inercia, swipe, regreso de la mano, impulso previo ("windup"), saltos del rastreo, pausa automática, render y la integración de los 3 hilos.
- El modelo de MediaPipe carga y corre correctamente.
- **Pendiente: validarlo a fondo con cámara real.** No se ha confirmado:
  - cómo se sienten los umbrales del swipe;
  - si la etiqueta izquierda/derecha de MediaPipe sale correcta con la imagen en espejo (si sale al revés, existe la tecla I y `INVERTIR_MANOS`);
  - si el OK y el puño se detectan cómodamente.
- **Historial de problemas en Windows:**
  - Al primer intento salió `ModuleNotFoundError: cv2`, porque VS Code usó el Python global en vez del entorno virtual. Se resolvió creando `.venv` en la carpeta y agregando `.vscode/settings.json`.
  - En el prototipo 1, el pellizco con dos manos fue difícil de usar, y por eso se cambió el esquema.

## 4. Tecnologías y entorno

- **SO:** Windows (laptop HP Victus). Editor: **VS Code** con extensión de Python. Terminal: PowerShell.
- **Python 3.14** (instalado en `C:\Users\mendo\AppData\Local\Programs\Python\Python314`).
- **Entorno virtual:** `Proyecto Final\.venv`, ya creado e instalado. `.vscode/settings.json` apunta a `${workspaceFolder}\.venv\Scripts\python.exe`.
- **Librerías** (`requirements.txt`: opencv-python, mediapipe, numpy, pillow). Versiones instaladas: **mediapipe 1.2.3, opencv-python 5.1, numpy 2.5, pillow 13**.
- **Modelo:** `modelos/gesture_recognizer.task` (MediaPipe Gesture Recognizer, float16, ~8 MB). Se usa por sus **21 landmarks por mano**, en imagen y en 3D "world", y por su **handedness**. Las categorías de gesto que trae (Closed_Fist, Open_Palm…) **no se usan**: las poses se calculan con geometría propia. No incluye "OK".
- **Cámara:** índice 0, 640×480 a 60 FPS pedidos, MJPG, buffer de 1. La imagen se voltea en espejo **antes** de detectar, así que la etiqueta "Left" de MediaPipe corresponde a la mano izquierda real.
- **Render:** solo OpenCV + numpy, sin GPU. Ventana de OpenCV de 1280×720 internos, redimensionable y con pantalla completa.
- **Ejecutar:**
  ```powershell
  cd "C:\Users\mendo\Documents\UV\5TO SEMESTRE\INTERFACES AVANZADAS\Proyecto Final"
  .venv\Scripts\python.exe main.py            # opciones: --camara 1  --dshow  --pantalla-completa
  .venv\Scripts\python.exe pruebas\test_proyecto.py   # pruebas sin cámara
  ```

## 5. Estructura del código

```
Proyecto Final/
├── main.py                  entrada: ventana, teclado y los 3 hilos (captura, detección, dibujo)
├── requirements.txt · CLAUDE.md (debe quedarse en la raíz)
├── universo/                el programa (paquete)
│   ├── config.py            TODOS los parámetros ajustables (con guía de calibración al inicio)
│   ├── manos.py             filtro One Euro, seguimiento de 2 manos, rol izq/der, poses (puño, OK)
│   ├── control.py           máquina de estados: activación con OK, zoom, giro e inclinación (swipe, agarre, inercia)
│   ├── tierra.py            esfera 3D texturizada que gira y se inclina (normales 3D + mipmaps)
│   └── render.py            estrellas, Tierra, cursores, barra de zoom, HUD, textos (Pillow), miniatura de cámara
├── pruebas/
│   ├── test_proyecto.py     pruebas sin cámara (unitarias + integración real con hilos)
│   ├── simulacion.py        manos 3D sintéticas (formato idéntico a MediaPipe)
│   └── capturas/            PNG que generan las pruebas
├── assets/tierra_nasa_bluemarble.jpg   (4096×2048, equirectangular)
├── modelos/gesture_recognizer.task
├── docs/                    CONTEXTO_PROYECTO.md (este archivo) · LEEME.txt (instrucciones de uso)
├── .venv/ · .vscode/settings.json
```

**Separación de responsabilidades (importante mantenerla):**
- `manos.py` y `control.py` **solo dependen de numpy**: no saben nada de cámara, MediaPipe ni ventanas. Así se prueban sin cámara y se podrían reutilizar con otro renderizador (Three.js, Unity, etc.).
- `main.py` es el único que toca la cámara y MediaPipe (`DetectorMediaPipe`).
- El detector se puede sustituir por uno falso en las pruebas (`DetectorGuion`, `FuenteFalsa`).

## 6. Detalles técnicos clave (cómo y por qué)

### 6.1 Arquitectura de hilos (fluidez)
1. **Captura:** lee la cámara sin parar y guarda **solo el último cuadro**. Sin cola no se acumula retraso.
2. **Detección:** procesa siempre el cuadro más nuevo con MediaPipe (modo VIDEO, timestamps crecientes forzados), luego `SeguidorManos.actualizar()` y `Control.actualizar()`.
3. **Principal (dibujo):** dibuja a 60 FPS sin esperar a la cámara. `Control.avanzar()` integra la inercia a la frecuencia del dibujo, y la clase `Suave` (exponencial, τ = 0.045 s) suaviza escala y giro. Así se ve fluido aunque la detección vaya a 30 FPS.
- Todo el estado compartido pasa por `Control.lock`. El dibujo lee una `instantanea()` inmutable (namedtuples `Estado` y `ManoVista`).
- El modelo se carga con `model_asset_buffer`, en memoria, para evitar problemas con rutas con espacios o acentos en Windows.

### 6.2 Manos (`manos.py`)
- **Filtro One Euro** sobre los 21 puntos (MINCUT 0.6, BETA 8, DCUT 0.6): suaviza mucho con la mano quieta (~3.3× menos temblor) y casi no añade retraso al moverse rápido.
- **Seguimiento:** la identidad se asigna por cercanía a la **posición predicha** (posición + velocidad), así las manos se pueden cruzar sin intercambiarse. Si se pierde una mano, se conserva 0.25 s.
- **Rol izquierda/derecha:** etiqueta de MediaPipe suavizada (EMA) con histéresis; un rol solo cambia con evidencia fuerte. Si las dos manos dicen ser la misma, decide la posición en pantalla. `INVERTIR_MANOS` y la tecla I permiten intercambiarlas.
- **Poses, medidas en coordenadas 3D "world"** (invariantes a la distancia y a la orientación de la mano):
  - **Puño:** promedio de (punta−muñeca)/(nudillo−muñeca) en 4 dedos: ~2 con la mano abierta, ~1 con el puño. Activa < 1.30 y suelta > 1.55 (histéresis), con antirrebote de 50/80 ms.
  - **OK:** distancia pulgar-índice / tamaño de la palma < 0.33 (suelta > 0.48), **y** al menos 2 de medio/anular/meñique estirados (rectitud > 0.78). Excluye el puño.
  - **`soltando`:** en cuanto el puño empieza a abrirse (antes de confirmarse), se **congela** el zoom o el giro. Así, abrir la mano mientras se mueve no altera nada.
- Velocidad de la palma (centro de la muñeca y los 4 nudillos, que no cambia al cerrar la mano): una **filtrada** para los swipes y una **cruda** para detectar saltos del rastreo.

### 6.3 Control (`control.py`)
- **Activación:** el OK debe sostenerse 0.3 s, con tolerancia a caídas de 0.12 s. Después hay que deshacer el OK (0.25 s) antes de volver a usarlo, para que no alterne en bucle. Una mano recién aparecida (< 0.2 s) no cuenta.
- **Pausa automática** tras 6 s sin manos (para la feria: el siguiente visitante empieza en pausa).
- **Zoom:** `log_escala += −2.5 · Δy/alto` mientras el puño izquierdo está cerrado. Trabaja en logaritmo, que es simétrico y sirve para escalas cósmicas. Rango ×0.08 a ×6. Agarrar no produce saltos.
- **Giro, agarre:** Δgiro = Δx / R, con R = radio de la Tierra limitado a 150–450 px. Agarrar la detiene; al soltar hereda la velocidad del arrastre.
- **Giro, swipe** (máquina de estados `reposo → trazo → refractario / ignorando / salto`):
  - Cuenta como swipe si |vx| > **1.5 anchos de pantalla/s**, es mayormente horizontal (|vx| > 1.3·|vy|) y se mantiene 2 detecciones seguidas. Termina con |vx| < 0.6.
  - **Impulso proporcional y aditivo:** `omega = omega_inicio + dir · 4.5 · clamp((|v| − 0.75)/(3.5 − 0.75))`, con tope de 4.5 rad/s. Un swipe a favor suma y uno en contra frena, como un globo real.
  - **Regreso de la mano:** durante 0.45 s tras un swipe, uno en sentido contrario se ignora, **salvo** que sea 1.6× más rápido que el anterior. Así un "windup" (echar la mano atrás antes del swipe) no bloquea el swipe real.
  - **Saltos del rastreo:** si la velocidad **cruda** pasa de 9 anchos/s, se ignora hasta que la mano se calme. Hay que usar la cruda porque el One Euro convierte un salto en un "deslizamiento" que parecería swipe.
- **Inercia:** la fricción reduce la velocidad a la mitad cada 1.6 s. En pausa, la velocidad tiende a 0.07 rad/s (giro de exhibición).
- **Zona cómoda de la cámara:** el 80 % central del cuadro (margen 0.10) se estira a toda la pantalla, para llegar a las orillas sin sacar la mano.

### 6.4 Tierra (`tierra.py`): esfera 3D rápida en CPU
- Una **tabla precalculada** (float32, 2 canales) dice qué latitud/longitud de la textura se ve en cada punto del disco. **Girar = sumar un desplazamiento a la longitud**, con `BORDER_WRAP` en `cv2.remap`.
- Por cuadro: `warpAffine` de la tabla (escala/posición) → `cv2.transform` (aplica el giro y el tamaño de la textura) → `cv2.remap` de la textura BGRA (el canal A es la máscara de mar) → composición con iluminación precalculada.
- La **iluminación** (luz ×200 para permitir brillo > 1, reflejo, alfa y brillo de atmósfera) se guarda en un sprite de 4 canales. El color de la atmósfera se aplica con una **LUT**, porque multiplicar por escalares en cv2 es lento.
- **Mipmaps** de tablas (factor 0.707) y de textura (pyrDown con envoltura horizontal, sin costura en 180°). Se elige el nivel de modo que haya ~1 texel por píxel. Sin parpadeo al alejar.
- **Caché:** lo que solo depende del tamaño y la posición (tabla deformada, luz, alfa, atmósfera) se reutiliza mientras la Tierra solo gira. El radio se redondea a 0.05 px para aprovecharla.
- **Rendimiento medido** (contenedor de 2 hilos, más lento que la laptop): solo girando, 3–6 ms por cuadro; con zoom y la Tierra llenando la pantalla, ~15 ms; cuadro completo con interfaz ~8–11 ms.
- La foto Blue Marble tiene una franja oscura cerca de la Antártida (artefacto del original). Casi no se ve porque queda en el borde inferior.
- Si falta la textura, se genera un planeta procedural.

### 6.5 Interfaz (`render.py`)
- Estrellas generadas una vez (brillo sub-píxel, colores cálidos y fríos, halo en las brillantes).
- Textos con Pillow, convertidos en "sellos" en caché (LRU de 400) con contorno. Fuentes: Segoe UI / Semibold; DejaVu como respaldo.
- Cursores con precisión sub-píxel (`shift=4`, `LINE_AA`).

## 7. Decisiones tomadas (y por qué)

1. **Python + OpenCV + MediaPipe en CPU** para el prototipo, porque es lo que el equipo ya domina. Al crecer hacia el sistema solar y las galaxias **probablemente convenga migrar el render a GPU** (Three.js en navegador, con MediaPipe web, o Unity). `manos.py` y `control.py` están separados justamente para eso.
2. **Puño para "agarrar"** en ambas manos (es la pose más robusta), **OK solo para activar** y **swipe con la mano abierta** para dar vuelo.
3. **Rol por mano** (izquierda = zoom, derecha = giro) en vez de gestos con dos manos a la vez: el autor lo pidió por comodidad.
4. **Giro solo en el eje polar** (como un globo real), sin inclinar la Tierra. Esto evita además una costura visible de la textura.
5. **Sin nubes** en la textura, para que los continentes se vean claros (es educativo).
6. Ideas alternativas guardadas: **juego de hechizos con gestos** y **control de la PC con gestos ("Jarvis")**.

## 8. Pendientes y siguientes pasos

**Inmediato:**
- Probar con la cámara real y calibrar en `universo/config.py`: OK_ON, PUNO_ON, SWIPE_V_ON, GANANCIA_ZOOM.
- Confirmar que la detección izquierda/derecha sale bien.
- Medir los FPS reales en la laptop.

**Por decidir:**
- ¿El OK también pausa, o solo activa?
- ¿Se mantiene la pausa automática de 6 s?
- ¿Inclinar la Tierra con un swipe vertical?
- Gesto para "volver a la Tierra" o reiniciar.
- Gesto de traslado (mover la vista) que no se confunda con el zoom.
- Condiciones de la demo: un usuario a la vez, luz, distancia de 40–80 cm, ¿webcam externa?

**Hoja de ruta:**
1. Luna (órbita y fases).
2. Sistema Solar con escala logarítmica y regla de escala visible.
3. Misión del cometa (desviar con física real).
4. Fichas educativas al tocar objetos.
5. Galaxia, cúmulos y universo observable.
6. Easter egg del multiverso.

Antes del punto 2, decidir si se migra el render a GPU.

## 9. Convenciones del proyecto

- **Código, comentarios, nombres de variables e interfaz en español.** En los comentarios de código se evitan acentos por compatibilidad; en la interfaz sí se usan (gracias a Pillow).
- Todos los parámetros van en `config.py`, comentados y con guía de calibración.
- Cada cambio de gestos o control debe venir con su **prueba sin cámara** en `test_proyecto.py`, usando `simulacion.py`.
- No romper la regla de que `manos.py` y `control.py` solo dependen de numpy.
- Rendimiento: el dibujo debe sostener ~60 FPS. Las operaciones costosas se precalculan o se ponen en caché, y no hay bucles en Python por píxel.
- No tocar la carpeta `imAgen` (son prácticas anteriores de la materia).
