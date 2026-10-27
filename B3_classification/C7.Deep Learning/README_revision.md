# Bloque de Deep Learning — revisión del material y propuesta para 3 sesiones

## 1. Qué hay en este paquete

| fichero | qué es |
|---|---|
| `DL1_NeuralNetworks_numpy.ipynb` | sesión 1 — redes desde cero, solo numpy |
| `DL2_DeepNetworks_PyTorch.ipynb` | sesión 2 — PyTorch y generalización |
| `DL3_ConvolutionalNetworks.ipynb` | sesión 3 — CNNs, augmentation, transfer learning |
| `deeplearning.pdf` / `.tex` | 57 transparencias para todo el bloque |
| `guion_dl.pdf` / `.tex` | guión de tres páginas, una por sesión, con los tiempos |
| `wooclap_deeplearning.md` | 24 preguntas de autoevaluación, 8 por sesión |
| `figs/`, `make_figs.py` | figuras de las transparencias |

Los notebooks **llevan las figuras incrustadas**, así que funcionan en Colab sin subir la
carpeta `figures/`. Solo necesitan la carpeta `data/` del material original (sesión 1);
las sesiones 2 y 3 descargan o generan lo que necesitan.

---

## 2. Un notebook por sesión, no uno solo

Lo he partido en tres. La razón decisiva es Colab: el entorno de ejecución se desconecta
entre semanas, así que un notebook único obligaría a reejecutar desde arriba cada sesión,
incluidos los bucles de entrenamiento de las sesiones anteriores. Con tres notebooks cada
sesión arranca limpia en dos minutos.

Además el notebook original tenía 154 celdas, y la parte 2 (PyTorch) solo necesitaba del
inicio la carga del dataset: tres celdas, que ahora están duplicadas al principio del
segundo notebook. El coste de partir era pequeño y el beneficio claro.

---

## 3. El orden: redes primero, PyTorch después

De acuerdo contigo, y no solo por costumbre. El argumento es que el notebook construye la
secuencia *topología → pérdida → forward → backward → actualización* a mano, y cuando
llega PyTorch cada pieza de la librería tiene un referente concreto: `backward()` sustituye
a una función que ellos escribieron, `nn.Linear` sustituye a una inicialización que ellos
eligieron, `optim.SGD` sustituye a dos líneas que ellos tecleaban. Si PyTorch va primero,
todo eso es vocabulario sin referente.

He reforzado esa idea en las transparencias de la sesión 2: el mismo modelo escrito cuatro
veces, subiendo un nivel de abstracción cada vez, señalando en cada peldaño qué se
esconde.

La planificación original dedicaba la sesión 1 a *Pytorch Introduction*. En el reparto
nuevo, los dos tutoriales oficiales de tensores y autograd quedan como **trabajo previo a
la sesión 2** (están enlazados al final del notebook 1, y son 30 minutos).

---

## 4. Errores encontrados

### 4.1. El gradiente de la capa oculta estaba mal

El más importante. En la red de 2 capas del notebook original:

* el texto decía que la activación era $h = \tanh(o)$, pero el código calculaba
  `h = 2*logistic(o) - 1`, que es $\tanh(o/2)$;
* y el backward usaba el factor $(1-h)^2$, cuando la derivada de la tangente hiperbólica
  es $1-h^2$.

Lo he comprobado numéricamente: el error relativo entre el gradiente analítico y el
numérico era de **0.58 en `W1` y 0.68 en `b1`** (el de la capa de salida sí era correcto,
error $2\cdot10^{-10}$). Con la corrección, los tres bajan a $\sim10^{-9}$.

Como ambos factores son positivos, el signo de cada componente se conservaba y la red
**entrenaba igualmente**, que es justo lo que hace que un gradiente mal derivado sobreviva
años. Pero explica dos cosas que estaban en el notebook como si fueran propiedades del
problema: que hiciera falta un paso de aprendizaje diez veces menor en la capa oculta
(`rho/10`), y los problemas de divergencia que menciona el ejercicio 6. Corregido, un único
$\rho$ sirve para las dos capas, y el mismo acierto se alcanza en la mitad de épocas
(dígitos binario: 0.952 con 2000 épocas y $\rho=0.5$, frente a 10000 épocas y
$\rho=0.001$).

He añadido una celda de **gradient checking** en el notebook y una transparencia sobre
ello. Me parece que el episodio da una clase mejor que la que daría el apartado si hubiera
estado bien desde el principio: un gradiente incorrecto no se delata en la curva de
pérdida.

### 4.2. Sección de transfer learning rota

En `C6_Classification_CNN`:

* `models.resnet18(pretrained=True)` está **deprecado** desde torchvision 0.13 → ahora
  `weights=models.ResNet18_Weights.DEFAULT`;
* `train_model` iteraba sobre `test_loader`, que no existe (el parámetro se llama
  `valid_loader`) → `NameError`;
* la última celda llamaba a `global_accuracy()` y `accuracy_per_class()`, que **no estaban
  definidas en ningún sitio**, y a `device`, tampoco definido;
* el backbone no se congelaba, así que se entrenaban los 11.2 M de parámetros;
* y con CIFAR10 completo a $224\times224$, 10 épocas y batch 4 no termina en una clase.

Reescrito: backbone congelado explícitamente (5130 parámetros entrenables, 0.05 %),
subconjunto de 4000/1000 imágenes, y las dos funciones de evaluación definidas.

### 4.3. Otros arreglos

* **Softmax sin estabilizar** (`np.exp(t)` y `t.exp()`): desborda con scores grandes. Ahora
  resta el máximo, con una nota explicando por qué no cambia el resultado.
* **Pérdida multiclase sin normalizar** (`-np.sum(y*log(q))` sin dividir por $K$): hacía
  que las curvas de las distintas secciones no fueran comparables entre sí.
* **Fuga de información en la normalización**: `MinMaxScaler` se ajustaba sobre todo el
  dataset *antes* de partir. Ahora se ajusta solo con train. El efecto numérico es
  pequeño, pero es incoherente con lo que les exigimos en el proyecto.
* `forze_binary` → `force_binary`; `np.save(X.npy',` (cadena rota); método `softmax`
  muerto dentro de `my_multiclass_net`; numeración 4.4.4 repetida dos veces.
* Comentario «la convergencia más rápida se debe a la inicialización de la capa oculta»:
  en esa red no hay capa oculta. Es la de salida.
* `fig.gca()`, `x.view(-1, 8*8*10)` → `torch.flatten(x, 1)`.
* Semillas fijas (`np.random.default_rng(42)`, `torch.manual_seed(42)`) para que los
  resultados se repitan de un año para otro.

---

## 5. Material nuevo

Lo mínimo para cerrar los dos «pending changes» que tenías apuntados, sin inventar temas:

* **Sección de generalización completa** en el notebook 2 (antes era solo texto): weight
  decay, dropout y early stopping, cada uno con su experimento y su comparación.
* **Data augmentation** en el notebook 3, con `torchvision.transforms` (5 líneas) y la
  comparación de curvas.
* Una función `train_net()` que encapsula el bucle, para no repetirlo seis veces — el otro
  punto de tu lista de pendientes.
* `gradient_check()` en el notebook 1.
* Ejercicios nuevos: 3 en el notebook 2 y 4 en el notebook 3.

**GANs fuera**, como pedías.

### Dos resultados que conviene conocer antes de clase

* **El weight decay no mejora el acierto en estos datos**: $0.792 \to 0.792$. Lo que sí
  hace es frenar el crecimiento de la pérdida de validación (1.08 → 0.79). El mensaje
  honesto es que la red regularizada no acierta más, se equivoca con menos confianza. Lo he
  escrito así en el notebook y en las transparencias.
* **El early stopping sobre la pérdida selecciona la época 139, con acierto 0.760**,
  mientras que el mejor acierto (0.792) llega mucho más tarde. He añadido un parámetro
  `monitor` para que puedan comparar ambos criterios. Es un buen apartado para discutir.

---

## 6. Reparto del tiempo

El detalle está en `guion_dl.pdf`. El esquema es el mismo en las tres sesiones: bloques de
teoría de 8–14 minutos alternados con bloques de notebook de 9–15, y las transparencias
llevan el ritmo: cada bloque de prácticas tiene su propia transparencia naranja con lo que
hay que hacer y cuántos minutos hay.

| | teoría | notebook | sesión |
|---|---|---|---|
| S1 — redes desde cero | 45 min | 39 min (3 bloques) | + 6 cierre |
| S2 — PyTorch | 47 min | 41 min (4 bloques) | |
| S3 — CNNs | 48 min | 42 min (3 bloques) | |

Aproximadamente mitad y mitad, que es lo que pedías. Dos avisos operativos:

* En la sesión 3 **hay que decir lo del entorno GPU al empezar**, y lanzar la descarga de
  CIFAR10 en los primeros minutos.
* En la sesión 2 los tres experimentos de generalización tardan varios minutos cada uno;
  el guión los coloca al final a propósito, para que arranquen mientras se comenta el
  cierre y terminen de mirarlos en casa.

---

## 7. Lo que no he podido comprobar aquí

Los notebooks 1 y 2 se ejecutan completos en este entorno (todas las celdas de código,
saltando los ejercicios). Del notebook 3 he verificado la red, el bucle de entrenamiento,
las funciones de evaluación, el conteo de parámetros de ResNet18 y que
`ResNet18_Weights.DEFAULT` existe en la versión actual de torchvision — pero **no la
descarga de CIFAR10 ni la de los pesos preentrenados**, porque la salida de red de este
contenedor no llega a esos dominios. Conviene que lo ejecutes una vez en Colab de principio
a fin antes de la sesión 3.
