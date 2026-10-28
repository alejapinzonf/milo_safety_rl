# rl/ — Filtro de seguridad adaptativo con PPO (paper)

Simulador 2D de Milo + PPO-Lagrangiano con escudo certificado. No necesita ROS, Gazebo ni
Docker: es Python puro y corre directo en tu PC. La matemática y qué archivo implementa cada
pieza está en [METODO.md](METODO.md).

```
rl/
├── milo_rl/            el simulador y el algoritmo
│   ├── config.py       TODOS los números (robot, brazo, escudo, recompensas, presupuestos, currículo)
│   ├── geometria.py    formas con altura, rayos del lidar, choques
│   ├── brazo.py        alcance del brazo, sus 2 sombras (tapa el lidar / agranda la huella)
│   ├── humanos.py      personas (quietas / caminando) y detector de piernas
│   ├── escena.py       escenas al azar: pieza, mueble, objeto, sillas, personas; A*
│   ├── memoria.py      memoria del punto ciego (Proposición 3)
│   ├── escudo.py       rampa sigmoide + escudo (Proposiciones 1, 2 y 4)
│   ├── entorno.py      entorno gymnasium + ejecutivo de la tarea
│   ├── baselines.py    nominal, mínimo, reglas
│   ├── lagrangiano.py  PPO-Lagrangiano (ascenso dual de los lambdas)
│   └── viz.py          dibujos y GIFs
├── scripts/
│   ├── entrenar.py     entrena (currículo, especialistas, ablaciones)
│   ├── evaluar.py      evalúa todos los métodos con las mismas escenas -> CSV
│   ├── graficas.py     figuras + tabla LaTeX del paper
│   ├── ver_episodio.py GIF de un episodio
│   └── lote.sh         TODO el plan de entrenamiento en orden
└── tests/              pytest (incluye la prueba "con escudo no choca nunca")
```

## 1. Instalar (una vez)

Usa un entorno aparte, para no tocar el Python del sistema ni el de ROS. Con Anaconda:

```bash
conda create -n milo_rl python=3.11 -y
conda activate milo_rl
cd ~/milo_paper/rl
pip install -r requirements.txt
```

Sin conda funciona igual con `python3 -m venv ~/venvs/milo_rl && source ~/venvs/milo_rl/bin/activate`.

> **¿GPU?** No hace falta. La red es chica (2 × 128) y el cuello de botella es el simulador, que
> corre en CPU: lo que importa es **cuántos núcleos** tienes. Si quieres ahorrarte los ~2 GB de
> CUDA de torch, antes del `pip install -r` corre
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## 2. Ver que todo anda (5 minutos)

```bash
conda activate milo_rl && cd ~/milo_paper/rl
python3 -m pytest -q tests                     # 17 pruebas, ~20 s
python3 scripts/entrenar.py --prueba           # 20 mil pasos por las 3 etapas, ~2 min
python3 scripts/ver_episodio.py --metodo reglas --escenario caminando --episodio 3
#   -> resultados/gifs/reglas_caminando_3.gif  (ábrelo para ver el simulador)
```

En el GIF se ve:
- **Robot:** la caja azul. El brazo es la cápsula naranja.
- **Lidar:** rayos azules. En rojo, los que tapa el brazo, a la distancia que certifica la memoria.
- **Personas:** en morado; la línea punteada es por donde caminan, y el círculo punteado lo que
  ve el detector.
- **Objeto:** la estrella. La x es dónde cree la percepción que está.
- **Salida:** el cuadrado verde.

## 3. Entrenar

**Lo principal: el generalista con currículo, 5 semillas, más la evaluación y las figuras:**

```bash
bash scripts/lote.sh principal
```

**Todo el paper (generalista + especialistas + ablaciones + evaluación):**

```bash
bash scripts/lote.sh
```

Detalles:
- Cada corrida usa `núcleos − 1` entornos en paralelo. Con 12 núcleos, una corrida de 2.75 M
  pasos (las 3 etapas) tarda unos 30–60 min (acá, con 2 núcleos, el 25 % tardó 50 min).
- `lote.sh` se salta lo que ya terminó: lo puedes cortar con Ctrl+C y volver a lanzar.
- Si tienes muchos núcleos, puedes correr dos semillas a la vez en dos terminales, con
  `--n-envs` a la mitad, por ejemplo:
  `python3 scripts/entrenar.py --semilla 0 --n-envs 6` y
  `python3 scripts/entrenar.py --semilla 1 --n-envs 6`.

**Corridas sueltas:**

```bash
python3 scripts/entrenar.py --semilla 0                 # generalista (etapas 1-3)
python3 scripts/entrenar.py --semilla 0 --etapa 3       # solo la etapa 3 (parte de la 2)
python3 scripts/entrenar.py --semilla 0 --especialista caminando
python3 scripts/entrenar.py --semilla 0 --sin-escudo    # u otra ablación (ver --help)
```

**Qué mirar mientras entrena.** Imprime una línea por rollout:

```
[  215040] éxito 0.92  choque 0.00  exp 0.33  soc 0.00  lambda (4.18, 0.00)
```

| Campo | Qué esperar |
|---|---|
| `éxito` | ciclos completos (agarra y llega a la salida). Tiene que subir o mantenerse alto |
| `choque` | **0 siempre con el escudo**. Si no, hay un bug: mándame el log |
| `exp` | exposición por episodio. Tiene que bajar hasta el presupuesto (0.5) o menos |
| `soc` | costo social; solo en las etapas con personas |
| `lambda` | multiplicadores del Lagrangiano: suben si se pasa del presupuesto, bajan si le sobra |

Lo mismo queda en `runs/<corrida>/semilla_<s>/progreso.csv`.

## 4. Evaluar y figuras

```bash
python3 scripts/evaluar.py --episodios 300   # baselines + todo lo de runs/, mismas escenas
python3 scripts/graficas.py                  # resultados/figs/: fig_resultados, fig_entrenamiento, tabla.tex
python3 scripts/ver_episodio.py --modelo runs/generalista/semilla_0/final.zip --escenario quietas
```

## 5. Perillas (todo en `milo_rl/config.py`)

| Qué | Dónde |
|---|---|
| Presupuestos del CMDP | `Recompensa.presupuesto_exp` (0.5) y `presupuesto_soc` (1.2) |
| Pasos de cada etapa del currículo | `CURRICULO` |
| Personas: cuántas, velocidad, ruido del detector | `Escenario` (`n_personas`, `v_caminar`, `sigma_persona`) |
| Proporción de objetos fáciles / difíciles / por mueble | `Escenario.p_facil`, `p_mueble` |
| Parámetros del escudo (ISO, memoria, piso duro) | `ParamsEscudo` |

## 6. Resultados de referencia (prueba corta hecha acá)

Corrí el generalista con solo el **25 % de los pasos** (690 mil, con 2 núcleos, ~50 min) y lo
evalué con `evaluar.py` en 60 episodios por escenario, con las mismas escenas para todos. Cada
celda es éxito / exposición:

| | Sin personas | Personas quietas | Personas caminando |
|---|---|---|---|
| Fijo nominal | 0.17 / 0 | 0.18 / 0 | 0.17 / 0 |
| Fijo mínimo | 0.93 / 11.8 | 0.70 / 11.0 | 0.67 / 9.8 |
| Reglas a mano | 0.92 / 0.76 | 0.58 / 0.47 | 0.63 / 0.55 |
| **PPO-Lag + escudo** | **0.93 / 0.15** | **0.67 / 0.12** | **0.70 / 0.17** |

**Choques, contactos y violaciones ISO: 0 en todos los métodos.**

- El nominal falla en los objetos "difíciles", que son el 75 %.
- El mínimo los agarra, pero va expuesto todo el tiempo.
- El PPO llega al **éxito del mínimo con ~1 % de su exposición y 3–4 veces menos que las
  reglas**.
- El costo social todavía está igual al de los baselines (~1.8–2.1). Su lambda recién estaba
  subiendo: con el entrenamiento completo debería bajar. Si no baja, se sube
  `presupuesto_soc` o se baja la velocidad cerca de personas en `Recompensa`.

Con personas, el éxito de TODOS baja (0.6–0.7): una parte de los episodios termina "atascado",
porque el escudo ISO no deja pasar cerca de alguien. Eso no depende del PPO.

## 6b. Prueba de estrés de la memoria (E3b)

En las 3 escenas normales la memoria del punto ciego casi no se nota: el planificador conoce el
mapa y el brazo se sube rápido. Para ponerla a prueba hay dos escenas extra, **solo de evaluación**
(`ciego`, `ciego_caminando`): una caja baja (cruza el plano del lidar) que el planificador NO
conoce, sobre la ruta de salida; solo el escudo puede evitarla, también cuando el brazo la tapa.
Se compara con y sin memoria usando una política de reglas que sube el brazo 4 s tarde
(`reglas_tarde`), y las políticas entrenadas (`generalista` vs `generalista_sin_memoria`):

```bash
bash scripts/estres.sh          # evalúa y saca la tabla (con p de Fisher)
python3 scripts/ver_episodio.py --metodo reglas_tarde_sin_memoria --escenario ciego --episodio 11
```

Con 300 episodios (2 núcleos, acá): `reglas_tarde` **0 choques** con memoria contra **10 (3.3 %)**
sin memoria en `ciego`, y 0 contra 3 en `ciego_caminando`. El éxito es ~2 % para todos porque el
planificador no esquiva la caja: la métrica de esta prueba son los choques, no el éxito.

## 7. Limitaciones que conviene saber

- **Fallas que no dependen del PPO.** Algunos episodios fallan por "agarre_fallido": la
  percepción del objeto se equivocó 2–3 cm. Esto les pasa a todos los métodos por igual.
- **Choques del brazo al manipular.** Los choques del brazo con muebles mientras alcanza o agarra
  no se cuentan: eso lo resuelve el planificador del brazo, no el filtro de la base. Los del brazo
  con personas sí se cuentan siempre.
- **Validación en Gazebo (E5).** Ver [LEEME_GAZEBO.md](LEEME_GAZEBO.md): un nodo ROS corre el
  filtro + escudo + PPO sobre la física de Gazebo (brazo bloqueado, personas quietas).
