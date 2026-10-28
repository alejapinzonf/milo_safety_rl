# Método: filtro de seguridad adaptativo con escudo certificado (lo que implementa `rl/`)

Borrador de la sección *Método* del paper, con cada pieza enlazada al código que la implementa.
Notación: frame del robot = `base_footprint` (+x adelante). Todas las medidas salen de
`milo_rl/config.py`.

## 1. Problema

Milo (base diferencial + lift + brazo de 6 GDL) hace un ciclo de agarre en una casa: navega hasta
un mueble, se acerca, agarra un objeto, retrocede y sigue a otro lugar. Su filtro de seguridad
(`andesrobot_safety`) frena con una rampa entre `d_stop = 0.30 m` y `d_slow = 0.70 m`, medidas
desde el borde de la huella `F_base = [-0.12, 0.44] × [-0.25, 0.25]`.

Con el alcance del brazo, `x_alc(z) = 0.1225 + r_h(z) - 0.03` (≈ 0.94 m a la altura de una
mesa), un objeto a profundidad `p` dentro del mueble solo se alcanza si
`p ≤ x_alc(z) - 0.44 - d_stop`. Con el filtro nominal eso es `p ≤ 0.20 m`; con el piso duro
(`d_min = 0.05 m`), `p ≤ 0.45 m`. Hay que relajar el filtro para agarrar, pero relajarlo siempre
expone al robot (y a las personas) sin necesidad.

Hay dos complicaciones más:

- **Punto ciego del brazo.** Después de agarrar, la pose de carga deja la pinza adelante y abajo,
  cruzando el plano del lidar (0.363 m). Eso tapa un cono frontal de unos 90°.
- **Personas**, quietas o caminando, que el filtro debe respetar siempre.

## 2. Qué decide la política (acción) y qué no

Cada 0.2 s la política elige `θ = (d_stop^f, d_stop^s, Δ_slow, v_max)` y una acción discreta
`despejar` (subir el brazo a la pose guardada, 2 s con la base quieta). La política **no maneja**:
el ejecutivo (A* + pure pursuit + secuencia de la tarea) pide `(v, ω)` y el filtro lo recorta.
`entorno.py`, `MiloEnv.accion_a_theta`.

## 3. Rampa sigmoide (`escudo.rampa`)

```
s(d) = 0                                       si d ≤ d_stop
s(d) = clip( (σ(k(d − d_mid)) − 0.05) / 0.9 )  si d_stop < d < d_slow
s(d) = 1                                       si d ≥ d_slow
d_mid = (d_stop + d_slow)/2,   k = 2 ln 19 / (d_slow − d_stop)
```

`σ` vale 0.05 en `d_stop` y 0.95 en `d_slow`, así que `s` es continua, monótona y sin quiebres.
Hay dos sectores: `d_stop^f` adelante y `d_stop^s` atrás/costados.

## 4. Escudo (lo que el PPO no puede tocar)

**Proyección.** `d_stop ← max(d_stop, d_min)`, `Δ_slow ≥ 0.05`, `v_max ≤ 0.4`
(`escudo.proyectar`).

**Proposición 1 (muebles).** Con desaceleración garantizada `a = 0.8 m/s²` y latencia
`T = 0.2 s`, si en cada paso

```
|v| ≤ v_cap(d) = −aT + √(a²T² + 2a (d − d_min))
```

(`d` es la distancia libre en el corredor de la huella, en la dirección de avance), entonces la
huella nunca queda a menos de `d_min` de un obstáculo estático. *Idea de la prueba:* `v_cap`
resuelve `vT + v²/2a = d − d_min`. Si se cumple en `k`, el robot puede frenar antes de `d_min`;
el escudo lo vuelve a imponer en `k+1`. Por inducción, el conjunto `{d ≥ d_min}` es invariante.
La planta frena con 1.0 m/s² ≥ `a`, así que la cota es conservadora
(`test_v_cap_frena_a_tiempo`).

**Proposición 2 (personas, ISO/TS 15066, SSM).**

```
S_p(v_r) = v_h (T_r + T_s + Δt) + v_r (T_r + Δt) + v_r²/2a + C + Z_d + Z_r
```

con `T_s = 0.3 s`, `C = 0.25 m` (el torso sobresale de las piernas que ve el lidar),
`Z_d = 3σ_det = 0.15 m`, `Z_r = 0.05 m` y `Δt` el paso de control (condición en tiempo
discreto). Se exige `d_h ≥ S_p(v_r)`, de donde sale otra `v_cap`. Para `v_h`:

- si en la escena se vio a alguien caminar: `v_h = V_max = 1.0 m/s` (como la ISO, que fija una
  velocidad humana) más `V_max · edad` de la detección;
- si nadie camina: la velocidad estimada (regresión sobre 1 s de detecciones);
- una persona recién detectada (< 1 s) se trata como si caminara.

Con error del detector ≤ 3σ, la condición se cumple con probabilidad ≥ 1 − P(|ε| > 3σ). En la
evaluación se mide con la posición REAL, sin `Z_d` (`violaciones_iso`).

**Proposición 3 (punto ciego).** Para cada rayo tapado por el brazo se toma la observación más
reciente en esa dirección (en el frame de odometría) y se pone un punto virtual a distancia

```
d_k = r_visto − r_u − e_h
r_u = σ0 + k_s·(recorrido desde que se vio) + k_ψ·|giro|·r_visto     (deriva de odometría)
e_h = V_max·(t − t_visto)   si hay gente caminando                     (erosión)
```

Si nunca se vio, el punto queda pegado al robot. Con eso, la Proposición 1 vale también en el
cono ciego. Una persona que entra al cono por un lado cruza rayos visibles: se detecta y queda
cubierta por la Proposición 2. Si entra por el fondo, tiene que recorrer `r_visto`, y eso lo
cubre la erosión. `memoria.py`.

**Proposición 4 (huella del brazo).** `F(q) = F_base ∪ cápsula(hombro → pinza)`. El corredor
usa la punta y el ancho de `F(q)`, y el giro (en el lugar, o siempre que el brazo esté afuera)
solo se permite si no hay obstáculos dentro de `r_giro(q) + 0.08`, ni personas dentro de
`r_giro(q) + S_p(0)`. Con el brazo guardado `r_giro = 0.51 m`; en carga baja, ≈ 0.68 m.
`brazo.py`, `escudo.filtrar`.

**Verificación empírica.** Con el escudo, ninguna política (aleatoria o entrenada) choca:
`test_escudo_sin_choques_con_acciones_al_azar`, y 0 contactos y 0 violaciones ISO en la
evaluación de los baselines.

## 5. CMDP y PPO-Lagrangiano (`lagrangiano.py`)

```
max_π  E[Σ γ^t r_t]   s.a.   E[C_exp] ≤ β_exp,   E[C_soc] ≤ β_soc
```

- `r`: +10 agarrar, +5 llegar, −10 chocar, −5 fallar, −0.01 por paso, más un *shaping* de
  potencial (distancia a la sub-meta).
- `C_exp = ∫ (|v|/v_max) · max(0, d_nom − d_stop)/d_nom dt`: exposición, es decir, moverse con el
  filtro relajado.
- `C_soc = ∫ Σ_h max(0, 1.2 − d_h)/1.2 · (0.2 + |v|/v_max) dt`: comodidad, o sea, moverse rápido
  dentro del espacio personal de Hall (1.2 m).
- Presupuestos: `β_exp = 0.5`, `β_soc = 1.2`, calibrados contra los baselines.
- PPO maximiza `r − λ_exp c_exp − λ_soc c_soc`. Después de cada rollout, ascenso dual:
  `λ ← max(0, λ + η (Ĵ_c − β))`, con `η = 0.05`.

**Observación (97):**

- el lidar en 16 sectores, la fracción tapada por sector y la distancia certificada por la memoria;
- el estado del brazo y `r_giro(q)`;
- la fase de la tarea;
- el objeto según la percepción (pose + error σ = 2 cm) y el margen de alcance;
- las 2 personas más cercanas según el detector (pose, velocidad, edad);
- los comandos;
- la acción anterior.

## 6. Escenas y currículo (`escena.py`, `config.py`)

Pieza de 6–8 × 5–7 m; un mueble objetivo (mesa rectangular, redonda, estante o suelo) con
sillas y faldones; 1–4 muebles extra. El objeto (botella, taza o galletas) se ubica por
**profundidad efectiva**, medida desde el borde accesible más cercano para que dar la vuelta no
ayude: 25 % alcanzable con el filtro nominal y 75 % solo relajando. Personas: 0, o 1–3 quietas,
o 1–3 caminando ida y vuelta (0.3–1.0 m/s, no esquivan, se dan vuelta si el robot les tapa el
paso más de 1.5 s).

Currículo del generalista:

1. solo objetos (1 M pasos);
2. 30 % sin personas + 70 % quietas (0.75 M);
3. 15 % + 15 % + 70 % caminando (1 M).

Cada etapa parte de la anterior, y la repetición de escenarios previos evita el olvido.

## 7. Experimentos (`scripts/lote.sh`)

| | Qué | Comando |
|---|---|---|
| E1 | baselines vs PPO en los 3 escenarios | `evaluar.py` |
| E2 | generalista vs especialistas (matriz) | `entrenar.py --especialista ...` |
| E3 | ablaciones: sin escudo, sin λ (fijo), sin memoria, sin personas en la obs., un sector, sin currículo | flags de `entrenar.py` |
| E3b | prueba de estrés del punto ciego: caja no mapeada, con y sin memoria | `scripts/estres.sh` |
| E4 | ruido del detector / percepción | cambiar `Escenario.sigma_*` y `p_perder_persona` |
| E5 | validación en Gazebo (casa), sim-to-sim | `./sim.sh rl ...`, `scripts/gazebo_lote.sh` (ver LEEME_GAZEBO.md) |

Métricas:

- éxito, tiempo;
- choques, contactos;
- violaciones e intrusiones ISO;
- exposición, costo social;
- tiempo moviéndose ciego, despejes, intervenciones del escudo.
