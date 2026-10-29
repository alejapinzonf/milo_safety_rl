# Validación en Gazebo (E5)

Cómo se corre el filtro adaptativo del paper (filtro + escudo + PPO) sobre la física de Gazebo, en
la casa RoboCup@Home. Es una validación **sim-to-sim**: el mismo código que se entrenó en el
simulador 2D, pero con la física, el lidar (500 rayos, con su ruido) y la odometría de Gazebo.

## Qué se valida y qué no

| Sí | No (se queda en el simulador 2D) |
|---|---|
| El PPO elige `d_stop` adelante / atrás-costados, `Δ_slow` y `v_max` cada 0.2 s | El brazo en movimiento: en la sim de navegación va **bloqueado** (ver `NOTAS.md`) |
| Rampa sigmoide + escudo (`v_cap`, separación ISO/TS 15066) sobre `/scan` real de Gazebo | El punto ciego y la memoria (E3b) |
| Nav2 lleva a Milo al pre-agarre; después orienta y se acerca hasta tener el objeto al alcance | Personas caminando: ahora sí, mundo `robocup_casa_caminando` (ver abajo) |
| Comparación con nominal / mínimo / reglas en las mismas condiciones | |
| Personas **quietas** de human-gazebo, detectadas solo si el lidar las ve | |

Como el brazo está bloqueado, "alcanzar" se mide con la misma fórmula del 2D: el objeto está al
alcance cuando `margen_alcance >= 0` (0.97 m de alcance desde el hombro, ver `rl/milo_rl/brazo.py`).

## Cómo funciona

```
Gazebo ─ /scan, /odom, TF ─┐
Nav2 ─ /nav/cmd_vel ─ twist_mux ─ /safety/cmd_vel_in ─┤  rl/ros/nodo_rl.py ── /cmd_vel ─ Gazebo
Nav2 ◄── NavigateToPose (pre-agarre de cada objeto) ──┘   (reemplaza al safety_filter)
```

El nodo es solo pegamento con ROS; la lógica está en `rl/milo_rl/adaptador.py` (observación de 97
números, filtro + escudo) y `rl/milo_rl/mision.py` (la máquina de estados). Ambos son Python puro y
`rl/tests/test_adaptador.py` comprueba que dan **lo mismo que el simulador 2D** con el mismo scan.
La política corre en numpy (sin torch) a partir de un `.npz`.

Por cada objeto de la lista: `navegando` (Nav2 al pre-agarre) → `acercando` (orienta y avanza
hasta `margen_alcance >= 0`) → `agarre` (1 s quieto; acá agarraría el brazo) → `retrocediendo`
0.6 m → siguiente objeto. Si algo falla (Nav2 no llega en 150 s, o no alcanza en 60 s) se anota
`falla_nav` / `falla_acercar` y sigue con el siguiente.

## Antes de correr (una vez)

1. **Mapa de la casa nueva.** La casa cambió (estantes, faldones, mesas grandes), así que hay que
   volver a mapearla: `./sim.sh casa` → `./sim.sh teleop` → `./sim.sh mapa casa_v2`.
   (Los lugares de `casa_lugares.yaml` no se usan en esta prueba: el pre-agarre de cada objeto sale
   de `worlds/robocup_casa_objetos.yaml`, campo `pre`.)
2. **Políticas.** Copia los `.npz` a `rl/politicas/` (`politica_s0.npz` … `politica_s4.npz`). Salen
   de los modelos entrenados con
   `python3 rl/scripts/exportar_politica.py runs/generalista/semilla_0/final.zip rl/politicas/politica_s0.npz`
   (comprueba que coincide con SB3).
3. **Contenedor.** `rl/` ahora se monta en el contenedor como `/rl` (`docker/compose.yaml`), así
   que hay que recrearlo: `./sim.sh stop` y volver a lanzar.
4. Solo para el mundo con personas: `python3 ros2_ws/src/andesrobot_gazebo/tools/poner_personas.py`
   (ya viene generado `worlds/robocup_casa_personas.world`).

## Correr

Una prueba rápida, con ventanas, un objeto:

```bash
./sim.sh rl casa_v2 ppo taza_comedor 1          # PPO, semilla 0
./sim.sh rl casa_v2 reglas taza_comedor 1       # o nominal | minimo | reglas
```

Sintaxis: `./sim.sh rl <mapa> <modo> [objetos,con,comas] [veces] [mundo] [semilla]`. En otra
terminal, `ros2 topic echo /rl/estado` muestra la fase, el margen de alcance y los márgenes que
elige el PPO. Cuando termina la lista, el nodo apaga el launch solo. El resultado queda en
`ros2_ws/resultados/rl_<modo>_<mundo>_s<semilla>_<fecha>.csv`.

Todo el experimento (3 baselines + PPO con 5 semillas, sin ventanas; ~1-2 h):

```bash
bash rl/scripts/gazebo_lote.sh casa_v2                          # sin personas
bash rl/scripts/gazebo_lote.sh casa_v2 robocup_casa_personas    # con 3 personas quietas
bash rl/scripts/gazebo_lote.sh casa_v2 robocup_casa_caminando   # con 3 personas caminando
python3 rl/scripts/gazebo_tabla.py ros2_ws/resultados/rl_*.csv  # tabla -> resumen_gazebo.csv
```

## Qué se mide (una fila por objeto)

`resultado` (exito | falla_nav | falla_acercar), `t_total`, `margen_final`, `exposicion` y
`costo_soc` (las mismas fórmulas del 2D), `d_min_mueble` y `d_min_persona` [m], `choque` (1 si la
huella llegó a < 1 cm de algo que ve el lidar), `intervenciones` del escudo, `violaciones_iso` e
`intrusiones_iso`. Lo esperable, si la validación sale como el 2D: éxito parecido entre PPO,
mínimo y reglas, **exposición del PPO muy menor** (≈ 0.2 contra ≈ 8 del mínimo y ≈ 0.9 de las
reglas), 0 choques y 0 violaciones ISO.

## Si algo falla

- `falla_nav` en todos los objetos: Nav2 no llega al pre-agarre. Mira en RViz si el mapa quedó
  bien alineado con el mundo y si el pre-agarre cae sobre celdas ocupadas del costmap (habría que
  remapear con más cuidado el borde de las mesas).
- `No existe rl/politicas/politica_s0.npz`: falta el paso 2.
- `ModuleNotFoundError: yaml` / `nav2_msgs`: el nodo corre dentro del contenedor, no en tu PC.
- El robot no se mueve y `/rl/estado` dice `navegando`: Nav2 aún no publica en
  `/safety/cmd_vel_in` (espera a que AMCL localice; mira que el robot esté donde parte en RViz).

## Personas que caminan (`robocup_casa_caminando`)

- `python3 ros2_ws/src/andesrobot_gazebo/tools/poner_caminantes.py` genera
  `worlds/robocup_casa_caminando.{world,yaml}` (ya vienen generados). Hay 3 caminantes; cada uno
  va y viene a 0.8 m/s (con 1 s de pausa al dar la vuelta) por un tramo libre de muebles que
  **cruza el camino al pre-agarre** de `taza_comedor`, `botella_laundry` y `botella_librero`.
- Cada caminante es un modelo cinemático (cilindro de colisión de 0.20 m de radio y 1.7 m de
  alto, así el lidar lo ve + figura humana solo visual). `nodo_rl.py` lo mueve con el servicio
  `/gazebo/set_entity_state` (plugin `gazebo_ros_state` del mundo) a 20 Hz y empiezan a caminar
  cuando arranca la misión. La fase de la ida y vuelta depende de la semilla.
- El filtro recibe la posición y velocidad **reales** de cada persona solo para medir
  (exposición, ISO, `d_min_persona`); para decidir usa el detector con ruido que ya existía
  (la persona cuenta como vista si el lidar tiene un punto a < 0.30 m de ella).
- Nav2 las ve como obstáculos dinámicos del lidar (costmap local). Milo no las esquiva "a
  propósito" con el PPO: el PPO solo ajusta los márgenes del filtro; esquivar es de Nav2.
- Probar: `MILO_GUI=true ./sim.sh rl casa_v2 ppo taza_comedor 1 robocup_casa_caminando 0`
