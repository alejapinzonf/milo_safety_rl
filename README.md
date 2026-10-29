# milo_safety_rl

Código, resultados y texto del paper: **márgenes de seguridad adaptativos para un manipulador móvil
(Milo, AndesRobot) con PPO-Lagrangiano y un escudo certificado**. Solo simulación.

```
rl/        simulador 2D, filtro + escudo, PPO-Lagrangiano, baselines, ablaciones, tests
           (LEEME.md = uso, METODO.md = matemáticas ↔ código, LEEME_GAZEBO.md = validación en Gazebo)
ros2_ws/   workspace ROS 2 Humble + Gazebo Classic de Milo (la casa RoboCup@Home v2 y Nav2)
docker/, sim.sh             cómo se levanta todo
results/   resultados que salen en el paper (principal/, estres/, gazebo/) con sus figuras
paper/     LaTeX del paper (IEEEtran, 6 páginas)
```

## Reproducir

```bash
cd rl && pip install -r requirements.txt && python3 -m pytest -q tests      # 23 tests
bash scripts/lote.sh                      # entrenamiento + evaluación (ver LEEME.md)
bash scripts/estres.sh                    # prueba de estrés del punto ciego (E3b)
```

Validación en Gazebo (E5, necesita Docker): `./sim.sh rl casa_v2 ppo taza_comedor 1` y
`bash rl/scripts/gazebo_lote.sh casa_v2` (detalles en `rl/LEEME_GAZEBO.md`). El mapa `casa_v2`
se crea una vez: `./sim.sh casa` → `./sim.sh teleop` → `./sim.sh mapa casa_v2`.

## Notas

- `rl/politicas/politica_s{0..4}.npz`: las 5 semillas del generalista exportadas a numpy.
- `results/*/episodios.csv.gz`: una fila por episodio (los `resumen.csv` salen de ahí).
- `poner_personas.py` necesita `human_gazebo` y `humanos.py` (de terceros, no incluidos); el
  mundo ya generado `robocup_casa_personas.world` sí viene y no los necesita.
