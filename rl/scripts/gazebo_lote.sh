#!/usr/bin/env bash
# Validación en Gazebo (E5): todos los métodos, una corrida de Gazebo por método. Desde la raíz del
# repo o desde rl/ (en tu PC, no dentro del contenedor):
#   bash rl/scripts/gazebo_lote.sh <mapa> [mundo] [veces] [objetos]
#   bash rl/scripts/gazebo_lote.sh casa_v2                          # casa sin personas
#   bash rl/scripts/gazebo_lote.sh casa_v2 robocup_casa_personas    # casa con 3 personas quietas
# Cada corrida visita los objetos en orden, `veces` veces, y guarda un CSV en ros2_ws/resultados/.
# Orden: baselines (nominal, mínimo, reglas) y el PPO con las 5 semillas (rl/politicas/).
set -u
cd "$(dirname "$0")/../.."
MAPA=${1:?uso: gazebo_lote.sh <mapa> [mundo] [veces] [objetos]}
MUNDO=${2:-robocup_casa}
VECES=${3:-1}
OBJ=${4:-botella_laundry,taza_laundry,galletas_laundry,taza_estante,botella_estante,galletas_librero,botella_librero,taza_comedor,botella_comedor,galletas_comedor}
export MILO_GUI=false MILO_RVIZ=false
for modo in nominal minimo reglas; do
  echo "=== $modo"; bash ./sim.sh rl "$MAPA" "$modo" "$OBJ" "$VECES" "$MUNDO" 0 || true
  bash ./sim.sh stop
done
for s in 0 1 2 3 4; do
  echo "=== ppo semilla $s"; bash ./sim.sh rl "$MAPA" ppo "$OBJ" "$VECES" "$MUNDO" "$s" || true
  bash ./sim.sh stop
done
python3 rl/scripts/gazebo_tabla.py ros2_ws/resultados/rl_*_"$MUNDO"_*.csv
