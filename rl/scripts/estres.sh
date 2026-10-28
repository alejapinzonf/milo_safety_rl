#!/usr/bin/env bash
# Prueba de estrés del punto ciego (E3b). Solo EVALÚA, no entrena. Desde rl/:
#   bash scripts/estres.sh            # 300 episodios por escenario (~10-20 min con 12 núcleos)
#   bash scripts/estres.sh 100        # versión rápida
# Compara, con y sin la memoria del escudo, en escenas con una caja baja que el planificador NO
# conoce (mapa desactualizado) sobre la ruta de salida: solo el escudo puede evitarla, también
# cuando el brazo la tapa. Resultados en resultados_estres/ (episodios.csv y resumen.csv).
set -e
cd "$(dirname "$0")/.."
N=${1:-300}
python3 scripts/evaluar.py --episodios "$N" --out resultados_estres \
  --escenarios ciego ciego_caminando \
  --solo reglas_tarde reglas_tarde_sin_memoria generalista generalista_sin_memoria
python3 scripts/estres_tabla.py resultados_estres/episodios.csv
