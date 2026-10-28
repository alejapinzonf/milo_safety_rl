#!/usr/bin/env bash
# Todos los entrenamientos del paper, en orden de prioridad. Desde rl/:
#   bash scripts/lote.sh            # todo
#   bash scripts/lote.sh principal  # solo el generalista (5 semillas) + evaluar + figuras
# Cada corrida se salta si ya existe su final.zip (se puede cortar con Ctrl+C y volver a lanzar).
# Tiempos aproximados con 12 núcleos: ~30-60 min por corrida de 2.75 M pasos.
set -e
cd "$(dirname "$0")/.."
SEMILLAS="0 1 2 3 4"
SEMILLAS_ABL="0 1 2"

corre() {  # corre <nombre_run> <semilla> <args...>
  local run=$1 s=$2; shift 2
  if [ -f "runs/$run/semilla_$s/final.zip" ]; then echo "ya está: $run s$s"; return; fi
  python3 scripts/entrenar.py --semilla "$s" "$@"
}

# 1) principal: generalista con currículo (las 3 etapas)
for s in $SEMILLAS; do corre generalista "$s"; done

if [ "$1" != "principal" ]; then
  # 2) especialistas (para la matriz entrenado-en x probado-en)
  for e in sin_personas quietas caminando; do
    for s in $SEMILLAS; do corre "especialista_$e" "$s" --especialista "$e"; done
  done
  # 3) ablaciones
  for s in $SEMILLAS_ABL; do
    corre mezcla_desde_cero "$s" --mezcla-desde-cero
    corre generalista_sin_escudo "$s" --sin-escudo
    corre generalista_sin_lagrangiano "$s" --sin-lagrangiano
    corre generalista_sin_memoria "$s" --sin-memoria
    corre generalista_sin_personas_obs "$s" --sin-personas-obs
    corre generalista_un_sector "$s" --un-sector
  done
fi

# 4) evaluación y figuras
python3 scripts/evaluar.py --episodios 300
python3 scripts/graficas.py
