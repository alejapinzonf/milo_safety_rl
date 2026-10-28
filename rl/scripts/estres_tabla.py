#!/usr/bin/env python3
"""Tabla de la prueba de estrés: choques con y sin memoria (y p de Fisher). Uso:
  python3 scripts/estres_tabla.py resultados_estres/episodios.csv"""
import sys

import pandas as pd
from scipy.stats import fisher_exact

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'resultados_estres/episodios.csv')
d['choque'] = (d['motivo'] == 'choque').astype(int)
d['con_memoria'] = ~d['metodo'].str.contains('sin_memoria')
d['familia'] = d['metodo'].str.replace('_sin_memoria', '')
filas = []
for (fam, esc), g in d.groupby(['familia', 'escenario']):
    a, b = g[g.con_memoria], g[~g.con_memoria]
    if len(a) == 0 or len(b) == 0:
        continue
    p = fisher_exact([[a.choque.sum(), len(a) - a.choque.sum()],
                      [b.choque.sum(), len(b) - b.choque.sum()]])[1]
    filas.append(dict(familia=fam, escenario=esc, n_con=len(a), choques_con_memoria=a.choque.sum(),
                      n_sin=len(b), choques_sin_memoria=b.choque.sum(),
                      contacto_caja_con=a.choque_oculta.sum(), contacto_caja_sin=b.choque_oculta.sum(),
                      t_ciego_mov_con=round(a.t_ciego_mov.mean(), 2), p_fisher=round(p, 4)))
t = pd.DataFrame(filas)
print(t.to_string(index=False))
t.to_csv(sys.argv[1].rsplit('/', 1)[0] + '/estres_tabla.csv', index=False)
