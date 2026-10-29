#!/usr/bin/env python3
"""Figuras y tabla del paper a partir de resultados/episodios.csv y runs/*/progreso.csv.

Uso (desde rl/):  python3 scripts/graficas.py
Salida en resultados/figs/:
  fig_resultados.pdf/png  éxito, exposición, costo social y violaciones ISO por método y escenario
                          (media e intervalo de confianza del 95 %, bootstrap)
  fig_entrenamiento.pdf   curvas de entrenamiento (éxito, exposición, lambdas) por corrida
  tabla.tex               tabla principal en LaTeX
"""
import glob
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = os.path.dirname(AQUI)
RES = os.path.join(RAIZ, 'resultados')
FIGS = os.path.join(RES, 'figs')
ORDEN = ['nominal', 'minimo', 'reglas', 'generalista_sin_escudo', 'generalista_sin_lagrangiano',
         'generalista']
NOMBRES = {'nominal': 'Fijo nominal', 'minimo': 'Fijo mínimo', 'reglas': 'Reglas',
           'generalista_sin_escudo': 'PPO sin escudo',
           'generalista_sin_lagrangiano': 'PPO + escudo (λ fijo)',
           'generalista': 'PPO-Lag + escudo (nuestro)'}
ESC = ['sin_personas', 'quietas', 'caminando']
ESC_N = {'sin_personas': 'Sin personas', 'quietas': 'Personas quietas',
         'caminando': 'Personas caminando'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def ic95(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    b = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def fig_resultados(df):
    metricas = [('exito', 'Tasa de éxito'), ('exposicion', 'Exposición [s]'),
                ('costo_soc', 'Costo social [s]'), ('violaciones_iso', 'Violaciones ISO / ep.')]
    metodos = [m for m in ORDEN if m in df.metodo.unique()] + \
        sorted(set(df.metodo.unique()) - set(ORDEN))
    fig, axs = plt.subplots(1, len(metricas), figsize=(7.2, 2.4))
    ancho = 0.8 / len(metodos)
    colores = plt.cm.tab10(np.arange(len(metodos)))
    for ax, (col, tit) in zip(axs, metricas):
        for j, m in enumerate(metodos):
            for i, e in enumerate(ESC):
                d = df[(df.metodo == m) & (df.escenario == e)]
                if d.empty:
                    continue
                mu = d[col].mean()
                lo, hi = ic95(d[col].values)
                ax.bar(i + (j - len(metodos) / 2 + 0.5) * ancho, mu, ancho, color=colores[j],
                       yerr=[[mu - lo], [hi - mu]], capsize=1.5, error_kw=dict(lw=0.6),
                       label=NOMBRES.get(m, m) if (i == 0 and ax is axs[0]) else None)
        ax.set_xticks(range(len(ESC)))
        ax.set_xticklabels([ESC_N[e].replace(' ', '\n', 1) for e in ESC], fontsize=6.5)
        ax.set_title(tit, fontsize=8)
        if col == 'exposicion':
            ax.set_yscale('symlog', linthresh=1.0)
    fig.legend(loc='upper center', ncol=min(len(metodos), 3), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGS, f'fig_resultados.{ext}'), bbox_inches='tight', dpi=200)
    plt.close(fig)


def fig_entrenamiento():
    rutas = sorted(glob.glob(os.path.join(RAIZ, 'runs', '*', 'semilla_*', 'progreso.csv')))
    if not rutas:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
    runs = {}
    for r in rutas:
        runs.setdefault(r.split(os.sep)[-3], []).append(pd.read_csv(r))
    for k, (run, dfs) in enumerate(runs.items()):
        c = plt.cm.tab10(k % 10)
        for d in dfs:
            axs[0].plot(d.pasos / 1e6, d.exito, color=c, lw=0.6, alpha=0.6)
            axs[1].plot(d.pasos / 1e6, d.exposicion, color=c, lw=0.6, alpha=0.6)
            axs[2].plot(d.pasos / 1e6, d.lambda_exp, color=c, lw=0.6, alpha=0.6,
                        label=run if d is dfs[0] else None)
    axs[0].set_title('Éxito')
    axs[1].set_title('Exposición por episodio')
    axs[2].set_title('λ exposición')
    for ax in axs:
        ax.set_xlabel('pasos [M]')
    axs[2].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_entrenamiento.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'fig_entrenamiento.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)


def tabla(df):
    metodos = [m for m in ORDEN if m in df.metodo.unique()]
    lin = [r'\begin{tabular}{l' + 'cccc' * len(ESC) + '}', r'\toprule',
           ' & ' + ' & '.join(r'\multicolumn{4}{c}{%s}' % ESC_N[e] for e in ESC) + r' \\',
           'Método' + ' & Éxito & Exp. & Soc. & ISO' * len(ESC) + r' \\', r'\midrule']
    for m in metodos:
        celdas = []
        for e in ESC:
            d = df[(df.metodo == m) & (df.escenario == e)]
            if d.empty:
                celdas += ['--'] * 4
                continue
            celdas += [f'{d.exito.mean():.2f}', f'{d.exposicion.mean():.2f}',
                       f'{d.costo_soc.mean():.2f}', f'{d.violaciones_iso.mean():.2f}']
        lin.append(NOMBRES.get(m, m) + ' & ' + ' & '.join(celdas) + r' \\')
    lin += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(FIGS, 'tabla.tex'), 'w') as fh:
        fh.write('\n'.join(lin) + '\n')


def main():
    os.makedirs(FIGS, exist_ok=True)
    ruta = os.path.join(RES, 'episodios.csv')
    if os.path.exists(ruta):
        df = pd.read_csv(ruta)
        fig_resultados(df)
        tabla(df)
    else:
        print('No hay', ruta, '(corre antes scripts/evaluar.py)')
    fig_entrenamiento()
    print('Figuras en', FIGS)


if __name__ == '__main__':
    main()
