"""Todos los números del simulador 2D, del escudo y del entrenamiento, en un solo lugar.

Las medidas del robot salen del URDF de Milo (andesrobot_description) y del filtro de seguridad
real (andesrobot_safety/config/safety_nav.yaml). Las del brazo, del informe del brazo
(docs/informe): tabla "Espacio de trabajo y rango con el lift". Si cambia el robot, se cambia AQUÍ.

Unidades: metros, segundos, radianes. Frame del robot = base_footprint (+x adelante, +y izquierda).
"""
from dataclasses import dataclass, field, replace

FP_X_MIN, FP_X_MAX, FP_HW = -0.12, 0.44, 0.25
BASE_Z = 0.374
LIDAR_X, LIDAR_Z = 0.408, 0.363
LIDAR_RAYOS = 180
LIDAR_RANGO = 6.0
LIDAR_SIGMA = 0.01

V_MAX, W_MAX = 0.40, 1.0
A_LIN, A_ANG = 0.8, 2.0
A_FRENO = 1.0
DT = 0.1
SUBPASOS = 2

HOMBRO_X = 0.1225
HOMBRO_Z = (0.613, 1.613)
RADIO_BRAZO = 0.848
MARGEN_ALCANCE = 0.03
GROSOR_BRAZO = 0.06
CARGA_BAJA_TCP = (0.62, 0.0)
CARGA_BAJA_Z = (0.28, 0.55)
GUARDADO_TCP = (0.30, 0.0)
T_ALCANZAR, T_AGARRAR, T_DESPEJAR = 2.0, 1.0, 2.0

PIERNA_R, PIERNA_SEP = 0.07, 0.11
PIERNA_Z = (0.0, 0.85)
TORSO_R, TORSO_Z = 0.22, (0.85, 1.75)
V_HUMANO_MAX = 1.0


@dataclass
class ParamsEscudo:
    """Parámetros del escudo (la parte certificada; el PPO NO los toca)."""

    d_min_mueble: float = 0.05
    a_frenado: float = 0.8
    t_reaccion: float = 0.2
    t_parada: float = 0.3
    c_intrusion: float = 0.25
    z_detector: float = 0.15
    z_robot: float = 0.05
    holgura_giro: float = 0.08
    margen_lateral: float = 0.10
    mem_sigma0: float = 0.02
    mem_k_s: float = 0.03
    mem_k_psi: float = 0.02
    mem_ventana: float = 6.0
    activo: bool = True
    usar_memoria: bool = True


@dataclass
class RangoAccion:
    """A qué valores físicos se mapea la acción del PPO (cada componente en [-1, 1])."""

    d_stop: tuple = (0.0, 0.40)
    d_slow_extra: tuple = (0.10, 0.60)
    v_max: tuple = (0.10, 0.40)


D_STOP_NOMINAL, D_SLOW_NOMINAL = 0.30, 0.70


@dataclass
class Escenario:
    """Qué se sortea en cada episodio. Las 3 pruebas del paper son 3 combinaciones de esto."""

    p_sin_personas: float = 1.0
    p_quietas: float = 0.0
    p_caminando: float = 0.0
    n_personas: tuple = (1, 3)
    v_caminar: tuple = (0.3, 1.0)
    p_mueble: dict = field(default_factory=lambda: {
        'mesa': 0.35, 'redonda': 0.25, 'estante': 0.30, 'suelo': 0.10})
    p_facil: float = 0.25
    n_distractores: tuple = (1, 4)
    sigma_objeto: float = 0.02
    sigma_persona: float = 0.05
    p_perder_persona: float = 0.05
    p_oculto: float = 0.0
    planificador_ve_oculta: bool = True


ESCENARIOS = {
    'sin_personas': Escenario(),
    'quietas': Escenario(p_sin_personas=0.0, p_quietas=1.0),
    'caminando': Escenario(p_sin_personas=0.0, p_caminando=1.0),
}


ESTRES = {
    'ciego': Escenario(p_oculto=1.0, planificador_ve_oculta=False),
    'ciego_caminando': Escenario(p_sin_personas=0.0, p_caminando=1.0, p_oculto=1.0,
                        planificador_ve_oculta=False),
}
TODOS = {**ESCENARIOS, **ESTRES}


@dataclass
class Recompensa:
    agarrar: float = 10.0
    llegar: float = 5.0
    choque: float = -10.0
    fallar: float = -5.0
    paso: float = -0.01
    progreso: float = 0.5
    ciego: float = 0.0
    presupuesto_exp: float = 0.5
    presupuesto_soc: float = 1.2


@dataclass
class ConfigEntorno:
    escenario: Escenario = field(default_factory=Escenario)
    escudo: ParamsEscudo = field(default_factory=ParamsEscudo)
    accion: RangoAccion = field(default_factory=RangoAccion)
    recompensa: Recompensa = field(default_factory=Recompensa)
    t_max: float = 90.0
    usar_personas_en_obs: bool = True
    un_sector: bool = False


def mezcla(pesos):
    """Escenario mezcla: pesos = {'sin_personas': p0, 'quietas': p1, 'caminando': p2}."""
    tot = sum(pesos.values())
    return replace(Escenario(), p_sin_personas=pesos.get('sin_personas', 0) / tot,
                   p_quietas=pesos.get('quietas', 0) / tot,
                   p_caminando=pesos.get('caminando', 0) / tot)


CURRICULO = [
    dict(nombre='etapa1_objetos', pasos=1_000_000, escenario=mezcla({'sin_personas': 1})),
    dict(nombre='etapa2_quietas', pasos=750_000,
         escenario=mezcla({'sin_personas': 0.3, 'quietas': 0.7})),
    dict(nombre='etapa3_caminando', pasos=1_000_000,
         escenario=mezcla({'sin_personas': 0.15, 'quietas': 0.15, 'caminando': 0.7})),
]
