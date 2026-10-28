"""Todos los números del simulador 2D, del escudo y del entrenamiento, en un solo lugar.

Las medidas del robot salen del URDF de Milo (andesrobot_description) y del filtro de seguridad
real (andesrobot_safety/config/safety_nav.yaml). Las del brazo, del informe del brazo
(docs/informe): tabla "Espacio de trabajo y rango con el lift". Si cambia el robot, se cambia AQUÍ.

Unidades: metros, segundos, radianes. Frame del robot = base_footprint (+x adelante, +y izquierda).
"""
from dataclasses import dataclass, field, replace

# ------------------------------------------------------------------------------------------------
# Robot
# ------------------------------------------------------------------------------------------------
FP_X_MIN, FP_X_MAX, FP_HW = -0.12, 0.44, 0.25  # huella de la base (caja), igual que safety.yaml
BASE_Z = 0.374  # alto de la base (hasta el lidar): lo que puede chocar la base
LIDAR_X, LIDAR_Z = 0.408, 0.363  # frame "laser" del URDF: 0.408 m adelante, plano a 0.363 m
LIDAR_RAYOS = 180  # el C1 real tiene 500; 180 (2°) alcanza en 2D y es 3x más rápido
LIDAR_RANGO = 6.0  # [m] el C1 llega a 12, pero en una pieza de casa no hace falta más
LIDAR_SIGMA = 0.01  # [m] ruido de cada rayo

# Dinámica de la base (diferencial). El escudo SUPONE que frena con A_ESCUDO; la planta real (el
# simulador) frena con A_FRENO >= A_ESCUDO: así la garantía es conservadora.
V_MAX, W_MAX = 0.40, 1.0
A_LIN, A_ANG = 0.8, 2.0  # aceleración máx (subir velocidad)
A_FRENO = 1.0  # desaceleración máx de la planta
DT = 0.1  # paso del simulador = periodo del lidar (10 Hz)
SUBPASOS = 2  # la política decide cada SUBPASOS * DT = 0.2 s (5 Hz)

# ------------------------------------------------------------------------------------------------
# Brazo (modelo 2D del brazo de 6 GDL + lift)
# ------------------------------------------------------------------------------------------------
HOMBRO_X = 0.1225  # eje de joint_1 en base_footprint
HOMBRO_Z = (0.613, 1.613)  # alto del hombro (joint_2) con el lift abajo / arriba
RADIO_BRAZO = 0.848  # distancia máx hombro -> pinza (informe)
MARGEN_ALCANCE = 0.03  # agarrar de verdad pide unos cm menos que el brazo estirado
GROSOR_BRAZO = 0.06  # radio de la cápsula con que se modela el brazo en 2D
# Pose de carga BAJA (la que deja el brazo después de agarrar): pinza adelante y abajo, cruza el
# plano del lidar -> lo tapa ("la garra cae"). Pose ALTA (después de "despejar"): brazo recogido
# arriba de la base, no tapa nada y no sale de la huella.
CARGA_BAJA_TCP = (0.62, 0.0)  # [m] punto de la pinza en base_footprint
CARGA_BAJA_Z = (0.28, 0.55)  # [m] alturas que ocupa el brazo en esa pose
GUARDADO_TCP = (0.30, 0.0)
T_ALCANZAR, T_AGARRAR, T_DESPEJAR = 2.0, 1.0, 2.0  # [s] duración de cada movimiento del brazo

# ------------------------------------------------------------------------------------------------
# Personas
# ------------------------------------------------------------------------------------------------
PIERNA_R, PIERNA_SEP = 0.07, 0.11  # radio de cada pierna y su separación del centro
PIERNA_Z = (0.0, 0.85)
TORSO_R, TORSO_Z = 0.22, (0.85, 1.75)
V_HUMANO_MAX = 1.0  # [m/s] cota de velocidad de una persona caminando en casa (para el escudo)


@dataclass
class ParamsEscudo:
    """Parámetros del escudo (la parte certificada; el PPO NO los toca)."""

    d_min_mueble: float = 0.05  # piso duro frente a muebles y muros [m]
    a_frenado: float = 0.8  # desaceleración que se supone [m/s²] (<= A_FRENO)
    t_reaccion: float = 0.2  # T_r: latencia lidar + control [s]
    # Separación ISO/TS 15066 frente a personas:
    #   S_p = v_h (T_r + T_s) + v_r T_r + v_r²/(2a) + C + Z_d + Z_r
    t_parada: float = 0.3  # T_s: tiempo que tarda en detenerse del todo [s]
    c_intrusion: float = 0.25  # C: el torso sobresale de las piernas que ve el lidar [m]
    z_detector: float = 0.15  # Z_d = 3 sigma del detector de personas [m]
    z_robot: float = 0.05  # Z_r: error de posición del robot [m]
    holgura_giro: float = 0.08  # igual que rotation_clearance de safety_nav.yaml
    margen_lateral: float = 0.10  # side_margin
    # Memoria del punto ciego: r_u = sigma0 + k_s * recorrido + k_psi * |giro| * distancia
    mem_sigma0: float = 0.02
    mem_k_s: float = 0.03
    mem_k_psi: float = 0.02
    mem_ventana: float = 6.0  # [s] cuánto se guardan los scans
    activo: bool = True  # False = ablación "sin escudo"
    usar_memoria: bool = True  # False = ablación "sin memoria" (lo tapado se ignora)


@dataclass
class RangoAccion:
    """A qué valores físicos se mapea la acción del PPO (cada componente en [-1, 1])."""

    d_stop: tuple = (0.0, 0.40)  # sin escudo puede llegar a 0: el escudo lo sube a d_min
    d_slow_extra: tuple = (0.10, 0.60)  # d_slow = d_stop + esto
    v_max: tuple = (0.10, 0.40)


# Filtro "de fábrica" (safety_nav.yaml): el nominal contra el que se mide la exposición
D_STOP_NOMINAL, D_SLOW_NOMINAL = 0.30, 0.70


@dataclass
class Escenario:
    """Qué se sortea en cada episodio. Las 3 pruebas del paper son 3 combinaciones de esto."""

    p_sin_personas: float = 1.0
    p_quietas: float = 0.0
    p_caminando: float = 0.0
    n_personas: tuple = (1, 3)  # cuántas, si hay (sorteado)
    v_caminar: tuple = (0.3, 1.0)
    # dónde está el objeto: mesa rectangular, redonda, estante o suelo
    p_mueble: dict = field(default_factory=lambda: {
        'mesa': 0.35, 'redonda': 0.25, 'estante': 0.30, 'suelo': 0.10})
    p_facil: float = 0.25  # objeto alcanzable con el filtro nominal (el resto: hay que relajar)
    n_distractores: tuple = (1, 4)  # muebles extra en la pieza
    sigma_objeto: float = 0.02  # error de la "percepción" del objeto [m]
    sigma_persona: float = 0.05  # error del detector de personas [m]
    p_perder_persona: float = 0.05  # prob. de que el detector no la vea en un scan
    # Prueba de estrés del punto ciego: caja baja (0.35 x 0.35 m, cruza el plano del lidar) que NO
    # está en el mapa del planificador, sobre la ruta a la salida. Solo se sortea si p_oculto > 0.
    p_oculto: float = 0.0
    planificador_ve_oculta: bool = True  # False: el ejecutivo NO la esquiva (solo el escudo la ve)


ESCENARIOS = {
    'sin_personas': Escenario(),
    'quietas': Escenario(p_sin_personas=0.0, p_quietas=1.0),
    'caminando': Escenario(p_sin_personas=0.0, p_caminando=1.0),
}


# Escenarios SOLO de evaluación (E3b): el obstáculo no mapeado queda sobre la ruta de salida y el
# planificador NO lo esquiva (mapa desactualizado): solo el escudo puede evitarlo, también cuando el
# brazo lo tapa. No se entrena con ellos.
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
    progreso: float = 0.5  # shaping por acercarse a la sub-meta (potencial)
    ciego: float = 0.0  # castigo extra por moverse con el lidar tapado (0: lo decide el escudo)
    # Costos (restricciones del CMDP, con presupuesto por episodio). Ver lagrangiano.py.
    # Calibrados con los baselines (evaluar.py): "reglas" gasta ~0.4-0.8 de exposición por
    # episodio y "minimo" ~10; con personas, "reglas" ~1.5-1.7 de costo social.
    presupuesto_exp: float = 0.5  # exposición: (|v|/v_max) * relajación relativa, integrada [s]
    presupuesto_soc: float = 1.2  # comodidad: cercanía (< 1.2 m) * velocidad, integrada [s]


@dataclass
class ConfigEntorno:
    escenario: Escenario = field(default_factory=Escenario)
    escudo: ParamsEscudo = field(default_factory=ParamsEscudo)
    accion: RangoAccion = field(default_factory=RangoAccion)
    recompensa: Recompensa = field(default_factory=Recompensa)
    t_max: float = 90.0  # [s] duración máx del episodio
    usar_personas_en_obs: bool = True  # False = ablación "sin datos de personas"
    un_sector: bool = False  # True = ablación: un solo d_stop para todo


def mezcla(pesos):
    """Escenario mezcla: pesos = {'sin_personas': p0, 'quietas': p1, 'caminando': p2}."""
    tot = sum(pesos.values())
    return replace(Escenario(), p_sin_personas=pesos.get('sin_personas', 0) / tot,
                   p_quietas=pesos.get('quietas', 0) / tot,
                   p_caminando=pesos.get('caminando', 0) / tot)


# Currículo del generalista: cada etapa parte del modelo de la anterior. Se repiten episodios de
# las etapas previas (30 %) para que no se le olvide lo aprendido.
CURRICULO = [
    dict(nombre='etapa1_objetos', pasos=1_000_000, escenario=mezcla({'sin_personas': 1})),
    dict(nombre='etapa2_quietas', pasos=750_000,
         escenario=mezcla({'sin_personas': 0.3, 'quietas': 0.7})),
    dict(nombre='etapa3_caminando', pasos=1_000_000,
         escenario=mezcla({'sin_personas': 0.15, 'quietas': 0.15, 'caminando': 0.7})),
]
