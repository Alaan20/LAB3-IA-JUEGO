"""
=============================================================================
  Laboratorio N° 3 - Aprendizaje por Refuerzo
  Inteligencia Artificial - Unidad N° 4
  Métodos Basados en Modelo (Value Iteration) y Libres de Modelo (Q-Learning)
=============================================================================

ALGORITMOS IMPLEMENTADOS:
  1. Value Iteration (Iteración de Valor) — Método Basado en Modelo
  2. Q-Learning                           — Método Libre de Modelo

ENTORNOS:
  - FrozenLake-v1 (modo determinístico y estocástico)
  - Taxi-v4 (entorno adicional)

ECUACIONES CLAVE:
  - Bellman: V*(s) = max_a { Σ P(s'|s,a) * [R(s,a,s') + γ * V*(s')] }
  - Q-Learning: Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
=============================================================================
"""

import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Usar backend sin pantalla para servidor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import time
import os

# ─────────────────────────────────────────────────────────
#   DIRECTORIO DE SALIDA
# ─────────────────────────────────────────────────────────
DIRECTORIO_GRAFICAS = "/home/claude/graficas_lab3"
os.makedirs(DIRECTORIO_GRAFICAS, exist_ok=True)


# =============================================================================
#   PUNTO 1-A: VALUE ITERATION (Iteración de Valor)
#   Método BASADO EN MODELO — necesita conocer env.P
# =============================================================================


def value_iteration(entorno, gamma=0.99, theta=1e-8):
    """
    Implementa el algoritmo Value Iteration (Iteración de Valor).

    Fundamento teórico:
    -------------------
    Aplica la ecuación de Bellman iterativamente sobre todos los estados
    hasta convergencia. En cada barrida actualiza:

        V(s) = max_a { Σ P(s'|s,a) · [R + γ · V(s')] }

    Al converger, extrae la política óptima:
        π*(s) = argmax_a Q(s,a)

    Parámetros:
    -----------
    entorno : gym.Env  — entorno Gym con acceso a entorno.P (modelo del entorno)
    gamma   : float    — factor de descuento (0 < γ ≤ 1)
    theta   : float    — criterio de convergencia (umbral de cambio mínimo)

    Retorna:
    --------
    V       : np.ndarray — función de valor óptima para cada estado
    politica: np.ndarray — política óptima (acción por estado)
    iteraciones: int     — cantidad de iteraciones hasta convergencia
    """

    # ── Paso 1: Obtener dimensiones del espacio de estados y acciones ──────
    num_estados = entorno.observation_space.n
    num_acciones = entorno.action_space.n

    # ── Paso 2: Inicializar la función de valor en 0 para todos los estados ─
    V = np.zeros(num_estados)  # V[s] = valor del estado s

    # Acceder a la dinámica del entorno (env.unwrapped.P en Gymnasium moderno)
    # P[estado][accion] = lista de (probabilidad, sig_estado, recompensa, terminal)
    modelo = entorno.unwrapped.P

    iteraciones = 0  # Contador de barridas (sweeps)

    # ── Paso 3: Bucle principal hasta convergencia ─────────────────────────
    while True:
        delta = 0.0  # Máximo cambio en esta barrida (criterio de parada)
        iteraciones += 1

        # Recorrer todos los estados
        for estado in range(num_estados):
            valor_anterior = V[estado]  # Guardar valor previo para calcular Δ

            # Calcular el valor Q para cada acción posible en este estado
            # modelo[estado][accion] = lista de (prob, sig_estado, recompensa, terminal)
            valores_q = np.zeros(num_acciones)
            for accion in range(num_acciones):
                for prob_transicion, sig_estado, recompensa, es_terminal in modelo[
                    estado
                ][accion]:
                    # Ecuación de Bellman: suma sobre estados siguientes ponderada por probabilidad
                    # En modo determinístico: prob_transicion = 1.0 siempre
                    # En modo estocástico:   prob_transicion < 1.0 (el entorno puede "resbalar")
                    valores_q[accion] += prob_transicion * (
                        recompensa + gamma * V[sig_estado]
                    )

            # Actualizar V(s) con el máximo Q-valor (acción óptima)
            V[estado] = np.max(valores_q)

            # Actualizar el cambio máximo de esta barrida
            delta = max(delta, abs(valor_anterior - V[estado]))

        # ── Verificar convergencia: si Δ < θ, detener ─────────────────────
        if delta < theta:
            break

    # ── Paso 4: Extraer la política óptima a partir de V ──────────────────
    politica = np.zeros(num_estados, dtype=int)
    for estado in range(num_estados):
        # Q(s,a) = Σ P(s'|s,a) · [R + γ · V(s')]
        valores_q = np.zeros(num_acciones)
        for accion in range(num_acciones):
            for prob_transicion, sig_estado, recompensa, es_terminal in modelo[estado][
                accion
            ]:
                valores_q[accion] += prob_transicion * (
                    recompensa + gamma * V[sig_estado]
                )

        # La política elige la acción que maximiza Q(s,a)
        politica[estado] = np.argmax(valores_q)

    return V, politica, iteraciones


# =============================================================================
#   PUNTO 1-B: Q-LEARNING
#   Método LIBRE DE MODELO — off-policy con exploración ε-greedy
# =============================================================================


def q_learning(
    entorno,
    episodios=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    decaimiento_epsilon=0.999,
    epsilon_min=0.01,
):
    """
    Implementa el algoritmo Q-Learning (aprendizaje por diferencia temporal).

    Fundamento teórico:
    -------------------
    Técnica off-policy que aprende la función Q sin necesitar el modelo.
    El agente interactúa con el entorno y actualiza:

        Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') - Q(s,a) ]
                              └──────────── error TD ─────────────┘

    La política ε-greedy balancea exploración y explotación:
        - Con probabilidad  ε : acción aleatoria (EXPLORACIÓN)
        - Con probabilidad 1-ε: argmax Q(s,.) (EXPLOTACIÓN / greedy)

    Al inicio ε=1.0 (exploración pura), decae hacia ε_min (explotación).

    Parámetros:
    -----------
    entorno           : gym.Env — entorno a resolver
    episodios         : int     — cantidad de episodios de entrenamiento
    alpha             : float   — tasa de aprendizaje (qué tan rápido actualiza Q)
    gamma             : float   — factor de descuento (importancia de recompensas futuras)
    epsilon           : float   — probabilidad inicial de exploración (1.0 = 100% aleatorio)
    decaimiento_epsilon: float  — factor multiplicativo de decaimiento de ε por episodio
    epsilon_min       : float   — valor mínimo de ε (no baja de aquí)

    Retorna:
    --------
    tabla_Q      : np.ndarray  — tabla Q aprendida [estados × acciones]
    politica     : np.ndarray  — política derivada de Q (argmax por estado)
    recompensas  : list        — recompensa total obtenida por episodio
    epsilons     : list        — valor de ε en cada episodio (curva de exploración)
    """

    # ── Paso 1: Dimensiones del espacio ────────────────────────────────────
    num_estados = entorno.observation_space.n
    num_acciones = entorno.action_space.n

    # ── Paso 2: Inicializar la tabla Q en ceros ────────────────────────────
    # tabla_Q[estado, accion] = valor Q estimado de tomar `accion` en `estado`
    tabla_Q = np.zeros((num_estados, num_acciones))

    recompensas = []  # Historial de recompensas por episodio
    epsilons = []  # Historial del valor ε por episodio

    # ── Paso 3: Bucle de entrenamiento — episodios ─────────────────────────
    for episodio in range(episodios):
        estado, _ = entorno.reset()  # Reiniciar entorno, obtener estado inicial
        recompensa_total = 0.0
        terminado = False

        # ── Bucle de pasos dentro del episodio ────────────────────────────
        while not terminado:
            # ── Política ε-greedy: selección de acción ─────────────────
            if np.random.random() < epsilon:
                # EXPLORACIÓN: acción aleatoria
                accion = entorno.action_space.sample()
            else:
                # EXPLOTACIÓN: acción con mayor Q en el estado actual
                accion = np.argmax(tabla_Q[estado, :])

            # ── Ejecutar acción en el entorno ───────────────────────────
            sig_estado, recompensa, terminado, truncado, _ = entorno.step(accion)
            terminado = terminado or truncado
            recompensa_total += recompensa

            # ── Actualización Q-Learning (ecuación de Bellman off-policy) ─
            # Objetivo TD = r + γ · max_a' Q(s', a')
            # Error TD    = Objetivo - Q(s, a)   ← cuánto me equivoqué
            objetivo_td = recompensa + gamma * np.max(tabla_Q[sig_estado, :]) * (
                not terminado
            )
            error_td = objetivo_td - tabla_Q[estado, accion]
            tabla_Q[estado, accion] += alpha * error_td

            estado = sig_estado  # Avanzar al siguiente estado

        # ── Registrar métricas del episodio ────────────────────────────────
        recompensas.append(recompensa_total)
        epsilons.append(epsilon)

        # ── Decaer ε: ir de exploración a explotación gradualmente ────────
        epsilon = max(epsilon_min, epsilon * decaimiento_epsilon)

    # ── Paso 4: Extraer política final: π(s) = argmax_a Q(s,a) ───────────
    politica = np.argmax(tabla_Q, axis=1)

    return tabla_Q, politica, recompensas, epsilons


# =============================================================================
#   UTILIDADES DE EVALUACIÓN Y VISUALIZACIÓN
# =============================================================================

SIMBOLOS_ACCIONES = {
    # FrozenLake: 0=Izquierda, 1=Abajo, 2=Derecha, 3=Arriba
    "frozen_lake": {0: "←", 1: "↓", 2: "→", 3: "↑"},
    # # Taxi: 0=Sur, 1=Norte, 2=Este, 3=Oeste, 4=Recoger, 5=Dejar
    # "taxi": {0: "↓", 1: "↑", 2: "→", 3: "←", 4: "P", 5: "D"},
}

MAPA_FROZEN_LAKE = ["SFFF", "FHFH", "FFFH", "HFFG"]


def evaluar_politica(entorno, politica, episodios_prueba=100):
    """
    Evalúa una política ejecutándola en el entorno sin exploración.

    Retorna:
    --------
    tasa_exito : float — porcentaje de episodios ganados (0.0 a 1.0)
    recompensa_media : float — recompensa promedio por episodio
    """
    exitos = 0
    recompensas = []

    for _ in range(episodios_prueba):
        estado, _ = entorno.reset()
        recompensa_total = 0.0
        terminado = False

        while not terminado:
            accion = int(politica[estado])  # Seguir la política sin exploración
            estado, recompensa, terminado, truncado, _ = entorno.step(accion)
            terminado = terminado or truncado
            recompensa_total += recompensa

        recompensas.append(recompensa_total)
        if recompensa_total > 0:
            exitos += 1

    tasa_exito = exitos / episodios_prueba
    recompensa_media = np.mean(recompensas)
    return tasa_exito, recompensa_media


def imprimir_politica_frozen_lake(politica, titulo="Política"):
    """
    Imprime la política en forma de grilla 4×4 para FrozenLake.
    S=Start, G=Goal, H=Hole, F=Frozen
    """
    simbolos = SIMBOLOS_ACCIONES["frozen_lake"]
    print(f"\n{'─'*30}")
    print(f"  {titulo}")
    print(f"{'─'*30}")
    for fila in range(4):
        linea = "  "
        for col in range(4):
            estado = fila * 4 + col
            celda = MAPA_FROZEN_LAKE[fila][col]
            if celda == "H":
                linea += " ✗ "  # Agujero
            elif celda == "G":
                linea += " ★ "  # Meta
            elif celda == "S":
                linea += f" {simbolos[politica[estado]]}ˢ"  # Inicio
            else:
                linea += f" {simbolos[politica[estado]]} "
        print(linea)
    print(f"{'─'*30}")
    print("  Leyenda: ✗=Agujero  ★=Meta  ˢ=Inicio\n")


def imprimir_tabla_valores(V, titulo="Tabla de Valores V(s)"):
    """
    Imprime la función de valor como grilla 4×4 para FrozenLake.
    """
    print(f"\n{'─'*40}")
    print(f"  {titulo}")
    print(f"{'─'*40}")
    for fila in range(4):
        linea = "  "
        for col in range(4):
            estado = fila * 4 + col
            linea += f"{V[estado]:6.3f}  "
        print(linea)
    print(f"{'─'*40}\n")


def suavizar_curva(recompensas, ventana=200):
    """Aplica media móvil para suavizar la curva de recompensas."""
    return np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")


# =============================================================================
#   GRÁFICAS
# =============================================================================


def graficar_valor_iteration(V_det, V_esto, nombre_archivo):
    """Visualiza las tablas de valor de Value Iteration para ambos modos."""
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Value Iteration — Función de Valor V*(s) — FrozenLake 4×4",
        fontsize=14,
        fontweight="bold",
    )

    for i, (V, modo) in enumerate([(V_det, "Determinístico"), (V_esto, "Estocástico")]):
        grilla = V.reshape(4, 4)
        im = ejes[i].imshow(
            grilla,
            cmap="YlOrRd",
            interpolation="nearest",
            vmin=0,
            vmax=max(V.max(), 0.01),
        )
        ejes[i].set_title(f"Modo {modo}", fontsize=12)

        # Anotar cada celda con su valor
        for fila in range(4):
            for col in range(4):
                celda = MAPA_FROZEN_LAKE[fila][col]
                texto = celda if celda in ("H", "G") else f"{grilla[fila, col]:.3f}"
                color = "black" if grilla[fila, col] < 0.5 else "white"
                ejes[i].text(
                    col,
                    fila,
                    texto,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                )

        ejes[i].set_xticks(range(4))
        ejes[i].set_yticks(range(4))
        ejes[i].set_xticklabels(range(4))
        ejes[i].set_yticklabels(range(4))
        ejes[i].set_xlabel("Columna")
        ejes[i].set_ylabel("Fila")
        plt.colorbar(im, ax=ejes[i], label="V*(s)")

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_GRAFICAS, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica guardada: {ruta}")


def graficar_politica_frozen_lake(
    politica_det, politica_esto, V_det, V_esto, nombre_archivo
):
    """Visualiza las políticas óptimas de ambos modos como flechas en grilla."""
    simbolos = SIMBOLOS_ACCIONES["frozen_lake"]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Política Óptima — FrozenLake 4×4", fontsize=14, fontweight="bold")

    for i, (politica, V, modo) in enumerate(
        [
            (politica_det, V_det, "Determinístico"),
            (politica_esto, V_esto, "Estocástico"),
        ]
    ):
        grilla_v = V.reshape(4, 4)
        ejes[i].imshow(
            grilla_v,
            cmap="Blues",
            interpolation="nearest",
            vmin=0,
            vmax=max(V.max(), 0.01),
        )
        ejes[i].set_title(f"Modo {modo}", fontsize=12)

        for fila in range(4):
            for col in range(4):
                estado = fila * 4 + col
                celda = MAPA_FROZEN_LAKE[fila][col]
                if celda == "H":
                    ejes[i].text(
                        col,
                        fila,
                        "✗",
                        ha="center",
                        va="center",
                        fontsize=20,
                        color="red",
                        fontweight="bold",
                    )
                elif celda == "G":
                    ejes[i].text(
                        col,
                        fila,
                        "★",
                        ha="center",
                        va="center",
                        fontsize=20,
                        color="gold",
                        fontweight="bold",
                    )
                else:
                    flecha = simbolos[politica[estado]]
                    ejes[i].text(
                        col,
                        fila,
                        flecha,
                        ha="center",
                        va="center",
                        fontsize=22,
                        color="black",
                        fontweight="bold",
                    )

        ejes[i].set_xticks(range(4))
        ejes[i].set_yticks(range(4))
        ejes[i].grid(True, color="white", linewidth=2)

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_GRAFICAS, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica guardada: {ruta}")


def graficar_recompensas_qlearning(
    recomp_det, recomp_esto, eps_det, eps_esto, nombre_archivo
):
    """
    Gráfica de recompensas y decaimiento de ε durante el entrenamiento Q-Learning.
    Muestra la curva suavizada para mayor claridad.
    """
    ventana = 300  # Ventana de suavizado

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])  # Recompensas determinístico
    ax2 = fig.add_subplot(gs[0, 1])  # Recompensas estocástico
    ax3 = fig.add_subplot(gs[1, 0])  # Épsilon determinístico
    ax4 = fig.add_subplot(gs[1, 1])  # Épsilon estocástico

    fig.suptitle(
        "Q-Learning — Entrenamiento en FrozenLake", fontsize=14, fontweight="bold"
    )

    # ── Recompensas ────────────────────────────────────────────────────────
    for ax, recomp, modo, color in [
        (ax1, recomp_det, "Determinístico", "steelblue"),
        (ax2, recomp_esto, "Estocástico", "darkorange"),
    ]:
        ax.plot(recomp, color=color, alpha=0.15, linewidth=0.5, label="Por episodio")
        if len(recomp) >= ventana:
            curva_suave = suavizar_curva(recomp, ventana)
            x_suave = range(ventana - 1, len(recomp))
            ax.plot(
                x_suave,
                curva_suave,
                color=color,
                linewidth=2,
                label=f"Media móvil ({ventana})",
            )
        ax.set_title(f"Recompensas — Modo {modo}", fontsize=11)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Recompensa total")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

    # ── Decaimiento de ε ───────────────────────────────────────────────────
    for ax, epsilons, modo, color in [
        (ax3, eps_det, "Determinístico", "steelblue"),
        (ax4, eps_esto, "Estocástico", "darkorange"),
    ]:
        ax.plot(epsilons, color=color, linewidth=1.5)
        ax.fill_between(range(len(epsilons)), epsilons, alpha=0.2, color=color)
        ax.set_title(f"Decaimiento de ε — Modo {modo}", fontsize=11)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Valor de ε (exploración)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.axhline(
            y=0.5,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="ε = 0.5 (50% exploración)",
        )
        ax.legend(fontsize=9)

    ruta = os.path.join(DIRECTORIO_GRAFICAS, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica guardada: {ruta}")


def graficar_comparacion(resultados, nombre_archivo):
    """
    Gráfica comparativa de tasa de éxito entre Value Iteration y Q-Learning
    en modos determinístico y estocástico.
    """
    etiquetas = list(resultados.keys())
    tasas = [resultados[k]["tasa_exito"] * 100 for k in etiquetas]
    colores = ["steelblue", "steelblue", "darkorange", "darkorange"]
    patrones = ["/", "\\", "/", "\\"]

    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.bar(etiquetas, tasas, color=colores, edgecolor="black", linewidth=0.8)

    # Agregar patrón alternado (método basado en modelo vs libre)
    for barra, patron in zip(barras, patrones):
        barra.set_hatch(patron)

    # Anotar el porcentaje encima de cada barra
    for barra, tasa in zip(barras, tasas):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 1,
            f"{tasa:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(
        "Comparación de Algoritmos — Tasa de Éxito en FrozenLake",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Tasa de Éxito (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)

    # Leyenda de patrones
    parche_vi = mpatches.Patch(
        facecolor="gray", hatch="/", label="Value Iteration (Basado en Modelo)"
    )
    parche_ql = mpatches.Patch(
        facecolor="gray", hatch="\\", label="Q-Learning (Libre de Modelo)"
    )
    ax.legend(handles=[parche_vi, parche_ql], fontsize=10, loc="upper right")

    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_GRAFICAS, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica guardada: {ruta}")


def graficar_tabla_q(tabla_Q, modo, nombre_archivo):
    """
    Visualiza la tabla Q como mapa de calor (heatmap).
    Eje X = acciones, Eje Y = estados.
    """
    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.imshow(tabla_Q, aspect="auto", cmap="hot_r", interpolation="nearest")
    ax.set_title(f"Tabla Q Final — FrozenLake {modo}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Acciones (0=←  1=↓  2=→  3=↑)", fontsize=11)
    ax.set_ylabel("Estado", fontsize=11)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["← (0)", "↓ (1)", "→ (2)", "↑ (3)"])

    # Resaltar estados especiales
    for estado in range(16):
        fila = estado // 4
        col = estado % 4
        celda = MAPA_FROZEN_LAKE[fila][col]
        if celda in ("H", "G"):
            ax.axhline(y=estado - 0.5, color="cyan", linewidth=0.5, alpha=0.5)
            ax.axhline(y=estado + 0.5, color="cyan", linewidth=0.5, alpha=0.5)
            ax.text(
                -0.8,
                estado,
                celda,
                va="center",
                fontsize=9,
                color="blue",
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Valor Q(s,a)")
    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_GRAFICAS, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica guardada: {ruta}")


def graficar_recompensas_taxi(recompensas, nombre_archivo):
    """Gráfica de entrenamiento Q-Learning para Taxi-v4."""
    ventana = 500
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Q-Learning — Entrenamiento en Taxi-v4", fontsize=14, fontweight="bold"
    )

    # Recompensas brutas + suavizadas
    ejes[0].plot(recompensas, color="purple", alpha=0.1, linewidth=0.3)
    if len(recompensas) >= ventana:
        curva_suave = suavizar_curva(recompensas, ventana)
        x_suave = range(ventana - 1, len(recompensas))
        ejes[0].plot(
            x_suave,
            curva_suave,
            color="purple",
            linewidth=2,
            label=f"Media móvil ({ventana})",
        )
    ejes[0].set_title("Recompensa por Episodio", fontsize=11)
    ejes[0].set_xlabel("Episodio")
    ejes[0].set_ylabel("Recompensa total")
    ejes[0].grid(True, alpha=0.3)
    ejes[0].legend()
    ejes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Histograma de recompensas (últimos 2000 episodios)
    ultimas = recompensas[-2000:]
    ejes[1].hist(ultimas, bins=40, color="purple", alpha=0.7, edgecolor="black")
    ejes[1].set_title(
        "Distribución de Recompensas (últimos 2000 episodios)", fontsize=11
    )
    ejes[1].set_xlabel("Recompensa total por episodio")
    ejes[1].set_ylabel("Frecuencia")
    ejes[1].grid(True, alpha=0.3)
    ejes[1].axvline(
        x=np.mean(ultimas),
        color="red",
        linestyle="--",
        label=f"Media: {np.mean(ultimas):.1f}",
    )
    ejes[1].legend()

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_GRAFICAS, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica guardada: {ruta}")


# =============================================================================
#   PUNTO 1: FROZEN LAKE — EJECUCIÓN PRINCIPAL
# =============================================================================


def ejecutar_frozen_lake():
    """
    Ejecuta el análisis completo del Punto 1:
    - Value Iteration (determinístico y estocástico)
    - Q-Learning       (determinístico y estocástico)
    - Comparación entre métodos
    """

    print("\n" + "═" * 65)
    print("  PUNTO 1: FROZEN LAKE")
    print("  Entorno: 4×4, agente = S(0,0), meta = G(3,3)")
    print("  Acciones: 0=← 1=↓ 2=→ 3=↑")
    print("═" * 65)

    resultados_comparacion = {}  # Para la gráfica final

    # ─────────────────────────────────────────────────────────────────────
    #  PARTE A: VALUE ITERATION
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  PARTE A: VALUE ITERATION (Basado en Modelo)")
    print("─" * 50)

    # ── 1. Modo Determinístico ────────────────────────────────────────────
    print("\n[1] Modo DETERMINÍSTICO (is_slippery=False)")
    print("    El agente va exactamente donde intenta ir (P=1.0)")
    entorno_det = gym.make("FrozenLake-v1", is_slippery=False)
    t_inicio = time.time()
    V_det, politica_vi_det, iters_det = value_iteration(entorno_det, gamma=0.99)
    tiempo_vi_det = time.time() - t_inicio

    print(f"    ✓ Convergió en {iters_det} iteraciones ({tiempo_vi_det:.4f}s)")
    imprimir_tabla_valores(V_det, "Tabla de Valores V*(s) — Determinístico")
    imprimir_politica_frozen_lake(
        politica_vi_det, "Política Óptima VI — Determinístico"
    )

    # Evaluar la política obtenida
    tasa_vi_det, recomp_vi_det = evaluar_politica(entorno_det, politica_vi_det)
    print(f"    Evaluación (100 episodios): Tasa de éxito = {tasa_vi_det*100:.1f}%")
    resultados_comparacion["VI\nDeterminístico"] = {
        "tasa_exito": tasa_vi_det,
        "recompensa_media": recomp_vi_det,
    }

    # ── 2. Modo Estocástico ───────────────────────────────────────────────
    print("\n[2] Modo ESTOCÁSTICO (is_slippery=True)")
    print("    El agente puede resbalar: 33% de ir a dirección deseada,")
    print("    33% cada dirección perpendicular → incertidumbre real")
    entorno_esto = gym.make("FrozenLake-v1", is_slippery=True)
    t_inicio = time.time()
    V_esto, politica_vi_esto, iters_esto = value_iteration(entorno_esto, gamma=0.99)
    tiempo_vi_esto = time.time() - t_inicio

    print(f"    ✓ Convergió en {iters_esto} iteraciones ({tiempo_vi_esto:.4f}s)")
    imprimir_tabla_valores(V_esto, "Tabla de Valores V*(s) — Estocástico")
    imprimir_politica_frozen_lake(politica_vi_esto, "Política Óptima VI — Estocástico")

    # Evaluar la política obtenida
    tasa_vi_esto, recomp_vi_esto = evaluar_politica(entorno_esto, politica_vi_esto)
    print(f"    Evaluación (100 episodios): Tasa de éxito = {tasa_vi_esto*100:.1f}%")
    resultados_comparacion["VI\nEstocástico"] = {
        "tasa_exito": tasa_vi_esto,
        "recompensa_media": recomp_vi_esto,
    }

    # Generar gráficas de Value Iteration
    graficar_valor_iteration(V_det, V_esto, "01_value_iteration_funcion_valor.png")
    graficar_politica_frozen_lake(
        politica_vi_det,
        politica_vi_esto,
        V_det,
        V_esto,
        "02_value_iteration_politica.png",
    )

    # ─────────────────────────────────────────────────────────────────────
    #  PARTE B: Q-LEARNING
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  PARTE B: Q-LEARNING (Libre de Modelo)")
    print("─" * 50)

    EPISODIOS_FL = 15000  # Episodios de entrenamiento para FrozenLake

    # ── 3. Q-Learning Determinístico ──────────────────────────────────────
    print(f"\n[3] Q-Learning — Modo DETERMINÍSTICO ({EPISODIOS_FL} episodios)")
    print("    ε=1.0 → decae 0.999 por episodio → ε_min=0.01")
    entorno_det2 = gym.make("FrozenLake-v1", is_slippery=False)
    t_inicio = time.time()
    tabla_Q_det, politica_ql_det, recomp_det, eps_det = q_learning(
        entorno_det2,
        episodios=EPISODIOS_FL,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        decaimiento_epsilon=0.9995,
        epsilon_min=0.01,
    )
    tiempo_ql_det = time.time() - t_inicio
    print(f"    ✓ Entrenamiento completo en {tiempo_ql_det:.2f}s")
    imprimir_politica_frozen_lake(politica_ql_det, "Política QL — Determinístico")

    tasa_ql_det, recomp_ql_det_media = evaluar_politica(entorno_det2, politica_ql_det)
    print(f"    Evaluación (100 episodios): Tasa de éxito = {tasa_ql_det*100:.1f}%")
    resultados_comparacion["Q-Learning\nDeterminístico"] = {
        "tasa_exito": tasa_ql_det,
        "recompensa_media": recomp_ql_det_media,
    }

    # ── 4. Q-Learning Estocástico ─────────────────────────────────────────
    print(f"\n[4] Q-Learning — Modo ESTOCÁSTICO ({EPISODIOS_FL} episodios)")
    print("    Mayor decaimiento lento de ε (más exploración para compensar ruido)")
    entorno_esto2 = gym.make("FrozenLake-v1", is_slippery=True)
    t_inicio = time.time()
    tabla_Q_esto, politica_ql_esto, recomp_esto, eps_esto = q_learning(
        entorno_esto2,
        episodios=EPISODIOS_FL,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        decaimiento_epsilon=0.9997,  # Decaimiento más lento: más exploración
        epsilon_min=0.05,  # ε mínimo mayor: mantener algo de exploración
    )
    tiempo_ql_esto = time.time() - t_inicio
    print(f"    ✓ Entrenamiento completo en {tiempo_ql_esto:.2f}s")
    imprimir_politica_frozen_lake(politica_ql_esto, "Política QL — Estocástico")

    tasa_ql_esto, recomp_ql_esto_media = evaluar_politica(
        entorno_esto2, politica_ql_esto
    )
    print(f"    Evaluación (100 episodios): Tasa de éxito = {tasa_ql_esto*100:.1f}%")
    resultados_comparacion["Q-Learning\nEstocástico"] = {
        "tasa_exito": tasa_ql_esto,
        "recompensa_media": recomp_ql_esto_media,
    }

    # Generar gráficas Q-Learning
    graficar_recompensas_qlearning(
        recomp_det, recomp_esto, eps_det, eps_esto, "03_qlearning_recompensas.png"
    )
    graficar_tabla_q(tabla_Q_det, "Determinístico", "04_tabla_Q_deterministica.png")
    graficar_tabla_q(tabla_Q_esto, "Estocástico", "05_tabla_Q_estocastica.png")
    graficar_comparacion(resultados_comparacion, "06_comparacion_algoritmos.png")

    # ─────────────────────────────────────────────────────────────────────
    #  TABLA RESUMEN
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  TABLA DE RESULTADOS — FROZEN LAKE")
    print("═" * 65)
    print(f"  {'Método':<28} {'Modo':<18} {'Tasa Éxito':>12} {'Tiempo':>10}")
    print("  " + "─" * 62)
    print(
        f"  {'Value Iteration':<28} {'Determinístico':<18} {tasa_vi_det*100:>10.1f}%  {tiempo_vi_det:>7.4f}s"
    )
    print(
        f"  {'Value Iteration':<28} {'Estocástico':<18} {tasa_vi_esto*100:>10.1f}%  {tiempo_vi_esto:>7.4f}s"
    )
    print(
        f"  {'Q-Learning':<28} {'Determinístico':<18} {tasa_ql_det*100:>10.1f}%  {tiempo_ql_det:>7.2f}s"
    )
    print(
        f"  {'Q-Learning':<28} {'Estocástico':<18} {tasa_ql_esto*100:>10.1f}%  {tiempo_ql_esto:>7.2f}s"
    )
    print("═" * 65)

    # ─────────────────────────────────────────────────────────────────────
    #  ANÁLISIS Y CONCLUSIONES
    # ─────────────────────────────────────────────────────────────────────
    print("""
  ANÁLISIS:
  ─────────────────────────────────────────────────────────────────
  [Value Iteration]
  • Converge a la política ÓPTIMA garantizada en pocas iteraciones.
  • Solo funciona si se conoce env.P (la función de transición).
  • En modo estocástico, pondera las probabilidades de deslizamiento,
    por eso la política parece "cautelosa" (rodea los agujeros).

  [Q-Learning]
  • Aprende sin conocer el modelo del entorno: solo interacciones.
  • Requiere muchos más episodios para explorar el espacio de estados.
  • En modo estocástico necesita más exploración y ε_min mayor.

  [¿Por qué el modo estocástico es más difícil?]
  • El agente no va donde intenta: acción real ≠ acción ejecutada.
  • Esto aumenta la varianza de las recompensas (ruido en el aprendizaje).
  • Value Iteration maneja esto explícitamente con las probabilidades.
  • Q-Learning necesita ver muchos más ejemplos para promediarlo.
  ─────────────────────────────────────────────────────────────────
""")

    return (
        tabla_Q_det,
        tabla_Q_esto,
        politica_vi_det,
        politica_vi_esto,
        politica_ql_det,
        politica_ql_esto,
    )


# =============================================================================
#   PUNTO 2: TAXI-v3 — ENTORNO ADICIONAL
# =============================================================================


def ejecutar_taxi():
    """
    Punto 2: Resolución del entorno Taxi-v4 con Q-Learning.

    Descripción del problema:
    ─────────────────────────
    El taxi debe recoger a un pasajero en una de 4 posiciones (R,G,Y,B)
    y llevarlo a su destino deseado en otra de las 4 posiciones.

    - Estados:  500  (25 posiciones del taxi × 5 posiciones pasajero × 4 destinos)
    - Acciones: 6    (0=Sur, 1=Norte, 2=Este, 3=Oeste, 4=Recoger, 5=Dejar)
    - Recompensa:
        • -1 por cada paso (penalidad por estar vivo)
        • +20 al dejar al pasajero en el destino correcto
        • -10 por ejecutar Recoger o Dejar incorrectamente
    """

    print("\n" + "═" * 65)
    print("  PUNTO 2: TAXI-v3 — ENTORNO ADICIONAL")
    print("  500 estados × 6 acciones — Espacio discreto")
    print("═" * 65)
    print("""
  DESCRIPCIÓN:
  • Mapa 5×5. El taxi parte de posición aleatoria.
  • Pasajero: puede estar en R(0,0), G(0,4), Y(4,0), B(4,3) o en el taxi.
  • Objetivo: recoger al pasajero y dejarlo en su destino.
  • MDP con 500 estados y dinámica accesible via env.P.
""")

    entorno_taxi = gym.make("Taxi-v4")

    # ─────────────────────────────────────────────────────────────────────
    #  PARTE A: VALUE ITERATION en Taxi
    # ─────────────────────────────────────────────────────────────────────
    print("─" * 50)
    print("  [A] VALUE ITERATION en Taxi-v4")
    print("─" * 50)

    t_inicio = time.time()
    V_taxi, politica_vi_taxi, iters_taxi = value_iteration(entorno_taxi, gamma=0.99)
    tiempo_vi_taxi = time.time() - t_inicio

    print(
        f"  ✓ Value Iteration convergió en {iters_taxi} iteraciones ({tiempo_vi_taxi:.4f}s)"
    )

    tasa_vi_taxi, recomp_vi_taxi = evaluar_politica(entorno_taxi, politica_vi_taxi)
    print(f"  Evaluación (100 episodios): Tasa de éxito = {tasa_vi_taxi*100:.1f}%")
    print(f"  Recompensa media = {recomp_vi_taxi:.2f}")

    # Mostrar algunos valores de estado representativos
    print("\n  Muestra de Valores V*(s) para Taxi (primeros 10 estados):")
    print("  " + "─" * 45)
    for s in range(10):
        print(f"  Estado {s:3d}: V = {V_taxi[s]:8.3f}")
    print("  ...")

    # ─────────────────────────────────────────────────────────────────────
    #  PARTE B: Q-LEARNING en Taxi
    # ─────────────────────────────────────────────────────────────────────
    print("\n─" * 50)
    print("  [B] Q-LEARNING en Taxi-v4")
    print("─" * 50)

    EPISODIOS_TAXI = 20000
    print(f"  Entrenando con {EPISODIOS_TAXI} episodios...")
    print("  Hiperparámetros: α=0.1, γ=0.99, ε_inicial=1.0, decaimiento=0.9995")

    t_inicio = time.time()
    tabla_Q_taxi, politica_ql_taxi, recomp_taxi, _ = q_learning(
        entorno_taxi,
        episodios=EPISODIOS_TAXI,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        decaimiento_epsilon=0.9995,
        epsilon_min=0.01,
    )
    tiempo_ql_taxi = time.time() - t_inicio

    print(f"  ✓ Entrenamiento completo en {tiempo_ql_taxi:.2f}s")

    tasa_ql_taxi, recomp_ql_taxi = evaluar_politica(entorno_taxi, politica_ql_taxi)
    print(f"  Evaluación (100 episodios): Tasa de éxito = {tasa_ql_taxi*100:.1f}%")
    print(f"  Recompensa media = {recomp_ql_taxi:.2f}")

    # Generar gráfica de entrenamiento
    graficar_recompensas_taxi(recomp_taxi, "07_taxi_qlearning_entrenamiento.png")

    # ─────────────────────────────────────────────────────────────────────
    #  COMPARACIÓN Value Iteration vs Q-Learning en Taxi
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  TABLA DE RESULTADOS — TAXI-v3")
    print("═" * 65)
    print(f"  {'Método':<25} {'Tasa Éxito':>12} {'Recomp. Media':>15} {'Tiempo':>10}")
    print("  " + "─" * 62)
    print(
        f"  {'Value Iteration':<25} {tasa_vi_taxi*100:>10.1f}%  {recomp_vi_taxi:>13.2f}  {tiempo_vi_taxi:>7.4f}s"
    )
    print(
        f"  {'Q-Learning':<25} {tasa_ql_taxi*100:>10.1f}%  {recomp_ql_taxi:>13.2f}  {tiempo_ql_taxi:>7.2f}s"
    )
    print("═" * 65)

    # ─────────────────────────────────────────────────────────────────────
    #  GRÁFICA COMPARATIVA TAXI
    # ─────────────────────────────────────────────────────────────────────
    resultados_taxi = {
        "VI\nTaxi-v4": {"tasa_exito": tasa_vi_taxi},
        "Q-Learning\nTaxi-v4": {"tasa_exito": tasa_ql_taxi},
    }
    graficar_comparacion(resultados_taxi, "08_taxi_comparacion.png")

    print("""
  ANÁLISIS TAXI-v3:
  ─────────────────────────────────────────────────────────────────
  • Taxi tiene 500 estados: Value Iteration los evalúa TODOS en cada
    barrida. Con Q-Learning, el agente los descubre gradualmente.
  • La recompensa de -1 por paso incentiva trayectorias cortas.
  • La penalización de -10 por acciones incorrectas (Recoger/Dejar
    en el lugar equivocado) es clave: Q-Learning debe aprender a
    NO usar esas acciones salvo en los momentos correctos.
  • Value Iteration garantiza optimalidad; Q-Learning aproxima.
  ─────────────────────────────────────────────────────────────────
""")

    return tabla_Q_taxi, politica_vi_taxi, politica_ql_taxi


# =============================================================================
#   FUNCIÓN PRINCIPAL
# =============================================================================


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     LABORATORIO N° 3 — APRENDIZAJE POR REFUERZO                 ║
║     Value Iteration  +  Q-Learning                              ║
║     FrozenLake-v1  |  Taxi-v4                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

    np.random.seed(42)  # Semilla para reproducibilidad

    # ── Punto 1: FrozenLake ────────────────────────────────────────────────
    ejecutar_frozen_lake()

    # ── Punto 2: Taxi-v4 ──────────────────────────────────────────────────
    ejecutar_taxi()

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  GRÁFICAS GENERADAS EN: {DIRECTORIO_GRAFICAS:<39}║
║                                                                  ║
║  01 - Value Iteration: Función de Valor                         ║
║  02 - Value Iteration: Política Óptima (flechas en grilla)      ║
║  03 - Q-Learning: Curvas de Recompensa + Decaimiento ε          ║
║  04 - Tabla Q Final (Determinístico)                            ║
║  05 - Tabla Q Final (Estocástico)                               ║
║  06 - Comparación Final: VI vs QL en FrozenLake                 ║
║  07 - Taxi: Curva de Entrenamiento Q-Learning                   ║
║  08 - Taxi: Comparación VI vs QL                                ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
