import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Configurar salida estándar y de error para UTF-8 para evitar errores de codificación en Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================================
# PARTE A: MÉTODO BASADO EN MODELO (VALUE ITERATION / ITERACIÓN DE VALOR)
# ============================================================================
def iteracion_de_valor(entorno, gamma=0.99, theta=1e-8):
    """
    Implementa el algoritmo Value Iteration resolviendo la Ecuación de Bellman.
    Asume acceso completo a las dinámicas del entorno (entorno.P).
    """
    # 1. Inicializar la función de valor (V) en ceros para todos los estados
    num_estados = entorno.observation_space.n
    num_acciones = entorno.action_space.n
    V = np.zeros(num_estados)

    # Extraemos el diccionario de transiciones.
    # Usamos .unwrapped para asegurar el acceso a la estructura base del entorno.
    P = entorno.unwrapped.P

    # Iteración de la función de valor hasta convergencia
    while True:
        delta = 0
        # 2. Recorrer todos los estados
        for s in range(num_estados):
            valor_anterior = V[s]
            valores_q = np.zeros(num_acciones)

            # 3. Evaluar todas las acciones posibles
            for a in range(num_acciones):
                # 4. Utilizar la dinámica del entorno P
                # prob: probabilidad de transición
                # siguiente_estado: estado resultante
                # recompensa: recompensa obtenida
                # terminado: indica si es un estado terminal (agujero o meta)
                for prob, siguiente_estado, recompensa, terminado in P[s][a]:
                    # Ecuación de Bellman para el valor esperado
                    valores_q[a] += prob * (
                        recompensa + gamma * V[siguiente_estado] * (not terminado)
                    )

            # 5. Actualizar los valores: el valor del estado es el máximo valor Q posible
            V[s] = np.max(valores_q)
            delta = max(delta, abs(valor_anterior - V[s]))

        # Criterio de convergencia: si el cambio máximo es menor a theta, nos detenemos
        if delta < theta:
            break

    # 6. Extraer la política óptima a partir de la función de valor V convergida
    politica = np.zeros(num_estados, dtype=int)
    for s in range(num_estados):
        valores_q = np.zeros(num_acciones)
        for a in range(num_acciones):
            for prob, siguiente_estado, recompensa, terminado in P[s][a]:
                valores_q[a] += prob * (
                    recompensa + gamma * V[siguiente_estado] * (not terminado)
                )
        # La acción óptima es la que maximiza el valor Q
        politica[s] = np.argmax(valores_q)

    return V, politica


# ============================================================================
# PARTE B: MÉTODO LIBRE DE MODELO (Q-LEARNING / APRENDIZAJE Q)
# ============================================================================
def aprendizaje_q(
    entorno,
    episodios=10000,
    alfa=0.1,
    gamma=0.99,
    epsilon=1.0,
    decaimiento_epsilon=0.999,
):
    """
    Implementa el algoritmo Q-Learning (Off-policy TD Control).
    El agente aprende a partir de la interacción, sin conocer entorno.P.
    """
    num_estados = entorno.observation_space.n
    num_acciones = entorno.action_space.n

    # 1. Inicializar la tabla Q arbitrariamente (ceros)
    Q = np.zeros((num_estados, num_acciones))
    recompensas_por_episodio = []

    for episodio in range(episodios):
        # Resetear el entorno para un nuevo episodio
        info_estado = entorno.reset()
        # Manejo de compatibilidad entre versiones de Gym/Gymnasium
        estado = info_estado[0] if isinstance(info_estado, tuple) else info_estado

        recompensa_total = 0
        terminado = False

        while not terminado:
            # 2. Aplicar una estrategia epsilon-greedy para Exploración vs Explotación
            if np.random.uniform(0, 1) < epsilon:
                accion = entorno.action_space.sample()  # Exploración: acción aleatoria
            else:
                accion = np.argmax(Q[estado, :])  # Explotación: mejor acción conocida

            # Tomar la acción en el entorno
            resultado_paso = entorno.step(accion)
            # Manejo de compatibilidad (gymnasium devuelve 5 valores, gym antiguo 4)
            if len(resultado_paso) == 5:
                siguiente_estado, recompensa, finalizado, truncado, _ = resultado_paso
                terminado = finalizado or truncado
            else:
                siguiente_estado, recompensa, terminado, _ = resultado_paso

            # 3. Actualizar Q mediante la ecuación de aprendizaje (Diferencia Temporal)
            # Calculamos el objetivo DT (Diferencia Temporal)
            mejor_siguiente_accion = np.argmax(Q[siguiente_estado, :])
            objetivo_dt = recompensa + gamma * Q[
                siguiente_estado, mejor_siguiente_accion
            ] * (not terminado)
            error_dt = objetivo_dt - Q[estado, accion]

            # Actualización de la Tabla Q
            Q[estado, accion] += alfa * error_dt

            # Avanzar al siguiente estado
            estado = siguiente_estado
            recompensa_total += recompensa

        # Decaimiento del parámetro epsilon para explorar menos a medida que el agente aprende
        epsilon = max(0.01, epsilon * decaimiento_epsilon)
        recompensas_por_episodio.append(recompensa_total)

    # 6. Obtener la política final (estrategia determinística derivada de Q)
    politica = np.argmax(Q, axis=1)

    return Q, politica, recompensas_por_episodio


# ============================================================================
# FUNCIONES AUXILIARES (EVALUACIÓN Y VISUALIZACIÓN)
# ============================================================================
def evaluar_politica(entorno, politica, episodios=100):
    """
    Ejecuta episodios de prueba (sin aprendizaje) para evaluar el éxito de una política.
    Retorna la tasa de éxito (porcentaje de veces que llega a la meta).
    """
    exitos = 0
    for _ in range(episodios):
        info_estado = entorno.reset()
        estado = info_estado[0] if isinstance(info_estado, tuple) else info_estado
        terminado = False
        while not terminado:
            accion = politica[estado]
            resultado_paso = entorno.step(accion)
            if len(resultado_paso) == 5:
                estado, recompensa, finalizado, truncado, _ = resultado_paso
                terminado = finalizado or truncado
            else:
                estado, recompensa, terminado, _ = resultado_paso
            if (
                recompensa == 1.0
            ):  # En FrozenLake, recompensa de 1 significa llegar a la meta
                exitos += 1
    return exitos / episodios * 100


def imprimir_politica(politica):
    """
    Traduce el arreglo numérico de la política en una matriz 4x4 de flechas
    visuales para el entorno FrozenLake.
    0: ← (Izquierda), 1: ↓ (Abajo), 2: → (Derecha), 3: ↑ (Arriba)
    """
    flechas = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    cuadricula = [flechas[a] for a in politica]
    for i in range(0, 16, 4):
        print(
            f"[{cuadricula[i]}] [{cuadricula[i+1]}] [{cuadricula[i+2]}] [{cuadricula[i+3]}]"
        )


def graficar_recompensas(recompensas_det, recompensas_estoc, ventana=100):
    """
    Grafica la suma acumulada/media móvil de recompensas durante el entrenamiento
    para visualizar el aprendizaje en Q-Learning.
    """
    # Calculamos la media móvil para suavizar la curva
    suavizado_det = np.convolve(
        recompensas_det, np.ones(ventana) / ventana, mode="valid"
    )
    suavizado_estoc = np.convolve(
        recompensas_estoc, np.ones(ventana) / ventana, mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(suavizado_det, label="Determinístico", alpha=0.8)
    plt.plot(suavizado_estoc, label="Estocástico", alpha=0.8)
    plt.title("Q-Learning: Recompensas promedio durante el Entrenamiento")
    plt.xlabel(f"Episodios (Media móvil de {ventana})")
    plt.ylabel("Tasa de Recompensa (Éxito)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Guardar gráfica
    plt.savefig("q_learning_rewards.png")
    print("\n[INFO] Gráfica guardada como 'q_learning_rewards.png'")


def visualizar_juego(
    nombre_entorno, politica, es_deslizante=False, episodios=3, retraso=0.5
):
    """
    Permite ver visualmente el juego realizado por el agente usando una política determinada.
    Intenta crear una versión del entorno con render_mode="human" (gráfica en ventana Pygame),
    y si falla, utiliza render_mode="ansi" para mostrarlo en modo texto en la terminal.
    """
    print(
        f"\n[VISUALIZACIÓN] Iniciando visualización gráfica ({episodios} episodios)..."
    )
    entorno_visual = None
    try:
        # Intentamos iniciar con render interactivo (Pygame)
        entorno_visual = gym.make(
            nombre_entorno, is_slippery=es_deslizante, render_mode="human"
        )
    except Exception as e:
        print(
            f"No se pudo inicializar en modo 'human' (posiblemente falta entorno gráfico o Pygame): {e}"
        )
        print("Intentando inicializar en modo texto ('ansi')...")
        try:
            entorno_visual = gym.make(
                nombre_entorno, is_slippery=es_deslizante, render_mode="ansi"
            )
        except Exception as e2:
            print(f"Error crítico al inicializar el entorno: {e2}")
            return

    for ep in range(episodios):
        print(f"\n--- Ejecutando episodio visual {ep + 1}/{episodios} ---")
        info_estado = entorno_visual.reset()
        estado = info_estado[0] if isinstance(info_estado, tuple) else info_estado
        terminado = False
        recompensa = 0.0

        if entorno_visual.render_mode == "ansi":
            print(entorno_visual.render())

        time.sleep(retraso)
        pasos = 0

        while not terminado:
            accion = politica[estado]
            resultado_paso = entorno_visual.step(accion)

            if len(resultado_paso) == 5:
                estado, recompensa, finalizado, truncado, _ = resultado_paso
                terminado = finalizado or truncado
            else:
                estado, recompensa, terminado, _ = resultado_paso

            if entorno_visual.render_mode == "ansi":
                print(entorno_visual.render())

            pasos += 1
            time.sleep(retraso)

            if pasos > 100:
                print("Límite de pasos (100) alcanzado. Deteniendo episodio.")
                break

        if recompensa == 1.0:
            print("¡Éxito! El agente llegó a la meta (G).")
        else:
            print("El agente cayó en un agujero (H).")
        time.sleep(1.0)

    entorno_visual.close()


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    # --- 1. ENTORNOS ---
    print("Inicializando entornos de FrozenLake...")
    try:
        # Gym/Gymnasium moderno
        entorno_det = gym.make("FrozenLake-v1", is_slippery=False)
        entorno_estoc = gym.make("FrozenLake-v1", is_slippery=True)
    except gym.error.DeprecatedEnv:
        entorno_det = gym.make("FrozenLake-v0", is_slippery=False)
        entorno_estoc = gym.make("FrozenLake-v0", is_slippery=True)

    # --- 2. VALUE ITERATION ---
    print("\n" + "=" * 50)
    print("EJECUTANDO VALUE ITERATION (Basado en Modelo)")
    print("=" * 50)

    V_det, politica_iv_det = iteracion_de_valor(entorno_det)
    print("\nPolítica Óptima - Determinístico (Value Iteration):")
    imprimir_politica(politica_iv_det)
    tasa_iv_det = evaluar_politica(entorno_det, politica_iv_det)
    print(f"Tasa de éxito (Determinístico): {tasa_iv_det:.2f}%")

    V_estoc, politica_iv_estoc = iteracion_de_valor(entorno_estoc)
    print("\nPolítica Óptima - Estocástico (Value Iteration):")
    imprimir_politica(politica_iv_estoc)
    tasa_iv_estoc = evaluar_politica(entorno_estoc, politica_iv_estoc)
    print(f"Tasa de éxito (Estocástico): {tasa_iv_estoc:.2f}%")

    # --- 3. Q-LEARNING ---
    print("\n" + "=" * 50)
    print("EJECUTANDO Q-LEARNING (Libre de Modelo)")
    print("=" * 50)

    # Entrenamiento Determinístico
    print("\nEntrenando agente en entorno Determinístico...")
    Q_det, politica_aq_det, recompensas_aq_det = aprendizaje_q(
        entorno_det, episodios=10000
    )
    print("Política Óptima - Determinístico (Q-Learning):")
    imprimir_politica(politica_aq_det)
    tasa_aq_det = evaluar_politica(entorno_det, politica_aq_det)
    print(f"Tasa de éxito (Determinístico): {tasa_aq_det:.2f}%")

    # Entrenamiento Estocástico
    # Aumentamos episodios para el estocástico por ser más complejo
    print("\nEntrenando agente en entorno Estocástico...")
    Q_estoc, politica_aq_estoc, recompensas_aq_estoc = aprendizaje_q(
        entorno_estoc, episodios=15000, alfa=0.1, decaimiento_epsilon=0.9995
    )
    print("Política Óptima - Estocástico (Q-Learning):")
    imprimir_politica(politica_aq_estoc)
    tasa_aq_estoc = evaluar_politica(entorno_estoc, politica_aq_estoc)
    print(f"Tasa de éxito (Estocástico): {tasa_aq_estoc:.2f}%")

    # Imprimimos parte de la Tabla Q del estocástico como evidencia
    print("\nTabla Q Final (Estocástico) - Primeros 4 Estados:")
    print(Q_estoc[:4])

    # Generamos gráfica
    graficar_recompensas(recompensas_aq_det, recompensas_aq_estoc)

    # --- 4. VISUALIZACIÓN EN VIVO ---
    print("\n" + "=" * 50)
    print("VISUALIZACIÓN DE LAS POLÍTICAS ÓPTIMAS EN ACCIÓN")
    print("=" * 50)

    # 1. Agente entrenado con Q-Learning en entorno Determinístico
    print("\nVisualizando agente en entorno DETERMINÍSTICO:")
    visualizar_juego(
        "FrozenLake-v1", politica_aq_det, es_deslizante=False, episodios=2, retraso=0.4
    )

    # 2. Agente entrenado con Q-Learning en entorno Estocástico
    print("\nVisualizando agente en entorno ESTOCÁSTICO:")
    visualizar_juego(
        "FrozenLake-v1", politica_aq_estoc, es_deslizante=True, episodios=2, retraso=0.4
    )

    print("\nEjecución finalizada. Revisar el archivo de recompensas guardado.")
