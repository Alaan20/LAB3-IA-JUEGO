import numpy as np

# 1. Configuración del entorno determinístico
estados = [0, 1, 2, 3]
acciones = ["Izquierda", "Derecha"]
gamma = 0.9  # Factor de descuento
theta = 0.001  # Umbral de convergencia

# Inicializar V(s) en 0
V = np.zeros(len(estados))


# Función de transición determinística: (estado, acción) -> (siguiente_estado, recompensa)
def transicion_determinista(s, a):
    if s == 3:  # Estado terminal, no haces nada
        return s, 0

    if a == "Izquierda":
        s_siguiente = max(0, s - 1)
    else:  # 'Derecha'
        s_siguiente = min(3, s + 1)

    # Recompensa
    if s_siguiente == 3:
        recompensa = 10.0  # ¡Llegamos a la meta!
    else:
        recompensa = -1.0  # Penalización por paso

    return s_siguiente, recompensa


# 2. Algoritmo de Value Iteration
iteracion = 0
while True:
    delta = 0
    V_viejo = V.copy()

    for s in estados:
        if s == 3:  # El estado terminal se queda en 0 o su valor fijo
            continue

        valores_acciones = []
        for a in acciones:
            s_siguiente, recompensa = transicion_determinista(s, a)
            # Bellman determinístico: R + gamma * V(s')
            valor_q = recompensa + gamma * V_viejo[s_siguiente]
            valores_acciones.append(valor_q)

        # Guardamos el máximo valor encontrado entre las acciones
        V[s] = max(valores_acciones)

        # Calcular el cambio máximo para la condición de parada
        delta = max(delta, abs(V_viejo[s] - V[s]))

    iteracion += 1
    print(f"Iteración {iteracion}: V = {np.round(V, 2)}")

    if delta < theta:
        print("¡El algoritmo ha convergido!")
        break

# 3. Extracción de la Política Óptima (Pi*)
politica = {}
for s in estados:
    if s == 3:
        politica[s] = "META"
        continue

    mejores_acciones = []
    for a in acciones:
        s_siguiente, recompensa = transicion_determinista(s, a)
        mejores_acciones.append((recompensa + gamma * V[s_siguiente], a))

    # Elegimos la acción que da el valor máximo
    politica[s] = max(mejores_acciones, key=lambda x: x[0])[1]

print("\nPolítica Óptima resultante:")
print(politica)


import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
observation, info = env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    # 0: Move left 1: Move down 2: Move right 3: Move top
    action = (
        env.action_space.sample()
    )  # Random action for now - real agents will be smarter!
    # Take the action and see what happens
    estado_actual, reward, terminated, truncated, info = env.step(action)
    # reward: +1 Reach goal - 0 Reach frozen - 0 Reach hole
    # terminated: The player moves into a hole or The player reaches the goal
    # truncated: The length of the episode is 100 for FrozenLake4x4, 200 for FrozenLake8x8.
    print(estado_actual, reward, terminated, truncated, info)
    total_reward += reward

    episode_over = terminated or truncated

print(f"\nEpisode finished! Total reward: {total_reward}")
env.close()
