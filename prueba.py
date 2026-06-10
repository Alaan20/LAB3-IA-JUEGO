import gymnasium as gym
import numpy as np


# 2. Algoritmo de Value Iteration
def value_iteration(entorno, gamma_p, theta_p):
    iteracion = 0
    modelo = entorno.unwrapped.P
    num_estados = entorno.observation_space.n
    num_acciones = entorno.action_space.n
    V = np.zeros(num_estados)
    while True:
        delta = 0.0

        for s in range(num_estados):
            valor_anterior = V[s]
            valores_q = np.zeros(num_acciones)

            for a in range(num_acciones):
                for prob, siguiente, recompensa, terminado in modelo[s][a]:
                    valores_q[a] += prob * (recompensa + gamma_p * V[siguiente])

            # Guardamos el máximo valor encontrado entre las acciones
            V[s] = max(valores_q)

            # Calcular el cambio máximo para la condición de parada
            delta = max(delta, abs(valor_anterior - V[s]))

        iteracion += 1
        ##print(f"Iteración {iteracion}: V = {np.round(V, 2)}")

        if delta < theta_p:
            print("¡El algoritmo ha convergido!")
            break

    politica_estados = {}
    for s in range(num_estados):
        valores_q = np.zeros(num_acciones)
        for a in range(num_acciones):
            for prob, siguiente, recompensa, terminado in modelo[s][a]:
                valores_q[a] += prob * (recompensa + gamma_p * V[siguiente])

        politica_estados[s] = np.argmax(valores_q)
    return V, politica_estados


def impresion_matriz_v_y_politica(nombre, matriz, policy):
    print(f"Matriz {nombre} de Valores:")
    print(np.round(matriz.reshape(8, 8), 4))
    print("\nPolítica Óptima:")
    for i in policy:
        print(i, "= ", policy.get(i))


env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False, map_name="8x8")
observation, info = env.reset()

episode_over = False
total_reward = 0
gamma = 0.99
theta = 1e-8
lista_estados = env.unwrapped.P
res, json_politica = value_iteration(env, gamma, theta)
tipo_matriz = "Determinista"
impresion_matriz_v_y_politica(tipo_matriz, res, json_politica)


estado_actual = observation
while not episode_over:
    # 0: Move left 1: Move down 2: Move right 3: Move top
    action = json_politica.get(estado_actual)
    estado_actual, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    episode_over = terminated or truncated

print(f"\nEjecución finalizada! Recompensa total: {total_reward}")

env.close()


env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True, map_name="8x8")
observation, info = env.reset()

episode_over = False
total_reward = 0
gamma = 0.99
theta = 1e-8

matriz_costos_v, json_politica = value_iteration(env, gamma, theta)
tipo_matriz = "Estocástico"
impresion_matriz_v_y_politica(tipo_matriz, matriz_costos_v, json_politica)


estado_actual = observation
while not episode_over:
    # 0: Move left 1: Move down 2: Move right 3: Move top

    action = json_politica.get(estado_actual)
    estado_actual, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    episode_over = terminated or truncated

print(f"\nEpisode finished! Total reward: {total_reward}")
