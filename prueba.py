import numpy as np

# Inicializar V(s) en 0
V = np.zeros(16)


# 2. Algoritmo de Value Iteration
def value_iteration(lista_estados_p, gamma_p, theta_p):
    iteracion = 0
    estados_terminales = {5, 7, 11, 12, 15}
    while True:
        delta = 0
        V_viejo = V.copy()

        for s in range(len(lista_estados_p)):
            if s in estados_terminales:
                continue

            valores_acciones = []
            for a in range(4):
                prob, siguiente, recompensa, terminado = lista_estados_p[s][a][0]

                valor_q = recompensa + gamma_p * V_viejo[siguiente]
                valores_acciones.append(valor_q)

            # Guardamos el máximo valor encontrado entre las acciones
            V[s] = max(valores_acciones)

            # Calcular el cambio máximo para la condición de parada
            delta = max(delta, abs(V_viejo[s] - V[s]))

        iteracion += 1
        ##print(f"Iteración {iteracion}: V = {np.round(V, 2)}")

        if delta < theta_p:
            print("¡El algoritmo ha convergido!")
            politica_estados = {}
            for s in range(16):
                valores_acciones = []
                lista_estados = []
                for a in range(4):
                    prob, siguiente, recompensa, terminado = lista_estados_p[s][a][0]
                    valor_p = recompensa + gamma_p * V[siguiente]
                    valores_acciones.append(valor_p)
                    lista_estados.append(siguiente)
                max_valor = max(valores_acciones)
                indice_mejor = valores_acciones.index(max_valor)
                politica_estados[s] = indice_mejor
            return V, politica_estados


import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
observation, info = env.reset()

episode_over = False
total_reward = 0
gamma = 0.99
theta = 1e-8
lista_estados = env.unwrapped.P
res, json_politica = value_iteration(lista_estados, gamma, theta)
print("Matrizzzz")
print(np.round(res.reshape(4, 4), 4))
print("\nPolítica Óptima")
for i in json_politica:
    print(i, "= ", json_politica.get(i))

it = 0
while not episode_over:
    # 0: Move left 1: Move down 2: Move right 3: Move top
    if it == 0:
        action = json_politica.get(observation)
        it += 1
    else:
        action = json_politica.get(estado_actual)
    estado_actual, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    episode_over = terminated or truncated

print(f"\nEpisode finished! Total reward: {total_reward}")

env.close()
