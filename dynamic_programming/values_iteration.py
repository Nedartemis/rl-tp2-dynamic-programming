import numpy as np
from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    Vi = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    converged: bool = False

    n_iter = 0
    while not converged and n_iter < max_iter:

        Vip1 = np.zeros(mdp.observation_space.n)

        for s in range(0, mdp.observation_space.n):

            # Estimer la fonction de valeur de chaque état
            vs = np.zeros(mdp.action_space.n)
            for a in range(0, mdp.action_space.n):
                sp, reward, _ = mdp.P[s][a]
                Pa_s_sp = 1
                Ra_s_sp = reward
                Vi_sp = Vi[sp]
                v = Pa_s_sp * (Ra_s_sp + gamma * Vi_sp)
                vs[a] = v

            # Choisir l'action qui maximise la valeur
            m = vs.max()

            # Mettre à jour la valeur de l'état
            Vip1[s] = m

        converged = np.allclose(Vi, Vip1)
        Vi = Vip1
        n_iter += 1

    # END SOLUTION
    return Vi


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    Vi = np.zeros((env.height, env.width))
    # BEGIN SOLUTION
    nb_actions = 4

    converged: bool = False
    n_iter = 0
    states = [(y, x) for y in range(env.height) for x in range(env.width)]

    while not converged and n_iter < max_iter:

        Vip1 = np.zeros((env.height, env.width))

        for s in states:

            env.current_position = s
            y, x = s

            # Estimer la fonction de valeur de chaque état
            value_for_each_action = np.zeros(nb_actions)
            for a in range(0, nb_actions):
                sp, reward, _, _ = env.step(action=a, make_move=False)

                Pa_s_sp = env.moving_prob[y, x, a]
                Ra_s_sp = reward
                Vi_sp = Vi[sp]
                v = Pa_s_sp * (Ra_s_sp + gamma * Vi_sp)
                value_for_each_action[a] = v

            # Choisir l'action qui maximise la valeur
            m = value_for_each_action.max()

            # Mettre à jour la valeur de l'état
            Vip1[s] = m

        converged = np.allclose(Vi, Vip1, rtol=theta)
        Vi = Vip1
        n_iter += 1

    return Vi
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    Vi = np.zeros((4, 4))
    # BEGIN SOLUTION
    nb_actions = 4

    converged: bool = False
    n_iter = 0
    states = [(y, x) for y in range(env.height) for x in range(env.width)]

    while not converged and n_iter < max_iter:

        Vip1 = np.zeros((env.height, env.width))

        for s in states:

            env.current_position = s
            y, x = s

            # Estimer la fonction de valeur de chaque état
            value_for_each_action = np.zeros(nb_actions)
            for action_tried in range(0, nb_actions):
                my_sum = 0
                for sp, reward, prob, _, action_effective in env.get_next_states(
                    action_tried
                ):
                    Pa_s_sp = prob * env.moving_prob[y, x, action_effective]
                    Ra_s_sp = reward
                    Vi_sp = Vi[sp]
                    my_sum += Pa_s_sp * (Ra_s_sp + gamma * Vi_sp)

                value_for_each_action[action_tried] = my_sum

            # Choisir l'action qui maximise la valeur
            m = value_for_each_action.max()

            # Mettre à jour la valeur de l'état
            Vip1[s] = m

        converged = np.allclose(Vi, Vip1, rtol=theta)
        Vi = Vip1
        n_iter += 1

    return Vi
    # END SOLUTION
