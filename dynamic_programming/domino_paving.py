# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    # BEGIN SOLUTION
    memo = [-1] * max(n + 1, 3)
    memo[2] = 3

    def rec(n2: int):
        if memo[n2] == -1:
            memo[n2] = (
                rec(n2 - 2) * rec(2)
                + sum(rec(k) for k in range(2, n2 - 4 + 1, 2)) * 2
                + 2
            )
        return memo[n2]

    if n % 2 == 1:
        return 0

    res = rec(n)
    print(memo)
    return res

    # END SOLUTION
