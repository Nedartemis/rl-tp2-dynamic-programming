"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    # BEGIN SOLUTION
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)
    # END SOLUTION


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.


def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """

    # BEGIN SOLUTION
    memo = [-1] * max(n + 1, 2)
    memo[0] = 0
    memo[1] = 1

    def rec(n2: int) -> int:
        if memo[n2] == -1:
            memo[n2] = rec(n2 - 2) + rec(n2 - 1)
        return memo[n2]

    return rec(n)
    # END SOLUTION
