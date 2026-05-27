from fractions import Fraction
from itertools import combinations, permutations, product
from collections import defaultdict


# ============================================================
#  Paw graph: 1--2--3--4--2
# ============================================================

V = (1, 2, 3, 4)

EDGES = {
    tuple(sorted(e))
    for e in [(1, 2), (2, 3), (3, 4), (2, 4)]
}

MU = {
    1: Fraction(1, 5),    # 0.20
    2: Fraction(7, 20),   # 0.35
    3: Fraction(1, 5),    # 0.20
    4: Fraction(1, 4),    # 0.25
}

# Truncation convention:
#     W_K = {x in W : |x| < K}
K_VALUES = [35]

DECIMAL_DIGITS = 10


# ============================================================
#  Basic graph and probability utilities
# ============================================================

def build_neighbors(V, edges):
    neighbors = {i: set() for i in V}
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


NEIGHBORS = build_neighbors(V, EDGES)


def mu_sum(S):
    """Return mu(S) exactly as a Fraction."""
    return sum((MU[i] for i in S), Fraction(0, 1))


def E(S):
    """Return the set of neighbors E(S)."""
    out = set()
    for i in S:
        out |= NEIGHBORS[i]
    return out


def is_independent_set(S):
    """Check whether S is an independent set of the graph."""
    S = list(S)

    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            if tuple(sorted((S[a], S[b]))) in EDGES:
                return False

    return True


def is_admissible_word(word):
    """
    A word is admissible if all letters belong to V and its support is
    an independent set.
    """
    if any(i not in V for i in word):
        return False

    return is_independent_set(set(word))


def independent_sets(include_empty=False):
    """List all independent sets of the graph."""
    result = []
    start = 0 if include_empty else 1

    for r in range(start, len(V) + 1):
        for S in combinations(V, r):
            S = frozenset(S)
            if is_independent_set(S):
                result.append(S)

    return result


def validate_model():
    """Check that MU is a probability vector and satisfies stability."""
    total_mu = sum(MU.values(), Fraction(0, 1))

    if total_mu != 1:
        raise ValueError(f"MU is not a probability vector. Sum = {total_mu}")

    if any(MU[i] <= 0 for i in V):
        raise ValueError("All arrival probabilities must be strictly positive.")

    violations = []

    for I in independent_sets(include_empty=False):
        lhs = mu_sum(I)
        rhs = mu_sum(E(I))

        if lhs >= rhs:
            violations.append((I, lhs, rhs))

    if violations:
        msg = ["The stability condition mu(I) < mu(E(I)) is violated:"]
        for I, lhs, rhs in violations:
            msg.append(f"  I={set(I)}, mu(I)={lhs}, mu(E(I))={rhs}")
        raise ValueError("\n".join(msg))


# ============================================================
#  Product-form weight H(x)
#
#  H(x_1...x_q)
#  =
#  prod_{l=1}^q mu(x_l) / mu(E({x_1,...,x_l}))
#
#  with H(empty) = 1.
# ============================================================

def H(word):
    if not is_admissible_word(word):
        raise ValueError(f"Word {word} is not admissible.")

    value = Fraction(1, 1)
    support = set()

    for i in word:
        support.add(i)
        denominator = mu_sum(E(support))

        if denominator <= 0:
            raise ValueError(f"Zero denominator for support {support}.")

        value *= MU[i] / denominator

    return value


# ============================================================
#  Original inverse normalising constant alpha^{-1}
#
#  alpha^{-1}
#  =
#  1
#  +
#  sum_{I independent}
#  sum_{sigma in S_|I|}
#  prod_j
#  mu(i_sigma(j))
#  /
#  [mu(E({i_sigma(1),...,i_sigma(j)}))
#   -
#   mu({i_sigma(1),...,i_sigma(j)})]
#
#  This follows the product-form normalising constant formula:
#  all independent sets and all orderings must be considered.
# ============================================================

def alpha_inverse_original():
    total = Fraction(1, 1)

    for I in independent_sets(include_empty=False):
        for sigma in permutations(tuple(I)):
            product_value = Fraction(1, 1)
            prefix = set()

            for i in sigma:
                prefix.add(i)

                denominator = mu_sum(E(prefix)) - mu_sum(prefix)

                if denominator <= 0:
                    raise ValueError(
                        f"Non-positive stability denominator for prefix {prefix}: "
                        f"{denominator}"
                    )

                product_value *= MU[i] / denominator

            total += product_value

    return total


# ============================================================
#  Truncated total weight Z_K
#
#  Z_K = sum_{x in W_K} H(x),  W_K = {x in W : |x| < K}.
#
#  The computation does not enumerate all words. It groups words
#  by their current support set.
#
#  dp[S] at length n =
#      sum of H(x) over all admissible words x of length n
#      with support S.
# ============================================================

def truncated_weight_and_length_sum(K):
    if K <= 0:
        raise ValueError("K must be positive.")

    current_dp = {frozenset(): Fraction(1, 1)}

    # Contribution of the empty word.
    Z_K = Fraction(1, 1)
    L_K = Fraction(0, 1)

    # Add words of length 1, ..., K-1.
    for length in range(1, K):
        next_dp = defaultdict(Fraction)

        for support, total_weight in current_dp.items():
            for i in V:
                new_support = frozenset(set(support) | {i})

                if is_independent_set(new_support):
                    denominator = mu_sum(E(new_support))

                    if denominator <= 0:
                        raise ValueError(
                            f"Zero denominator for support {new_support}."
                        )

                    factor = MU[i] / denominator
                    next_dp[new_support] += total_weight * factor

        level_weight = sum(next_dp.values(), Fraction(0, 1))

        Z_K += level_weight
        L_K += length * level_weight

        current_dp = next_dp

    return Z_K, L_K


# ============================================================
#  Probabilities
#
#  Original:
#      pi(x) = H(x) / alpha_inverse.
#
#  Truncated:
#      pi^K(x) = H(x) / Z_K, if x in W_K,
#              = 0, otherwise.
#
#  Correction factor:
#      c_K = pi^K(x) / pi(x)
#          = alpha_inverse / Z_K,
#  for every x in W_K with H(x)>0.
# ============================================================

def pi_original(word, alpha_inv):
    if not is_admissible_word(word):
        return Fraction(0, 1)

    return H(word) / alpha_inv


def pi_truncated(word, K, Z_K):
    if not is_admissible_word(word):
        return Fraction(0, 1)

    if len(word) >= K:
        return Fraction(0, 1)

    return H(word) / Z_K


def correction_factor(alpha_inv, Z_K):
    return alpha_inv / Z_K


# ============================================================
#  Formatting utilities
# ============================================================

def dec(x, digits=DECIMAL_DIGITS):
    return f"{float(x):.{digits}f}"


def print_summary(K_values):
    validate_model()

    alpha_inv = alpha_inverse_original()

    print("Original inverse normalising constant")
    print("-------------------------------------")
    print("alpha_inverse exact   =", alpha_inv)
    print("alpha_inverse decimal =", dec(alpha_inv))
    print()

    print("Truncation summary")
    print("------------------")
    header = (
        f"{'K':>5s} "
        f"{'Z_K':>18s} "
        f"{'pi(W_K)':>18s} "
        f"{'c_K':>18s} "
        f"{'tail':>18s} "
        f"{'E_K[|W|]':>18s}"
    )
    print(header)
    print("-" * len(header))

    results = {}

    for K in K_values:
        Z_K, L_K = truncated_weight_and_length_sum(K)

        mass_inside = Z_K / alpha_inv
        c_K = correction_factor(alpha_inv, Z_K)
        tail = Fraction(1, 1) - mass_inside
        mean_length = L_K / Z_K

        results[K] = {
            "Z_K": Z_K,
            "mass_inside": mass_inside,
            "c_K": c_K,
            "tail": tail,
            "mean_length": mean_length,
        }

        print(
            f"{K:5d} "
            f"{dec(Z_K):>18s} "
            f"{dec(mass_inside):>18s} "
            f"{dec(c_K):>18s} "
            f"{dec(tail):>18s} "
            f"{dec(mean_length):>18s}"
        )

    print()
    return alpha_inv, results


# ============================================================
#  Ratio check
#
#  This function is deliberately strict:
#  - if |x| >= K, then x is outside W_K and pi^K(x)=0;
#  - no ratio is printed for outside states;
#  - no ratio is printed for inadmissible states.
# ============================================================

def ratio_check(K, test_states, alpha_inv, Z_K):
    c_K = correction_factor(alpha_inv, Z_K)

    print(f"Ratio check for K = {K}")
    print("---------------------")
    print("The theoretical correction factor is")
    print("c_K exact   =", c_K)
    print("c_K decimal =", dec(c_K))
    print()

    header = (
        f"{'state':>28s} "
        f"{'status':>14s} "
        f"{'H(x)':>14s} "
        f"{'pi(x)':>14s} "
        f"{'pi^K(x)':>14s} "
        f"{'ratio':>14s}"
    )
    print(header)
    print("-" * len(header))

    for word in test_states:
        word = tuple(word)

        if not is_admissible_word(word):
            print(
                f"{str(word):>28s} "
                f"{'inadmissible':>14s} "
                f"{'-':>14s} "
                f"{'-':>14s} "
                f"{'-':>14s} "
                f"{'-':>14s}"
            )
            continue

        if len(word) >= K:
            pi_x = pi_original(word, alpha_inv)
            print(
                f"{str(word):>28s} "
                f"{'outside W_K':>14s} "
                f"{dec(H(word)):>14s} "
                f"{dec(pi_x):>14s} "
                f"{dec(Fraction(0, 1)):>14s} "
                f"{'-':>14s}"
            )
            continue

        h_x = H(word)
        pi_x = pi_original(word, alpha_inv)
        piK_x = pi_truncated(word, K, Z_K)
        ratio = piK_x / pi_x

        print(
            f"{str(word):>28s} "
            f"{'inside W_K':>14s} "
            f"{dec(h_x):>14s} "
            f"{dec(pi_x):>14s} "
            f"{dec(piK_x):>14s} "
            f"{dec(ratio):>14s}"
        )

    print()


# ============================================================
#  Optional automatic test-state generation
#
#  For small K, we can enumerate all words in W_K to check.
#  For large K, this should not be used.
# ============================================================

def generate_admissible_words_below_K(K):
    for length in range(K):
        for word in product(V, repeat=length):
            if is_admissible_word(word):
                yield word


# ============================================================
#  Main execution
# ============================================================

if __name__ == "__main__":
    alpha_inv, results = print_summary(K_VALUES)

    # Use automatic exhaustive ratio check only for small K.
    # For K=5 this is safe. It enumerates words of length 0,1,2,3,4 only.
    K_check = K_VALUES[0]
    Z_K_check = results[K_check]["Z_K"]

    if K_check <= 6:
        test_states = list(generate_admissible_words_below_K(K_check))
    else:
        test_states = [
            (),
            (1,),
            (2,),
            (3,),
            (4,),
            (1,1),
            (1,3),
            (3,1),
            (1,1,1),
            (1,1,1,1),
            (1,1,1,1,1,1,1,1,1,1,1,1),
            (1,3,1,3,1,3,1,3,1,3,1,3),
        ]

    # Add two deliberately outside states to verify that the program
    # correctly assigns pi^K(x)=0 and does not compute a ratio.
    test_states += [
        (1,) * 12,
        (1, 3) * 6,
    ]

    ratio_check(
        K=K_check,
        test_states=test_states,
        alpha_inv=alpha_inv,
        Z_K=Z_K_check,
    )