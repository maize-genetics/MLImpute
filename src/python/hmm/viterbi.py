import torch


def build_pair_states(n_parents):
    """Return list [(i,j) …] with i ≤ j covering all unordered parent pairs."""
    return [(i, j) for i in range(n_parents) for j in range(i, n_parents)]


def make_diploid_emissions(log_em, pair_states):
    """
    Diploid emission = sum of two haploid log-probs (independent chromosomes).
    log_em:  T × N
    returns: T × P  where P=len(pair_states)
    """
    T, _ = log_em.shape
    dip = torch.empty(T, len(pair_states), dtype=log_em.dtype, device=log_em.device)
    for k, (i, j) in enumerate(pair_states):
        dip[:, k] = log_em[:, i] + log_em[:, j]
    return dip


def make_diploid_transitions(log_A, pair_states):
    """
    Diploid transition under independent chromosome movement:
      P((a,b)→(c,d)) = A[a,c] * A[b,d]   (unordered c≤d)
    log_A : N × N
    returns: P × P log-prob matrix
    """
    P = len(pair_states)
    dip = torch.empty(P, P, dtype=log_A.dtype, device=log_A.device)
    for p, (a, b) in enumerate(pair_states):
        outer = log_A[a].unsqueeze(1) + log_A[b]          # N × N
        for q, (c, d) in enumerate(pair_states):
            dip[p, q] = outer[c, d]
    return dip


def viterbi_decode(log_emit, log_trans, log_start):
    """
    Generic log-space Viterbi for a discrete HMM.
    log_emit : T × S
    log_trans: S × S
    log_start: S
    returns   : list[int] length T (best-path state indices)
    """
    T, S = log_emit.shape
    dp = torch.full((T, S), float('-inf'), dtype=log_emit.dtype, device=log_emit.device)
    bp = torch.zeros((T, S), dtype=torch.long, device=log_emit.device)

    dp[0] = log_start + log_emit[0]
    for t in range(1, T):
        scores = dp[t - 1].unsqueeze(1) + log_trans  # S × S
        dp[t], bp[t] = torch.max(scores, dim=0)      # best prev → state j
        dp[t] += log_emit[t]

    # backtrack
    state = torch.argmax(dp[-1]).item()
    path = [state]
    for t in range(T - 1, 0, -1):
        state = bp[t, state].item()
        path.append(state)
    path.reverse()
    return path
