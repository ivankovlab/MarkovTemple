import numpy as np
from pydtmc import MarkovChain
from random import choices


class Rules:

    def __init__(self, bond_weights: dict):
        self.bond_weights = bond_weights
        self.define_bond_weights()

    def define_bond_weights(self):
        # Complete the bonding weights table to symmetric.
        # Dictionary of bonding propensities for residues.
        for pair in list(self.bond_weights.keys()):
            a, b = pair[0], pair[1]
            self.bond_weights[(b, a)] = self.bond_weights[(a, b)]

    def segment(self, a: int, b: int):
        """Create generator of segment of whole numbers
           from a to b including both boundaries.
        """

        return range(a, b+1)


    def calc_P_abs_at(self, P: np.array, B: np.array, absorb_num: int, avoid_num: int):
        """Calculate transition matrix for a specified absorbing state of two.

        Parameters
            P : np.array[float]
                Transition matrix N x N.
            B : np.array[float]
                N x 2 matrix of absorbing probabilities.
            absorb_num : int
                Number of the absorbing state.
            avoid_num : int
                Number of tje absorbing state that is avoided.

        Returns
            Transition matrix (N-1) x (N-1).
        """

        P_abs = np.zeros(P.shape, dtype=float)

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if avoid_num == i: continue
                P_abs[i, j] = B[j, absorb_num] / B[i, absorb_num] * P[i, j]

        P_abs = np.delete(P_abs, avoid_num, 0)
        P_abs = np.delete(P_abs, avoid_num, 1)

        return P_abs


    def get_bond_weight(self, seq_1: str, seq_2: str, n: int, m: int) -> float:
        """Calculate statistical weight of hydrogen bond formation
           between n-th residue of the first sequence and m-th
           residue of the second sequence."""

        return self.bond_weights[(seq_1[n-1],seq_2[m-1])][0]


    def get_unbond_weight(self, seq_1: str, seq_2: str, n: int, m: int) -> float:
        """Calculate statistical weight of hydrogen bond breakage
           between n-th residue of the first sequence and m-th
           residue of the second sequence."""

        return self.bond_weights[(seq_1[n-1],seq_2[m-1])][1]


    def calc_mean_residence_times_shift(self, seq_1: str, seq_2: str,
                                        s: int) -> float:
        """Calculate mean residence times for all incorrect docks with the given shift."""

        assert len(seq_1) == len(seq_2)

        N = len(seq_1)
        all_monomer = self.segment(1, N)

        # Select available position of contacts of the first monomer.
        if s == 0:
            n_range = self.segment(1, N)
        elif s > 0:
            n_range = self.segment(1, N - s)
        else:
            n_range = self.segment(1 - s, N)

        num = dict()
        cnt = 0
        for n_1 in n_range:
            for n_2 in n_range:
                if n_1 > n_2: continue
                num[(n_1, n_2)] = cnt
                cnt += 1
        num[0] = cnt
        cnt += 1

        P = np.zeros((cnt, cnt), dtype=float)

        for n_1 in n_range:
            for n_2 in n_range:
                if n_1 > n_2: continue
                if (n_2 + 1 in n_range) and (n_2 + 1 + s in all_monomer):
                    P[num[(n_1,n_2)], num[(n_1,n_2+1)]] = self.get_bond_weight(seq_1, seq_2, n_2 + 1, n_2 + 1 + s)
                if n_1 < n_2:
                    P[num[(n_1,n_2)], num[(n_1,n_2-1)]] = self.get_unbond_weight(seq_1, seq_2, n_2, n_2 + s)
                if (n_1 - 1 in n_range) and (n_1 - 1 + s in all_monomer):
                    P[num[(n_1,n_2)], num[(n_1-1,n_2)]] = self.get_bond_weight(seq_1, seq_2, n_1 - 1, n_1 - 1 + s)
                if n_1 < n_2:
                    P[num[(n_1,n_2)], num[(n_1+1,n_2)]] = self.get_unbond_weight(seq_1, seq_2, n_1, n_1 + s)
                if n_1 == n_2:
                    P[num[(n_1,n_2)], num[0]] = self.get_unbond_weight(seq_1, seq_2, n_1, n_1 + s)

        P[num[0], num[0]] = 1

        for row in P:
            row /= sum(row)

        mc = MarkovChain(P)
        mean_absorption_times = mc.mean_absorption_times()

        mean_absorption_times_dock = []
        for n in n_range:
            mean_absorption_times_dock.append(mean_absorption_times[num[(n, n)]])

        return mean_absorption_times_dock


    def calc_mean_grow_time(self, seq: str, step_time: float, verbose=False) -> float:
        """Calculate mean time between monomer attachments to a protofilament in ns."""

        N = len(seq)

        residence_times = []

        # Calculate residence times for all misaligned docks with same and opposite orientations.
        seq_rev = seq[::-1]

        # All possible incorrect shifts between chains, including out-of-register.
        for s in self.segment(1-N, N-1):
            if s != 0:  # For in-register orientations, zero shift is a correct dock.
                residence_times += self.calc_mean_residence_times_shift(seq, seq, s)
            # Any dock is incorrect for opposite orientations.
            residence_times += self.calc_mean_residence_times_shift(seq, seq_rev, s)

        # Calculate residence times for lock and fail scenarios for the correctly aligned docks.
        num = dict()
        cnt = 0
        for n in list(self.segment(1, N))[::-1]:  # There are N correct dock states for chains of N residues.
            for m in self.segment(n, N):
                num[(n, m)] = cnt  # Enumerate all (n, m) dock states.
                cnt += 1
        num[0] = cnt  # Enumerate fail state as the last one.
        cnt += 1

        # Initiate the transition matrix.
        P = np.zeros((cnt, cnt), dtype=float)

        for n in list(self.segment(1, N))[::-1]:
            for m in self.segment(n, N):
                if n == 1 and m == N: continue  # It is the lock state, there are no transitions from it.
                if m < N:  # Attach in C-direction.
                    P[num[(n,m)], num[(n,m+1)]] = self.get_bond_weight(seq, seq, m+1, m+1)
                if n < m:  # Detach in C-direction.
                    P[num[(n,m)], num[(n,m-1)]] = self.get_unbond_weight(seq, seq, m, m)
                if 1 < n:  # Attach in N-direction.
                    P[num[(n,m)], num[(n-1,m)]] = self.get_bond_weight(seq, seq, n-1, n-1)
                if n < m:  # Detach in N-direction.
                    P[num[(n,m)], num[(n+1,m)]] = self.get_unbond_weight(seq, seq, n, n)
                if n == m:  # Fail.
                    P[num[(n,m)], num[0]] = self.get_unbond_weight(seq, seq, n, n)

        P[num[(1, N)], num[(1, N)]] = 1  # Dock is the terminal state.
        P[num[0], num[0]] = 1  # Fail is the terminal state.

        if verbose:
            print("Markov chain built.")

        # Normalize the transition probabilities for each state.
        for row in P:
            row /= sum(row)

        # Define numbers of transient and absorbing states.
        t = cnt - 2  # Transient.
        r = 2        # Absorbing.

        # Submatrix for inner transitions between transient states only.
        Q = P[:t, :t]
        # Submatrix for transitions from transient to absorbing states.
        R = P[:t, t:]

        # Calculate limiting matrix of transient states, for infinite time limit.
        F = np.linalg.inv(np.identity(t) - Q)

        # Calculate limiting matrix of absorbance.
        B = F @ R

        # Probabilities of absorbing states.
        p_lock = np.sum(B[:, 0]) / np.sum(B)
        p_fail = np.sum(B[:, 1]) / np.sum(B)

        # Add terminal states to the limiting absorbance matrix.
        B = np.vstack((B, np.identity(2)))

        # Calculate transition matrices for selected absorbing states.
        P_lock = self.calc_P_abs_at(P, B, 0, num[0])
        for row in P_lock:
            row /= sum(row)
        P_fail = self.calc_P_abs_at(P, B, 1, num[(1, N)])

        # Build Markov chains for selected absorbing states.
        MC_lock = MarkovChain(P_lock)
        mean_absorption_times_lock = MC_lock.mean_absorption_times()[:-1]
        mean_time_lock = 0
        for n in self.segment(1, N):
            mean_time_lock += mean_absorption_times_lock[num[(n,n)]]
        mean_time_lock /= N

        MC_fail = MarkovChain(P_fail)
        mean_absorption_times_fail = MC_fail.mean_absorption_times()[:-1]
        mean_time_fail = 0
        for n in self.segment(1, N):
            mean_time_fail += mean_absorption_times_fail[num[(n,n)]]
        mean_time_fail /= N

        num_incorrect_docks = len(residence_times)
        num_docks = num_incorrect_docks + N
        p_fail = num_incorrect_docks / num_docks + p_fail * N / num_docks

        mean_time_fail = np.mean(residence_times) * num_incorrect_docks / num_docks + mean_time_fail * N / num_docks

        if verbose:
            print("mean fail probability mixed", p_fail)
            print("total number of dock events", num_docks)
            print("number of incorrect dock events", num_incorrect_docks)
            print("mean fail time (ms)", mean_time_fail / 10**6, "mean residence time (ms)", np.mean(residence_times) / 10**6)
            print("mean fail time mixed (ms)", mean_time_fail / 10**6)
            print(p_fail / (1 - p_fail))
            print("mean lock time (ns)", mean_time_lock)

        return (mean_time_lock + mean_time_fail * p_fail / (1 - p_fail)) * step_time


    def calc_mean_grow_rate(self, seq: str, step_time: float, verbose=False) -> float:
        """Calculate protofilament growth rate in nm / min.

        Parameters
            seq : str
                Polypeptide sequence.
            step_time : float
                Mean time of formation and breakage of hydrogen bond [ns].

        Return
            Mean elongation rate [nm / min].
        """

        return 1 / self.calc_mean_grow_time(seq, step_time, verbose) * 10**9 * 60

    def calc_mean_grow_rate_stepwise(self, seqs: list, step_time: float, verbose=False) -> float:
        '''Consequential segment-wise attachments. Time of consequential chemical reactions:
           1 / time = 1 / time_1 + ... + 1 / time_S, where S is the number of irreversible segments.'''

        return 1 / sum(self.calc_mean_grow_time(seq, step_time, verbose) for seq in seqs) * 10**9 * 60

    def gen_rand_seq(self, distribution: dict, length: int) -> str:
        return "".join(choices(
                       population=list(distribution.keys()),
                       weights=list(distribution.values()),
                       k=length
                      ))
