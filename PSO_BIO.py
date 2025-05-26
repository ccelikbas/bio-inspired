import numpy as np
import matplotlib.pyplot as plt

def generate_large_test_case(num_aircraft=100, num_runways=2,
                             time_horizon=2000, min_window=30, max_window=150,
                             min_sep=3, max_sep=7, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = num_aircraft
    runways = np.random.randint(0, num_runways, size=n)
    E = np.random.uniform(0, time_horizon - max_window, size=n).round().astype(int)
    windows = np.random.uniform(min_window, max_window, size=n).round().astype(int)
    L = E + windows
    T = np.array([np.random.randint(E[i], L[i] + 1) for i in range(n)])
    g = np.random.randint(1, 6, size=n)
    h = np.random.randint(1, 6, size=n)
    SS = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and runways[i] == runways[j]:
                SS[i, j] = np.random.randint(min_sep, max_sep + 1)
    return E, L, T, g, h, SS, runways

class AircraftLandingProblem:
    def __init__(self, E, L, T, g, h, SS):
        self.n = len(E)
        self.E, self.L = np.array(E), np.array(L)
        self.T, self.g = np.array(T), np.array(g)
        self.h, self.SS = np.array(h), np.array(SS)

class PSO_ALP:
    def __init__(self, problem, S=50, w=0.7, c1=1.4, c2=1.4, max_iter=200):
        self.p = problem
        self.S, self.w = S, w
        self.c1, self.c2 = c1, c2
        self.max_iter = max_iter
        self.X = np.random.uniform(self.p.E, self.p.L, (S, self.p.n))
        self.V = np.zeros((S, self.p.n))
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(S, np.inf)
        self.gbest, self.gbest_cost = None, np.inf
        self.history = []

    def _decode_and_repair(self, Xk):
        xk = np.clip(np.rint(Xk), self.p.E, self.p.L)
        order = np.argsort(xk)
        for idx in range(1, len(order)):
            prev, cur = order[idx-1], order[idx]
            sep = self.p.SS[prev, cur]
            if xk[cur] - xk[prev] < sep:
                xk[cur] = xk[prev] + sep
        return xk

    def _fitness(self, Xk):
        xk = self._decode_and_repair(Xk)
        earliness = np.maximum(0, self.p.T - xk)
        tardiness = np.maximum(0, xk - self.p.T)
        return np.sum(self.p.g * earliness + self.p.h * tardiness)

    def optimize(self):
        for _ in range(self.max_iter):
            for k in range(self.S):
                cost = self._fitness(self.X[k])
                if cost < self.pbest_cost[k]:
                    self.pbest_cost[k] = cost
                    self.pbest[k] = self.X[k].copy()
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest = self.X[k].copy()
            r1, r2 = np.random.rand(self.S, self.p.n), np.random.rand(self.S, self.p.n)
            self.V = (self.w * self.V +
                      self.c1 * r1 * (self.pbest - self.X) +
                      self.c2 * r2 * (self.gbest - self.X))
            self.X += self.V
            self.history.append(self.gbest_cost)
        final_schedule = self._decode_and_repair(self.gbest)
        return final_schedule, self.gbest_cost, self.history

if __name__ == "__main__":
    # generate and solve
    E, L, T, g, h, SS, runways = generate_large_test_case(seed=42)
    alp = AircraftLandingProblem(E, L, T, g, h, SS)
    pso = PSO_ALP(alp, S=100, w=0.7, c1=1.4, c2=1.4, max_iter=300)
    schedule, cost, history = pso.optimize()

    print("Final landing times:", schedule)
    print("Total penalty cost:", cost)

    # plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Best cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("PSO Convergence")
    plt.grid(True)
    plt.show()

    # space–time per runway
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for r, ax in zip([0, 1], axes):
        idx_r = np.where(runways == r)[0]
        sched_r = schedule[idx_r]
        E_r, L_r, T_r = E[idx_r], L[idx_r], T[idx_r]
        order = np.argsort(sched_r)
        y = np.arange(len(order))
        ax.errorbar(
            T_r[order], y,
            xerr=[T_r[order] - E_r[order], L_r[order] - T_r[order]],
            fmt='o', color='gray', ecolor='gray', alpha=0.6,
            label='Target window'
        )
        ax.scatter(
            sched_r[order], y,
            marker='x', color='C1', s=50,
            label='Scheduled time'
        )
        ax.set_title(f'Runway {r}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Landing sequence')
        ax.legend()
        ax.grid(True)

    plt.suptitle('Space–Time Network per Runway (Sorted by Schedule)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
