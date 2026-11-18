"""
Realistic BB84 quantum key distribution simulation - engine.

PROTOCOL (what modern fibre QKD systems actually run)
  * Efficient BB84: Alice and Bob pick the Z basis with probability q_Z (e.g. 0.9).
    Z-Z rounds form the key; X-X rounds are announced to estimate the phase error.
  * Weak laser pulses with 3 intensities (signal, decoy, vacuum), chosen at random.
  * Finite-key security bounds (Lim, Curty, Walenta, Xu & Zbinden, PRA 89, 022307, 2014),
    with Chernoff bounds for the statistical fluctuations.
  * Real post-processing on the actual bit strings:
      Cascade error correction -> verification hash -> Toeplitz privacy amplification,
      with the classical messages authenticated (Wegman-Carter), which costs key.

PHYSICS
  Source  : phase-randomized coherent pulses, intensity fluctuations, imperfect X-state
            preparation, optical isolator (Trojan-horse protection)
  Channel : fibre loss (dB/km) + fixed loss, depolarizing noise, slow polarization drift
  Bob     : optical loss, active basis choice, two single-photon detectors with
            different efficiencies, dark counts, afterpulsing, dead time, double clicks
            (-> random bit), random detector scrambling, random efficiency-check slots
  Qiskit  : every photon's polarization measurement is simulated with Qiskit Aer.

EAVESDROPPER (placed at Alice's output, the strongest position)
  intercept_resend, breidbart  - measure and re-send, compensating the loss
  pns                          - photon-number splitting
  blinding                     - blinds Bob's detectors with bright light and controls them
  timeshift                    - exploits detector-efficiency mismatch by shifting arrival time
  trojan                       - shines light into Alice's encoder and reads the reflection

SPEED
  A session has ~1e9 pulses (one second at 1 GHz). Only pulses that can produce a detector
  click are simulated individually; the rest are counted exactly in bulk. This is
  statistically exact because pulses are independent.
"""
import numpy as np
from math import exp, log, log2, sqrt, ceil, factorial, comb

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

TWO_PI = 2 * np.pi
TABLE_POINTS = 2048                          # resolution of the Qiskit lookup table
_QISKIT_TABLES = {}                          # shared between links with the same noise
COS2_PI8 = np.cos(np.pi / 8) ** 2           # Breidbart guessing probability (0.854)


# ============================================================================
# Small helpers
# ============================================================================
def h2(p):
    """Binary entropy, safe for arrays and scalars."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def xor_prob(p, q):
    """Probability that exactly one of two independent bit flips happens."""
    return p + q - 2 * p * q


def db_to_t(db):
    return 10 ** (-db / 10)


# ============================================================================
# Qiskit part
# ============================================================================
def photon_circuit(angle):
    """
    One photon on the Z-X great circle of the Bloch sphere.
    A state prepared at angle a = ry(a)|0>, measured in a basis at angle b, behaves
    like ry(a - b)|0> measured in Z; drift is one more ry rotation. Depolarizing noise
    commutes with rotations, so it can sit after the single combined rotation.
      Z basis: 0    X basis: pi/2    Breidbart basis: pi/4    bit b adds b*pi
    """
    qc = QuantumCircuit(1, 1)
    qc.ry(float(angle), 0)
    qc.id(0)                       # the channel: depolarizing noise is attached here
    qc.measure(0, 0)
    return qc


def make_backend(p_channel, p_pbs):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p_channel, 1), ["id"])
    noise.add_all_qubit_readout_error(
        ReadoutError([[1 - p_pbs, p_pbs], [p_pbs, 1 - p_pbs]]))
    # density-matrix method: exact state, then samples any number of shots cheaply
    return AerSimulator(method="density_matrix", noise_model=noise)


# ============================================================================
# Post-processing building blocks
# ============================================================================
def cascade(a, b, qber_est, rng, passes=4):
    """
    Cascade error correction (Brassard & Salvail), batched the way real implementations
    run it: all blocks of a pass are handled in parallel in each communication round.
    Alice's string `a` is used only to answer parity questions.
    Returns Bob's corrected string, bits leaked to Eve, and communication rounds.
    """
    n = len(a)
    a_ = np.append(a, 0).astype(np.int8)          # index n = padding
    b_ = np.append(b, 0).astype(np.int8)
    k1 = int(np.clip(0.73 / max(qber_est, 1e-4), 4, max(4, n)))
    passes_info = []
    leaked, rounds = 0, 0

    def fix_blocks(q, blocks):
        """Check the given blocks of pass q; binary-search one error in each odd one."""
        nonlocal leaked, rounds
        perm_pad, inv, k = passes_info[q]
        if len(blocks) == 0:
            return np.empty(0, dtype=np.int64)
        idx = perm_pad[blocks[:, None] * k + np.arange(k)]
        d = a_[idx] ^ b_[idx]
        odd = (d.sum(axis=1) & 1).astype(bool)
        if not odd.any():
            return np.empty(0, dtype=np.int64)
        first = np.argmax(d[odd], axis=1)
        pos = idx[odd, first]
        b_[pos] ^= 1
        steps = max(1, ceil(log2(k)))
        leaked += int(odd.sum()) * steps
        rounds += steps
        return pos

    for p in range(passes):
        k = int(min(n, k1 * 2 ** p))
        perm = np.arange(n) if p == 0 else rng.permutation(n)
        nb = ceil(n / k)
        perm_pad = np.concatenate([perm, np.full(nb * k - n, n)])
        inv = np.empty(n, dtype=np.int64)
        inv[perm] = np.arange(n)
        passes_info.append((perm_pad, inv, k))
        leaked += nb                                   # Alice announces all block parities
        rounds += 1
        queue = {p: np.arange(nb)}
        while queue:
            q, blocks = queue.popitem()
            pos = fix_blocks(q, blocks)
            if len(pos) == 0:
                continue
            for r in range(p + 1):                     # cascade back to every other pass
                if r == q:
                    continue
                _, inv_r, k_r = passes_info[r]
                cand = np.unique(inv_r[pos] // k_r)
                queue[r] = np.unique(np.concatenate([queue.get(r, cand), cand]))
    return b_[:n], leaked, rounds


def toeplitz_hash(x_list, out_len, rng):
    """
    Two-universal Toeplitz hashing of several equal-length bit strings with ONE random
    seed (as Alice and Bob do). Computed as a convolution via FFT.
    """
    n = len(x_list[0])
    if out_len <= 0 or n == 0:
        return [np.zeros(0, dtype=np.int8) for _ in x_list]
    seed = rng.integers(0, 2, n + out_len - 1).astype(np.float64)
    size = 1 << int(ceil(log2(len(seed) + n)))
    fs = np.fft.rfft(seed, size)
    out = []
    for x in x_list:
        conv = np.fft.irfft(fs * np.fft.rfft(x.astype(np.float64), size), size)
        out.append((np.rint(conv[n - 1:n - 1 + out_len]).astype(np.int64) & 1)
                   .astype(np.int8))
    return out


def finite_key(nZ, nX, mX, lam_ec, mu, p_mu, eps_sec, eps_cor, asymptotic=False):
    """
    Secret key length for 3-intensity decoy BB84 with finite-size effects
    (Lim et al. 2014). nZ, nX: detections per intensity in Z-Z / X-X rounds;
    mX: errors per intensity in X-X rounds; lam_ec: bits leaked by error correction.
    Returns a dict with every term so the key budget can be shown.
    """
    m1, m2, m3 = mu
    p = np.asarray(p_mu, float)
    mu_a = np.asarray(mu, float)
    nZ, nX, mX = (np.asarray(v, float) for v in (nZ, nX, mX))

    def tau(n):
        return float(np.sum(p * np.exp(-mu_a) * mu_a ** n / factorial(n)))

    t0, t1 = tau(0), tau(1)

    beta = log(21 / eps_sec)          # each of the ~21 bounds fails with prob <= eps/21

    def bounds(n_k, total):
        """
        Bounds on the expected counts per intensity. Uses multiplicative Chernoff
        bounds on each observed count (tighter than Lim et al.'s Hoeffding bound,
        as in Zhang et al. PRA 95 012333 (2017) and Yin et al. Sci. Rep. 10 14312 (2020)).
        """
        n_k = np.maximum(n_k, 0)
        if asymptotic:
            lo = hi = n_k
        else:
            lo = np.maximum(0.0, n_k - np.sqrt(2 * beta * n_k))
            hi = n_k + beta + np.sqrt(2 * beta * n_k + beta ** 2)
        return np.exp(mu_a) / p * lo, np.exp(mu_a) / p * hi

    def s01(n_k):
        lo, hi = bounds(n_k, n_k.sum())
        s0 = max(0.0, t0 * (m2 * lo[2] - m3 * hi[1]) / (m2 - m3))
        s1 = (t1 * m1 * (lo[1] - hi[2] - (m2 ** 2 - m3 ** 2) / m1 ** 2 * (hi[0] - s0 / t0))
              / (m1 * (m2 - m3) - m2 ** 2 + m3 ** 2))
        return s0, max(0.0, s1)

    sZ0, sZ1 = s01(nZ)
    _, sX1 = s01(nX)
    lo_m, hi_m = bounds(mX, mX.sum())
    vX1 = max(0.0, t1 * (hi_m[1] - lo_m[2]) / (m2 - m3))

    if sX1 <= 0 or sZ1 <= 0:
        phi = 0.5
    else:
        b = min(0.5, vX1 / sX1)
        gam = 0.0
        if not asymptotic and 0 < b < 0.5:
            c, d = sZ1, sX1
            arg = (c + d) / (c * d * (1 - b) * b) * (21 ** 2) / eps_sec ** 2
            gam = sqrt((c + d) * (1 - b) * b / (c * d * log(2)) * log2(max(arg, 1.0)))
        phi = min(0.5, b + gam)

    margin = 0.0 if asymptotic else 6 * log2(21 / eps_sec) + log2(2 / eps_cor)
    ell = sZ0 + sZ1 * (1 - float(h2(phi))) - lam_ec - margin
    return {"sZ0": sZ0, "sZ1": sZ1, "sX1": sX1, "vX1": vX1, "phi": phi,
            "lam_ec": lam_ec, "margin": margin, "ell": max(0.0, ell), "ell_raw": ell,
            "nZ_total": float(nZ.sum())}


# ============================================================================
# The link
# ============================================================================
class BB84Link:
    DEFAULTS = dict(
        # ---- source (Alice) ----
        rep_rate_hz=1e9,               # pulses per second
        mu=(0.5, 0.15, 0.0),           # signal, decoy, vacuum (mean photons per pulse)
        p_mu=(0.6, 0.3, 0.1),          # how often each intensity is sent
        intensity_fluct=0.03,          # relative pulse-to-pulse intensity noise (sd)
        q_z=0.8,                       # probability of the Z (key) basis, Alice and Bob
        x_prep_error=0.05,             # X states prepared off by this angle (radians)
        alice_isolation_db=60.0,       # optical isolator in front of Alice's encoder
        # ---- channel ----
        distance_km=50.0,
        alpha_db_km=0.2,               # standard telecom fibre at 1550 nm
        fixed_loss_db=0.0,             # connectors, splices, free-space link, ...
        p_channel=0.02,                # depolarizing noise (flips a bit p/2 of the time)
        drift_sd=0.05,                 # polarization drift kick per chunk (radians)
        drift_pull=0.1,                # active compensation pulls the drift back
        chunk_pulses=100_000_000,      # drift updates once per chunk (0.1 s at 1 GHz)
        qiskit_mode="table",           # "table": Qiskit computes the exact outcome
                                       #   probabilities once, photons are then sampled
                                       #   from them (fast); "shots": one Qiskit run per
                                       #   group of photons (slow, same statistics)
        # ---- Bob ----
        bob_loss_db=2.0,               # Bob's optics insertion loss
        det_eff=(0.20, 0.18),          # efficiency of detector 0 and detector 1
        p_pbs=0.01,                    # polarizing beam splitter sends photon to wrong detector
        p_dark=2e-6,                   # dark-count probability per detector per pulse slot
        p_afterpulse=0.02,             # chance a click causes a later false click
        afterpulse_decay_slots=20,     # afterpulses spread over this many slots
        dead_time_ns=10.0,             # both detectors blind after any click
        scramble=True,                 # random detector assignment (vs time-shift attacks)
        check_fraction=1e-3,           # random slots with detectors switched off (vs blinding)
        jitter_ps=50.0,                # detector timing jitter (standard deviation)
        gate_ps=1000.0,                # detection gate; dark counts land anywhere in it
        # ---- Alice's monitor photodiode (watchdog against Trojan-horse light) ----
        watchdog_bg=3.0,               # background counts per telemetry timestep
        # ---- post-processing / security ----
        cascade_passes=4,
        f_ec_theory=1.16,              # Cascade efficiency used for theory curves
        eps_sec=1e-10,                 # secrecy parameter
        eps_cor=1e-15,                 # correctness parameter (verification hash)
        auth_tag_bits=64,              # Wegman-Carter tag per authenticated message
        # ---- Eve ----
        eve_active=False,
        eve_attack="intercept_resend",
        eve_fraction=1.0,
        eve_noiseless=True,            # Eve's own detector is perfect
        timeshift_ratio=None,          # efficiency of the "slow" detector after a time shift
                                       #   (None: use the STEALTH table below)
        trojan_photons=10.0,           # Trojan photons returning to Eve with no isolator
        eve_stealth=True,              # True: Eve hides every signature she can
                                       # False: a naive Eve who doesn't bother
    )
    # How a stealthy vs a naive Eve behaves for the attacks that leave timing traces
    STEALTH = {
        True:  dict(timeshift_ps=60.0, timeshift_ratio=0.5, blind_offset_ps=-20.0,
                    blind_jitter_ps=40.0, resend_jitter_ps=10.0, watchdog_counts=1.0),
        False: dict(timeshift_ps=300.0, timeshift_ratio=0.03, blind_offset_ps=-80.0,
                    blind_jitter_ps=15.0, resend_jitter_ps=40.0, watchdog_counts=40.0),
    }
    ATTACKS = ("intercept_resend", "breidbart", "pns", "blinding", "timeshift", "trojan")

    def __init__(self, seed=None, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        for k, v in {**self.DEFAULTS, **params}.items():
            setattr(self, k, v)
        self.rng = np.random.default_rng(seed)
        self.theta = 0.0
        self._clock = 0                  # global pulse-slot counter (telemetry mode)
        self._last_click = -10 ** 18
        self._cache = {}
        self._backends = None

    # ---- derived quantities --------------------------------------------------
    @property
    def eta_channel(self):
        return db_to_t(self.alpha_db_km * self.distance_km + self.fixed_loss_db)

    @property
    def eta_bob_optics(self):
        return db_to_t(self.bob_loss_db)

    @property
    def eta_bob(self):
        """Mean probability that a photon entering Bob's box is detected."""
        return self.eta_bob_optics * float(np.mean(self.det_eff))

    @property
    def dead_slots(self):
        return int(round(self.dead_time_ns * 1e-9 * self.rep_rate_hz))

    def backends(self):
        if self._backends is None:
            self._backends = {True: make_backend(self.p_channel, self.p_pbs),
                              False: AerSimulator(method="density_matrix")}
        return self._backends

    # ---- Qiskit measurement ---------------------------------------------------
    def _qiskit_table(self, noisy):
        """
        Probability of outcome 1 for every angle on a fine grid, computed by Qiskit Aer
        (density-matrix method, so the probabilities are exact). Built once per noise
        setting and shared; one Qiskit job with all angles bound to a Parameter.
        """
        key = (noisy, float(self.p_channel), float(self.p_pbs))
        if key not in _QISKIT_TABLES:
            backend = self.backends()[noisy]
            grid = np.linspace(0, TWO_PI, TABLE_POINTS + 1)
            th = Parameter("theta")
            qc = QuantumCircuit(1)
            qc.ry(th, 0)
            qc.id(0)                                     # channel noise slot
            qc.save_probabilities(label="p")
            tqc = transpile(qc, backend, optimization_level=0)
            try:
                res = backend.run(tqc, parameter_binds=[{th: list(grid)}]).result()
                p1 = np.array([np.asarray(res.data(i)["p"])[1] for i in range(len(grid))])
            except Exception:                             # older Aer: bind one at a time
                p1 = np.array([np.asarray(backend.run(tqc.assign_parameters({th: g}))
                                          .result().data(0)["p"])[1] for g in grid])
            ro = self.p_pbs if noisy else 0.0             # readout (PBS) error, applied
            _QISKIT_TABLES[key] = (grid, p1 * (1 - ro) + (1 - p1) * ro)   # like Aer does
        return _QISKIT_TABLES[key]

    def _measure(self, angles, noisy=True):
        """
        Measure many photons (outcome 0/1 for each).
        table mode: look up Qiskit's exact outcome probability and sample each photon.
        shots mode: photons with the same angle share one Qiskit run; the outcomes are
        exchangeable, so the counts are scattered over the photons at random.
        """
        out = np.zeros(len(angles), dtype=np.int8)
        if len(angles) == 0:
            return out
        if self.qiskit_mode == "table":
            grid, table = self._qiskit_table(noisy)
            p = np.interp(np.mod(angles, TWO_PI), grid, table)
            return (self.rng.random(len(angles)) < p).astype(np.int8)
        backend = self.backends()[noisy]
        keys = np.round(np.mod(angles, TWO_PI), 6)
        if len(self._cache) > 256:
            self._cache.clear()
        uniq, inverse = np.unique(keys, return_inverse=True)
        for g, u in enumerate(uniq):
            idx = np.where(inverse == g)[0]
            ck = (float(u), noisy)
            if ck not in self._cache:
                self._cache[ck] = transpile(photon_circuit(u), backend,
                                            optimization_level=0)
            counts = backend.run(self._cache[ck], shots=len(idx),
                                 seed_simulator=int(self.rng.integers(0, 2 ** 31 - 1))
                                 ).result().get_counts()
            ones = int(counts.get("1", 0))
            if ones:
                out[self.rng.choice(idx, ones, replace=False)] = 1
        return out

    # ---- Eve's strategies (tuned so Bob's detection rate looks normal) -------
    def _poisson(self, m, nmax):
        n = np.arange(nmax + 1)
        pmf = np.exp(-m) * m ** n / np.array([factorial(int(i)) for i in n])
        pmf[-1] += max(0.0, 1 - pmf.sum())
        return pmf

    def _honest_yield(self, n):
        return 1 - (1 - self.eta_channel * self.eta_bob) ** n

    def _resend_eta(self):
        """Measure-resend Eve: transmission of her lossless line to match Bob's rate."""
        m = self.mu[0]
        target = 1 - exp(-m * self.eta_channel * self.eta_bob)
        return min(1.0, target / ((1 - exp(-m)) * self.eta_bob))

    def _pns_strategy(self):
        m, eb = self.mu[0], self.eta_bob
        pmf = self._poisson(m, 12)
        n = np.arange(13)
        target = 1 - exp(-m * self.eta_channel * eb)
        g_multi = float(np.sum(pmf[2:] * (1 - (1 - eb) ** (n[2:] - 1))))
        if g_multi >= target:
            return target / g_multi, 0.0
        return 1.0, min(1.0, (target - g_multi) / (pmf[1] * self.eta_channel * eb))

    def _ts_ratio(self):
        if self.timeshift_ratio is not None:
            return self.timeshift_ratio
        return self.STEALTH[bool(self.eve_stealth)]["timeshift_ratio"]

    def _p_basis_match(self):
        return 0.5        # blinding Eve picks Z/X 50:50 (keeps Bob's basis statistics normal)

    def _branches(self, n):
        """
        What happens to a pulse that leaves Alice with n photons.
        Returns [(probability, photons on to Bob, transmission to Bob's box, kind)]
        kind: 0 untouched, 1 measure-resend, 2 blinding trigger, 3 PNS (Eve kept one),
              4 time-shifted
        """
        honest = [(1.0, n, self.eta_channel, 0)]
        if not self.eve_active or n == 0:
            return honest
        f, a = self.eve_fraction, self.eve_attack
        rest = [(1 - f, n, self.eta_channel, 0)] if f < 1 else []
        st = self.eve_stealth
        if a in ("intercept_resend", "breidbart"):
            # stealthy: lossless line tuned to keep Bob's rate; naive: re-sends into the fibre
            return rest + [(f, 1, self._resend_eta() if st else self.eta_channel, 1)]
        if a == "pns":
            s, t = self._pns_strategy() if st else (1.0, 0.0)
            if n == 1:
                return rest + [(f * t, 1, self.eta_channel, 0)]
            return rest + [(f * s, n - 1, 1.0, 3)]
        if a == "blinding":
            r = min(1.0, self._honest_yield(n) / self._p_basis_match()) if st else 1.0
            return rest + [(f * r, 1, 1.0, 2)]
        if a == "timeshift":
            c = 2 / (1 + self._ts_ratio()) if st else 1.0  # stealthy Eve makes up the loss
            return rest + [(f, n, min(1.0, self.eta_channel * c), 4)]
        if a == "trojan":
            return honest                                   # passive: never touches photons
        raise ValueError(f"Unknown attack {a!r}")

    # ---- one chunk of pulses ----------------------------------------------------
    def _chunk(self, n_slots, offset):
        rng = self.rng
        eta_max = self.eta_bob_optics * max(self.det_eff)

        # (a) group pulses by intensity class, actual intensity and photon number
        gh_z, gh_w = np.polynomial.hermite_e.hermegauss(5)
        gh_w = gh_w / gh_w.sum()
        nmax = 10
        n_cls = rng.multinomial(n_slots, self.p_mu)
        ev = {k: [] for k in ("cls", "n", "kind", "m")}
        n_intercept = 0
        for k, (mk, Nk) in enumerate(zip(self.mu, n_cls)):
            if Nk == 0 or mk == 0:
                continue
            for z, Nsub in zip(gh_z, rng.multinomial(Nk, gh_w)):
                m_act = max(0.0, mk * (1 + self.intensity_fluct * z))
                for n, Ng in enumerate(rng.multinomial(Nsub, self._poisson(m_act, nmax))):
                    if n == 0 or Ng == 0:
                        continue
                    br = self._branches(n)
                    probs = [b[0] for b in br]
                    split = rng.multinomial(Ng, probs + [max(0.0, 1 - sum(probs))])
                    for (pb, nb, eta, kind), Nb in zip(br, split[:-1]):
                        if kind != 0:
                            n_intercept += Nb
                        if Nb == 0 or nb == 0:
                            continue
                        if kind == 2:                          # bright trigger: always arrives
                            n_ev = Nb
                            m = np.zeros(n_ev, dtype=np.int64)
                        else:
                            pa = eta * eta_max
                            p_ev = 1 - (1 - pa) ** nb
                            n_ev = rng.binomial(Nb, p_ev)
                            if n_ev == 0:
                                continue
                            j = np.arange(1, nb + 1)
                            pmf = np.array([comb(nb, int(i)) for i in j]) * pa ** j * (1 - pa) ** (nb - j)
                            m = rng.choice(j, n_ev, p=pmf / pmf.sum())
                        ev["cls"].append(np.full(n_ev, k))
                        ev["n"].append(np.full(n_ev, n))
                        ev["kind"].append(np.full(n_ev, kind))
                        ev["m"].append(m)
        E = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for k, v in ev.items()}
        if self.eve_active and self.eve_attack == "trojan":
            n_intercept = int(n_slots * self.eve_fraction)
        nE = len(E["cls"])
        E["pos"] = rng.integers(offset, offset + n_slots, nE)
        order = np.argsort(E["pos"], kind="stable")
        E = {k: v[order] for k, v in E.items()}
        keep = np.r_[True, np.diff(E["pos"]) > 0] if nE else np.zeros(0, bool)
        E = {k: v[keep] for k, v in E.items()}
        nE = len(E["pos"])

        # (b) random choices for every event slot
        E.update(self._random_slot_data(nE))
        E["eguess"] = np.full(nE, -1, dtype=np.int8)
        E["econf"] = np.full(nE, 0.5)
        a_ang = E["a_basis"] * (np.pi / 2) + E["a_bit"] * np.pi + E["a_basis"] * self.x_prep_error
        state = a_ang.copy()
        kind = E["kind"]

        # (c) Eve acts
        e_basis = (rng.random(nE) >= self.q_z).astype(np.int8)    # Eve mimics the bias
        if self.eve_active and self.eve_attack == "blinding":
            e_basis = rng.integers(0, 2, nE).astype(np.int8)
        if self.eve_active:
            mr = np.where(kind == 1)[0]
            if len(mr):
                if self.eve_attack == "breidbart":
                    phi = np.full(len(mr), np.pi / 4)
                else:
                    phi = e_basis[mr] * (np.pi / 2)
                o = self._measure(a_ang[mr] - phi, noisy=not self.eve_noiseless)
                state[mr] = phi + o * np.pi
                E["eguess"][mr] = o
                E["econf"][mr] = (COS2_PI8 if self.eve_attack == "breidbart" else
                                  np.where(e_basis[mr] == E["a_basis"][mr], 1.0, 0.5))
            bl = np.where(kind == 2)[0]
            if len(bl):
                o = self._measure(a_ang[bl] - e_basis[bl] * (np.pi / 2), noisy=False)
                E["eguess"][bl] = o
                E["econf"][bl] = np.where(e_basis[bl] == E["a_basis"][bl], 1.0, 0.5)
                noise = self._blind_noise() if self.eve_stealth else 0.0
                trig = o ^ (rng.random(len(bl)) < noise).astype(np.int8)
            pn = np.where(kind == 3)[0]
            E["eguess"][pn] = E["a_bit"][pn]
            E["econf"][pn] = 1.0
            ts = np.where(kind == 4)[0]
            fav = rng.integers(0, 2, nE).astype(np.int8)
            E["eguess"][ts] = fav[ts]
            E["econf"][ts] = 0.5 if self.scramble else 1 / (1 + self._ts_ratio())
            if self.eve_attack == "trojan":
                p_leak = (1 - exp(-self.trojan_photons * db_to_t(self.alice_isolation_db))) \
                         * self.eve_fraction
                lk = np.where(rng.random(nE) < p_leak)[0]
                E["eguess"][lk] = E["a_bit"][lk]
                E["econf"][lk] = 1.0

        # (d) photons reach Bob's detectors (Qiskit measures each photon)
        c0 = np.zeros(nE, dtype=bool)
        c1 = np.zeros(nE, dtype=bool)
        m = E["m"]
        ph = np.repeat(np.arange(nE), m)
        if len(ph):
            ang = state[ph] - E["b_basis"][ph] * (np.pi / 2) + self.theta
            o = self._measure(ang, noisy=True)
            det = o ^ E["s"][ph]
            eff = np.asarray(self.det_eff)[det].astype(float)
            if self.eve_active and self.eve_attack == "timeshift":
                slow = (kind[ph] == 4) & (det != fav[ph])
                eff[slow] *= self._ts_ratio()
            eff[E["check"][ph]] = 0.0
            acc = rng.random(len(ph)) < eff / eta_max
            c0 |= np.bincount(ph[acc & (det == 0)], minlength=nE) > 0
            c1 |= np.bincount(ph[acc & (det == 1)], minlength=nE) > 0
        if self.eve_active and self.eve_attack == "blinding":
            bl = np.where(kind == 2)[0]
            if len(bl):
                hitd = (trig ^ E["s"][bl]).astype(np.int8)
                fire = E["b_basis"][bl] == e_basis[bl]      # wrong basis -> light splits, no click
                c0[bl[fire & (hitd == 0)]] = True
                c1[bl[fire & (hitd == 1)]] = True
        E["c0"], E["c1"] = c0, c1

        # (d2) click times relative to the expected arrival (picoseconds)
        S = self.STEALTH[bool(self.eve_stealth)]
        t = rng.normal(0.0, self.jitter_ps, nE)
        if self.eve_active:
            t[kind == 1] += rng.normal(0.0, S["resend_jitter_ps"], int(np.sum(kind == 1)))
            b2 = kind == 2                    # bright trigger: sharp, early leading edge
            t[b2] = rng.normal(S["blind_offset_ps"], S["blind_jitter_ps"], int(b2.sum()))
            t4 = kind == 4                    # shifted towards the favoured detector's window
            t[t4] += np.where(fav[t4] == 0, -1.0, 1.0) * S["timeshift_ps"]
        E["t_ps"] = t

        # (e) dark counts and afterpulses. A blinded detector has neither: a stealthy Eve
        #     fakes them with extra trigger pulses so the statistics look normal; a naive
        #     Eve doesn't.
        blinded = self.eve_active and self.eve_attack == "blinding"
        if not (blinded and not self.eve_stealth):
            for d in (0, 1):
                nd = rng.binomial(n_slots, self.p_dark)
                E = self._merge_clicks(E, rng.integers(offset, offset + n_slots, nd), d)
            for d in (0, 1):
                src = E["pos"][E[f"c{d}"]]
                src = src[rng.random(len(src)) < self.p_afterpulse]
                ap = src + self.dead_slots + 1 + rng.geometric(1 / self.afterpulse_decay_slots,
                                                               len(src))
                E = self._merge_clicks(E, ap[ap < offset + n_slots], d)

        # (f) dead time: after any click both detectors are blind for dead_slots
        clicked = np.where(E["c0"] | E["c1"])[0]
        acc = self._dead_time(E["pos"][clicked])
        sel = clicked[acc]
        E = {k: v[sel] for k, v in E.items()}

        # (g) Bob's bit: one detector -> its bit; double click -> random bit
        both = E["c0"] & E["c1"]
        det = np.where(E["c1"], 1, 0).astype(np.int8)
        det[both] = rng.integers(0, 2, int(both.sum()))
        E["b_bit"] = det ^ E["s"]
        E["det"], E["double"] = det, both
        return E, n_cls, n_intercept

    def _blind_noise(self):
        """Eve adds artificial errors so the blinded link shows a normal-looking QBER
        (the faked dark counts and afterpulses add the rest)."""
        return honest_expectation(self, 1e9)["e_z"]

    def _random_slot_data(self, n):
        rng = self.rng
        return {
            "a_bit": rng.integers(0, 2, n).astype(np.int8),
            "a_basis": (rng.random(n) >= self.q_z).astype(np.int8),     # 0 = Z, 1 = X
            "b_basis": (rng.random(n) >= self.q_z).astype(np.int8),
            "s": (rng.integers(0, 2, n) if self.scramble else np.zeros(n)).astype(np.int8),
            "check": rng.random(n) < self.check_fraction,
        }

    def _merge_clicks(self, E, positions, d):
        """Add clicks on detector d at given slots (new slots get fresh random data)."""
        if len(positions) == 0:
            return E
        positions = np.unique(positions)
        i = np.searchsorted(E["pos"], positions)
        i_c = np.minimum(i, max(len(E["pos"]) - 1, 0))
        hit = (len(E["pos"]) > 0) & (E["pos"][i_c] == positions) if len(E["pos"]) else \
            np.zeros(len(positions), bool)
        E[f"c{d}"][i_c[hit]] = True
        new = positions[~hit]
        nn = len(new)
        if nn == 0:
            return E
        add = {"pos": new, "cls": self.rng.choice(3, nn, p=self.p_mu),
               "n": np.zeros(nn, int), "kind": np.full(nn, -1), "m": np.zeros(nn, int),
               "eguess": np.full(nn, -1, dtype=np.int8), "econf": np.full(nn, 0.5),
               "c0": np.full(nn, d == 0), "c1": np.full(nn, d == 1),
               "t_ps": self.rng.uniform(-self.gate_ps / 2, self.gate_ps / 2, nn)}
        add.update(self._random_slot_data(nn))
        E = {k: np.concatenate([E[k], add[k]]) for k in E}
        order = np.argsort(E["pos"], kind="stable")
        return {k: v[order] for k, v in E.items()}

    def _dead_time(self, pos):
        """Sequential dead-time filter; only clicks close together need the slow path."""
        acc = np.ones(len(pos), dtype=bool)
        if len(pos) == 0:
            return acc
        d = self.dead_slots
        prev = np.r_[self._last_click, pos[:-1]]
        close = np.where(pos - prev <= d)[0]
        last_acc = self._last_click
        j_prev = -1
        for i in close:
            # last accepted click before i
            if i - 1 >= 0 and acc[i - 1]:
                last = pos[i - 1]
            else:
                j = i - 1
                while j >= 0 and not acc[j]:
                    j -= 1
                last = pos[j] if j >= 0 else last_acc
            if pos[i] - last <= d:
                acc[i] = False
        self._last_click = int(pos[acc][-1]) if acc.any() else self._last_click
        return acc

    # ---- telemetry: what Alice and Bob's monitoring sees in one timestep ------
    TELEMETRY_FEATURES = [
        "qber_z",             # error rate of the key (Z) basis, known after error correction
        "qber_x",             # error rate of the test (X) basis
        "rate_signal",        # detections per 1000 signal pulses
        "rate_decoy",         # detections per 1000 decoy pulses
        "vacuum_clicks",      # clicks in vacuum slots (dark counts + afterpulses)
        "check_clicks",       # clicks in detector-efficiency check slots
        "double_click_frac",  # fraction of clicks where both detectors fired
        "det1_frac",          # fraction of single clicks on detector 1
        "timing_offset_ps",   # mean click time relative to the expected arrival
        "timing_jitter_ps",   # spread of click times
        "watchdog_counts",    # Alice's monitor photodiode (light coming back into Alice)
    ]

    def telemetry_step(self, n_pulses=1_000_000):
        """
        Send n_pulses and return the monitoring features Alice and Bob would log for
        this timestep, plus ground truth that only the simulation knows.
        Polarization drift evolves by one step per call.
        """
        rng = self.rng
        self.theta = (1 - self.drift_pull) * self.theta + rng.normal(0, self.drift_sd)
        E, n_cls, n_int = self._chunk(int(n_pulses), self._clock)
        self._clock += int(n_pulses)

        ok = ~E["check"]
        zz = ok & (E["a_basis"] == 0) & (E["b_basis"] == 0)
        xx = ok & (E["a_basis"] == 1) & (E["b_basis"] == 1)
        err = E["a_bit"] != E["b_bit"]
        single = ok & ~E["double"]
        t = E["t_ps"][ok]

        light = self.watchdog_bg
        if self.eve_active and self.eve_attack == "trojan":
            light += self.eve_fraction * self.STEALTH[bool(self.eve_stealth)]["watchdog_counts"]

        def frac(num, den):
            return float(num) / den if den else np.nan

        feats = {
            "qber_z": frac(np.sum(err & zz), np.sum(zz)),
            "qber_x": frac(np.sum(err & xx), np.sum(xx)),
            "rate_signal": 1e3 * frac(np.sum(ok & (E["cls"] == 0)), n_cls[0]),
            "rate_decoy": 1e3 * frac(np.sum(ok & (E["cls"] == 1)), n_cls[1]),
            "vacuum_clicks": int(np.sum(ok & (E["cls"] == 2))),
            "check_clicks": int(np.sum(E["check"])),
            "double_click_frac": frac(np.sum(ok & E["double"]), np.sum(ok)),
            "det1_frac": frac(np.sum(single & (E["det"] == 1)), np.sum(single)),
            "timing_offset_ps": float(np.mean(t)) if len(t) else np.nan,
            "timing_jitter_ps": float(np.std(t)) if len(t) else np.nan,
            "watchdog_counts": int(rng.poisson(light)),
        }
        truth = {
            "eve_info": float(np.mean(1 - h2(E["econf"][zz]))) if zz.any() else 0.0,
            "n_intercepted": int(n_int),
            "misalignment": float(np.sin(self.theta / 2) ** 2),
        }
        return feats, truth

    # ---- a full QKD session ---------------------------------------------------
    def run_session(self, n_pulses, postprocess=True):
        """Send n_pulses, then (optionally) run the full post-processing."""
        rng = self.rng
        chunks = []
        a_key, b_key, e_guess, e_conf = [], [], [], []
        nZ = np.zeros(3)
        nX = np.zeros(3)
        mX = np.zeros(3)
        n_sent = np.zeros(3)
        n_check = n_check_clicks = n_clicks = n_intercept = 0
        done = 0
        while done < n_pulses:
            n_slots = int(min(self.chunk_pulses, n_pulses - done))
            self.theta = (1 - self.drift_pull) * self.theta + rng.normal(0, self.drift_sd)
            E, n_cls, n_int = self._chunk(n_slots, done)
            n_intercept += n_int
            n_sent += n_cls
            n_clicks += len(E["pos"])
            chk = E["check"]
            n_check_clicks += int(chk.sum())
            n_check += rng.binomial(n_slots, self.check_fraction)
            ok = ~chk
            zz = ok & (E["a_basis"] == 0) & (E["b_basis"] == 0)
            xx = ok & (E["a_basis"] == 1) & (E["b_basis"] == 1)
            err = E["a_bit"] != E["b_bit"]
            for k in range(3):
                nZ[k] += np.sum(zz & (E["cls"] == k))
                nX[k] += np.sum(xx & (E["cls"] == k))
                mX[k] += np.sum(xx & err & (E["cls"] == k))
            a_key.append(E["a_bit"][zz])
            b_key.append(E["b_bit"][zz])
            e_guess.append(E["eguess"][zz])
            e_conf.append(E["econf"][zz])
            chunks.append({"t_s": (done + n_slots / 2) / self.rep_rate_hz,
                           "theta": self.theta,
                           "x_qber": err[xx].mean() if xx.any() else np.nan,
                           "z_qber": err[zz].mean() if zz.any() else np.nan,
                           "n_xx": int(xx.sum())})
            done += n_slots

        a = np.concatenate(a_key)
        b = np.concatenate(b_key)
        eg = np.concatenate(e_guess)
        ec = np.concatenate(e_conf)
        nZt, nXt = nZ.sum(), nX.sum()
        res = {"n_pulses": n_pulses, "chunks": chunks, "nZ": nZ, "nX": nX, "mX": mX,
               "n_sent": n_sent, "n_intercept": n_intercept,
               "x_qber": mX.sum() / nXt if nXt else np.nan,
               "z_qber_true": float(np.mean(a != b)) if len(a) else np.nan,
               "n_clicks": n_clicks, "n_check": n_check, "n_check_clicks": n_check_clicks}

        # detection-rate monitor: measured gain per intensity vs an honest link
        exp_h = honest_expectation(self, n_pulses)
        res["gain_ratio"] = (nZ + nX) / np.maximum(exp_h["nZ"] + exp_h["nX"], 1e-9)

        # detector check slots: clicks there should only be dark counts / afterpulses
        expect = n_check * (2 * self.p_dark + self.p_afterpulse * n_clicks / n_pulses)
        res["check_expected"] = expect
        res["check_alarm"] = n_check_clicks > expect + 6 * sqrt(expect) + 10

        # Eve's knowledge of the raw key (ground truth only a simulation can see)
        guess = np.where(eg >= 0, eg, rng.integers(0, 2, len(eg)))
        res["eve_raw_correct"] = float(np.mean(guess == a)) if len(a) else np.nan
        res["eve_info_bits"] = float(np.sum(1 - h2(ec)))
        res["eve_info_frac"] = res["eve_info_bits"] / len(a) if len(a) else np.nan
        if not postprocess:
            return res

        # ---------------- post-processing ----------------
        # 1. error correction (Cascade), seeded with the X-basis error estimate
        qb_est = res["x_qber"] if np.isfinite(res["x_qber"]) else 0.05
        b_corr, leaked, rounds = cascade(a, b, qb_est, rng, self.cascade_passes)
        res["ec_leaked"] = leaked
        res["ec_rounds"] = rounds
        res["z_qber"] = float(np.mean(a != b))       # known exactly after EC
        res["f_ec"] = leaked / max(1e-9, len(a) * float(h2(res["z_qber"])))
        # 2. verification: compare a short universal hash of both strings
        ver_bits = int(ceil(log2(2 / self.eps_cor)))
        ha, hb = toeplitz_hash([a, b_corr], ver_bits, rng)
        res["verified"] = bool(np.array_equal(ha, hb))
        res["residual_errors"] = int(np.sum(a != b_corr))
        # 3. key length from the finite-key bound
        fk = finite_key(nZ, nX, mX, leaked, self.mu, self.p_mu, self.eps_sec, self.eps_cor)
        res["fk"] = fk
        ell = int(fk["ell"]) if res["verified"] and not res["check_alarm"] else 0
        # 4. authentication of every classical message costs key
        n_msgs = 2 * rounds + 8      # sifting, basis info, parameter estimation, EC, PA seed
        res["auth_cost"] = n_msgs * self.auth_tag_bits
        res["ell"] = ell
        res["net_key"] = max(0, ell - res["auth_cost"])
        # 5. privacy amplification (Toeplitz hashing) and ground-truth checks
        if ell > 0:
            fa, fb, fe = toeplitz_hash([a, b_corr, guess], ell, rng)
            res["keys_match"] = bool(np.array_equal(fa, fb))
            res["eve_final_agree"] = float(np.mean(fe == fa))
            removed = len(a) - ell
            leftover = res["eve_info_bits"] + leaked + ver_bits - removed
            res["eve_final_info_frac"] = float(np.clip(leftover / ell, 0, 1))
        else:
            res["keys_match"] = None
            res["eve_final_agree"] = np.nan
            res["eve_final_info_frac"] = np.nan
        res["key_rate_bps"] = res["net_key"] / n_pulses * self.rep_rate_hz
        return res


# ============================================================================
# Theory for an honest link (expected counts -> finite-key length)
# ============================================================================
def honest_expectation(link, n_pulses, distance_km=None):
    L = link.distance_km if distance_km is None else distance_km
    eta_ch = db_to_t(link.alpha_db_km * L + link.fixed_loss_db)
    Y0 = 1 - (1 - link.p_dark) ** 2                              # either detector dark
    sd = link.drift_sd / sqrt(max(1e-12, 1 - (1 - link.drift_pull) ** 2))
    mis = (1 - exp(-sd ** 2 / 2)) / 2                              # mean sin^2(theta/2)
    e_z = xor_prob(xor_prob(link.p_channel / 2, link.p_pbs), mis)
    e_x = xor_prob(e_z, np.sin(link.x_prep_error / 2) ** 2)
    mu = np.asarray(link.mu, float)
    p = np.asarray(link.p_mu, float)
    # dead-time saturation (iterate once)
    eta = eta_ch * link.eta_bob
    R = float(np.sum(p * (1 - (1 - Y0) * np.exp(-eta * mu))))
    eta *= 1 / (1 + R * link.dead_slots)
    sig = 1 - np.exp(-eta * mu)
    Q = 1 - (1 - Y0) * np.exp(-eta * mu)
    R = float(np.sum(p * Q))
    ap = link.p_afterpulse * R
    Qt = Q + ap
    errZ = 0.5 * Y0 + e_z * sig + 0.5 * ap
    errX = 0.5 * Y0 + e_x * sig + 0.5 * ap
    usable = n_pulses * (1 - link.check_fraction)
    nZ = usable * p * link.q_z ** 2 * Qt
    nX = usable * p * (1 - link.q_z) ** 2 * Qt
    mX = usable * p * (1 - link.q_z) ** 2 * errX
    E_Z = float(np.sum(p * errZ) / np.sum(p * Qt))
    return {"nZ": nZ, "nX": nX, "mX": mX, "E_Z": E_Z, "E_X": float(mX.sum() / nX.sum()),
            "eta": eta, "Y0": Y0, "e_z": e_z, "eta_ch": eta_ch}


def theory_key(link, n_pulses, distance_km=None, asymptotic=False):
    """Expected secret bits per second for an honest link."""
    ex = honest_expectation(link, n_pulses, distance_km)
    lam = link.f_ec_theory * ex["nZ"].sum() * float(h2(ex["E_Z"]))
    fk = finite_key(ex["nZ"], ex["nX"], ex["mX"], lam, link.mu, link.p_mu,
                    link.eps_sec, link.eps_cor, asymptotic=asymptotic)
    auth = 0 if asymptotic else 2 * link.auth_tag_bits * 40
    bits = max(0.0, fk["ell"] - auth)
    return {**ex, **fk, "bits": bits, "bps": bits / n_pulses * link.rep_rate_hz,
            "per_pulse": bits / n_pulses}


# ============================================================================
# Other protocols (asymptotic, same detectors) for comparison
# ============================================================================
def protocol_rates(link, distances):
    """Secret bits per pulse for several protocols, each with its best intensity."""
    out = {k: [] for k in ("BB84, decoy states (weak laser)",
                           "BB84, no decoy states (weak laser)",
                           "BB84, perfect single-photon source",
                           "Six-state, perfect single-photon source",
                           "PLOB bound (no protocol can beat this)")}
    f = link.f_ec_theory
    q = link.q_z ** 2
    mus = np.linspace(0.02, 1.0, 50)
    for L in distances:
        ex = honest_expectation(link, 1e9, L)
        eta, Y0, ed = ex["eta"], ex["Y0"], ex["e_z"]
        # decoy (infinite decoys, best mu)
        best_d = best_n = 0.0
        for m in mus:
            Qm = 1 - (1 - Y0) * exp(-eta * m)
            Em = (0.5 * Y0 + ed * (1 - exp(-eta * m))) / Qm
            Y1 = Y0 + eta - Y0 * eta
            e1 = (0.5 * Y0 + ed * eta) / Y1
            Q1 = Y1 * m * exp(-m)
            best_d = max(best_d, q * (Q1 * (1 - h2(e1)) - f * Qm * h2(Em)))
            pm = 1 - exp(-m) * (1 + m)
            Q1l = Qm - pm
            if Q1l > 0:
                e1u = min(0.5, Em * Qm / Q1l)
                best_n = max(best_n, q * (Q1l * (1 - h2(e1u)) - f * Qm * h2(Em)))
        out["BB84, decoy states (weak laser)"].append(best_d)
        out["BB84, no decoy states (weak laser)"].append(best_n)
        Qs = Y0 + eta - Y0 * eta
        Es = (0.5 * Y0 + ed * eta) / Qs
        out["BB84, perfect single-photon source"].append(
            max(0.0, q * Qs * (1 - h2(Es) - f * h2(Es))))
        e = Es
        if e < 2 / 3:
            S = -(1 - 1.5 * e) * np.log2(max(1 - 1.5 * e, 1e-15)) \
                - 1.5 * e * np.log2(max(e / 2, 1e-15))
        else:
            S = 2.0
        out["Six-state, perfect single-photon source"].append(
            max(0.0, q * Qs * (1 - (S - h2(e)) - f * h2(e))))
        out["PLOB bound (no protocol can beat this)"].append(-np.log2(1 - min(ex["eta_ch"], 1 - 1e-12)))
    return {k: np.asarray(v, float) for k, v in out.items()}


PRESETS = {
    "InGaAs avalanche photodiodes": dict(det_eff=(0.20, 0.18), p_dark=2e-6,
                                         p_afterpulse=0.04, dead_time_ns=10.0),
    "Superconducting nanowire (SNSPD)": dict(det_eff=(0.85, 0.82), p_dark=1e-8,
                                             p_afterpulse=0.0, dead_time_ns=30.0),
    "SNSPD + ultra-low-loss fibre": dict(det_eff=(0.85, 0.82), p_dark=1e-8,
                                         p_afterpulse=0.0, dead_time_ns=30.0,
                                         alpha_db_km=0.16),
    "Cheap room-temperature APDs": dict(det_eff=(0.10, 0.07), p_dark=1e-5,
                                        p_afterpulse=0.08, dead_time_ns=50.0),
}
