"""
Full BB84 report: every attack, defences, finite-key analysis, distance, hardware and
protocol comparisons. Saves one PNG per graph and bb84_report.pdf.
Run:  python bb84_report.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb84 import *   # the BB84 engine (bb84.py)
# ============================================================================
# ============================================================================
#                            EXPERIMENTS AND REPORT
# ============================================================================
# ============================================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# ---------------- settings (lower the pulse counts for a faster run) --------
SEED = 7
SESSION_PULSES = 1_000_000_000        # one session = 1 s of pulses at 1 GHz
SWEEP_PULSES = 200_000_000            # attack-strength sweep (no post-processing)
SWEEP_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DRIFT_RUN_CHUNKS = 120                # live-monitor run: 120 x 0.01 s
DISTANCES_SIM = [25, 40, 55, 70, 85]
BLOCK_SIZES_SIM = [1e8, 3e8, 1e9, 3e9]

ALARM_QBER = 0.05          # error-rate alarm
GAIN_TOL = 0.15            # detection-rate alarm: more than 15% off the expected rate

# scenario: label, short label, attack (None = no Eve), extra link settings
SCENARIOS = [
    ("No eavesdropper",                       "No Eve",              None, {}),
    ("Intercept-resend on 20% of pulses",     "Int-resend 20%",      "intercept_resend", dict(eve_fraction=0.2)),
    ("Intercept-resend on every pulse",       "Int-resend 100%",     "intercept_resend", {}),
    ("Breidbart attack",                      "Breidbart",           "breidbart", {}),
    ("Photon-number splitting",               "PNS",                 "pns", {}),
    ("Detector blinding (check slots on)",    "Blinding",            "blinding", {}),
    ("Detector blinding, NO check slots",     "Blinding, no check",  "blinding", dict(check_fraction=0.0)),
    ("Time-shift, NO detector scrambling",    "Time-shift",          "timeshift", dict(scramble=False, timeshift_ratio=0.03)),
    ("Time-shift (detector scrambling on)",   "Time-shift, scrambled", "timeshift", dict(timeshift_ratio=0.03)),
    ("Trojan horse, NO isolator",             "Trojan, no isolator", "trojan", dict(alice_isolation_db=0.0)),
    ("Trojan horse (60 dB isolator)",         "Trojan, isolator",    "trojan", {}),
]
ATTACK_COLOURS = {None: "#2a78d6", "intercept_resend": "#eb6834", "breidbart": "#1baf7a",
                  "pns": "#eda100", "blinding": "#e87ba4", "timeshift": "#008300",
                  "trojan": "#6250d6"}
ATTACK_NAMES = {"intercept_resend": "Intercept-resend", "breidbart": "Breidbart",
                "pns": "Photon-number splitting", "blinding": "Detector blinding",
                "timeshift": "Time-shift", "trojan": "Trojan horse"}

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT, GOOD, PASS_CELL = "#d03b3b", "#0ca30c", "#f0efec"


def line(char="=", width=92):
    print(char * width)


def fmt(x, unit=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.2f} {suf}{unit}"
    return f"{x:.0f} {unit}".strip()


def checks_for(r):
    return {
        "Error rate (QBER)": bool(max(r["x_qber"], r["z_qber"]) >= ALARM_QBER),
        "Detection rate": bool(np.any(np.abs(r["gain_ratio"] - 1) > GAIN_TOL)),
        "Decoy-state /\nfinite-key bound": bool(r["fk"]["ell_raw"] <= 0),
        "Detector\ncheck slots": bool(r["check_alarm"]),
        "Error-correction\nverification": bool(not r["verified"]),
    }


def outcome(r):
    if r["ell"] <= 0:
        return "ABORTED"
    if np.isfinite(r["eve_final_info_frac"]) and r["eve_final_info_frac"] > 0.01:
        return "BREACH"
    return "SECURE KEY"


# ============================================================================
# 1. Scenarios
# ============================================================================
t_start = time.time()
base = BB84Link(seed=SEED)
th_base = theory_key(base, SESSION_PULSES)
line()
print("REALISTIC BB84 QUANTUM KEY DISTRIBUTION SIMULATION")
line()
print(f"Protocol : efficient BB84, Z basis {base.q_z:.0%} of the time, 3 intensities "
      f"{base.mu} sent {tuple(f'{p:.0%}' for p in base.p_mu)}")
print(f"Source   : {base.rep_rate_hz / 1e9:g} GHz, intensity noise {base.intensity_fluct:.0%}, "
      f"X-state prep error {base.x_prep_error} rad, isolator {base.alice_isolation_db:g} dB")
print(f"Channel  : {base.distance_km:g} km x {base.alpha_db_km} dB/km "
      f"(+{base.fixed_loss_db:g} dB) -> {base.eta_channel:.1%} transmission, "
      f"depolarizing {base.p_channel:.0%}, drift on")
print(f"Bob      : {base.bob_loss_db:g} dB optics, detectors {base.det_eff[0]:.0%}/"
      f"{base.det_eff[1]:.0%} efficient, dark {base.p_dark:g}/slot, afterpulse "
      f"{base.p_afterpulse:.0%}, dead time {base.dead_time_ns:g} ns")
print(f"Defences : detector scrambling {'on' if base.scramble else 'off'}, "
      f"{base.check_fraction:.1%} efficiency-check slots, decoy states, finite-key bounds")
print(f"Security : eps_sec = {base.eps_sec:g}, eps_cor = {base.eps_cor:g}, "
      f"{base.auth_tag_bits}-bit authentication tags")
print(f"Session  : {SESSION_PULSES:.0e} pulses ({SESSION_PULSES / base.rep_rate_hz:g} s)")
print(f"Theory for an honest link: QBER {th_base['E_Z']:.2%}, "
      f"secret key {fmt(th_base['bps'], 'bit/s')}")
print()

results = []
for label, short, attack, extra in SCENARIOS:
    t0 = time.time()
    kw = dict(extra)
    if attack:
        kw.update(eve_active=True, eve_attack=attack)
    link = BB84Link(seed=SEED, **kw)
    print(f"Running: {label} ...", end=" ", flush=True)
    r = link.run_session(SESSION_PULSES)
    r.update(label=label, short=short, attack=attack, colour=ATTACK_COLOURS[attack],
             countermeasure=short in ("Blinding", "Time-shift, scrambled", "Trojan, isolator"))
    results.append(r)
    print(f"{time.time() - t0:.0f} s")

# ============================================================================
# 2. Printed report
# ============================================================================
print()
line()
print("RESULTS BY SCENARIO (one session each)")
line()
for r in results:
    fk = r["fk"]
    ch = checks_for(r)
    fired = [k.replace("\n", " ") for k, v in ch.items() if v]
    print(f"\n{r['label']}")
    line("-")
    print(f"  Error rate (QBER): key basis Z {r['z_qber']:6.2%}   test basis X {r['x_qber']:6.2%}")
    print(f"  Detection rate vs honest link: signal {r['gain_ratio'][0]:4.0%}, "
          f"decoy {r['gain_ratio'][1]:4.0%}, vacuum {r['gain_ratio'][2]:4.0%}")
    print(f"  Clicks in detector-check slots: {r['n_check_clicks']} "
          f"(expected ~{r['check_expected']:.0f})")
    print(f"  Sifted key bits: {fmt(fk['nZ_total'])}   single-photon bits (lower bound): "
          f"{fmt(fk['sZ1'])}   phase error (upper bound): {fk['phi']:.1%}")
    print(f"  Cascade: leaked {fmt(r['ec_leaked'])} bits in {r['ec_rounds']} rounds "
          f"(efficiency f = {r['f_ec']:.2f}), verification {'passed' if r['verified'] else 'FAILED'}")
    print(f"  Final key: {fmt(r['ell'])} bits - {fmt(r['auth_cost'])} authentication = "
          f"{fmt(r['net_key'])} bits  ->  {fmt(r['key_rate_bps'], 'bit/s')}")
    if r["keys_match"] is not None:
        print(f"  Alice's and Bob's final keys identical: {'yes' if r['keys_match'] else 'NO'}")
    print(f"  Eve's correct guesses on raw key: {r['eve_raw_correct']:.1%}   "
          f"(information: {r['eve_info_frac']:.0%} of a bit per bit)")
    if np.isfinite(r["eve_final_info_frac"]):
        print(f"  Eve's information left in the FINAL key: {r['eve_final_info_frac']:.1%}"
              f"  (0% = privacy amplification removed everything she knew)")
        print(f"  A simple bit-by-bit guess by Eve matches {r['eve_final_agree']:.1%} of final bits")
    print(f"  Alarms raised: {', '.join(fired) if fired else 'none'}")
    print(f"  OUTCOME: {outcome(r)}")

print()
line()
print("COMPACT TABLE")
line()
table = pd.DataFrame([{
    "Scenario": r["short"],
    "QBER Z": f"{r['z_qber']:.1%}", "QBER X": f"{r['x_qber']:.1%}",
    "Key rate": fmt(r["key_rate_bps"], "b/s"),
    "Eve raw": f"{r['eve_info_frac']:.0%}",
    "Eve final": ("-" if not np.isfinite(r["eve_final_info_frac"])
                  else f"{r['eve_final_info_frac']:.0%}"),
    "Outcome": outcome(r)} for r in results]).set_index("Scenario")
print(table.to_string())
print("\n'Eve raw' = her information on the raw key; 'Eve final' = on the final secret key.")

# ============================================================================
# 3. Attack-strength sweep (no post-processing needed)
# ============================================================================
print("\nSweeping attack strength for each attack ...", flush=True)
sweep_rows = []
for attack in ATTACK_NAMES:
    for f in SWEEP_FRACTIONS:
        kw = dict(eve_active=f > 0, eve_attack=attack, eve_fraction=f)
        if attack == "timeshift":
            kw["scramble"] = False
            kw["timeshift_ratio"] = 0.03
        if attack == "trojan":
            kw["alice_isolation_db"] = 0.0
        if attack == "blinding":
            kw["check_fraction"] = 0.0
        r = BB84Link(seed=SEED + 1, **kw).run_session(SWEEP_PULSES, postprocess=False)
        sweep_rows.append({"attack": attack, "fraction": f, "z_qber": r["z_qber_true"],
                           "x_qber": r["x_qber"], "eve_correct": r["eve_raw_correct"],
                           "eve_info": r["eve_info_frac"]})
sweep = pd.DataFrame(sweep_rows)
for attack, d in sweep.groupby("attack", sort=False):
    e = d.iloc[-1]
    print(f"  {ATTACK_NAMES[attack]:<24} at 100%: QBER Z {e.z_qber:5.1%}, X {e.x_qber:5.1%}, "
          f"Eve guesses {e.eve_correct:5.1%} of raw key")

# ============================================================================
# 4. Live monitor with polarization drift
# ============================================================================
print("\nLive-monitor run (polarization drift, no Eve) ...", flush=True)
mon = BB84Link(seed=SEED + 2, chunk_pulses=10_000_000)
mon_r = mon.run_session(DRIFT_RUN_CHUNKS * 10_000_000, postprocess=False)
mon_df = pd.DataFrame(mon_r["chunks"])

# ============================================================================
# 5. Distance and session-length sweeps (no Eve)
# ============================================================================
print("Sweeping fibre length (no Eve) ...", flush=True)
dist_rows = []
for L in DISTANCES_SIM:
    r = BB84Link(seed=SEED + 3, distance_km=L).run_session(SESSION_PULSES)
    dist_rows.append({"km": L, "qber": r["z_qber"], "bps": r["key_rate_bps"]})
    print(f"  {L:4d} km: QBER {r['z_qber']:5.2%}, key {fmt(r['key_rate_bps'], 'bit/s')}")
dist = pd.DataFrame(dist_rows)

print("Sweeping session length at 50 km (finite-key effect) ...", flush=True)
size_rows = []
for N in BLOCK_SIZES_SIM:
    r = BB84Link(seed=SEED + 4).run_session(int(N))
    size_rows.append({"N": N, "bps": r["key_rate_bps"]})
    print(f"  {N:8.0e} pulses ({N / base.rep_rate_hz:6.2f} s): key {fmt(r['key_rate_bps'], 'bit/s')}")
sizes = pd.DataFrame(size_rows)

km = np.linspace(0, 260, 131)
th_lines = {}
for label, N, asym in [("1 s sessions (1e9 pulses)", 1e9, False),
                       ("10 s sessions", 1e10, False),
                       ("1000 s sessions", 1e12, False),
                       ("Infinitely long sessions", 1e12, True)]:
    th_lines[label] = np.array([theory_key(base, N, L, asymptotic=asym)["bps"] for L in km])
qber_th = np.array([honest_expectation(base, 1e9, L)["E_Z"] for L in km])
max_km = {k: (km[v > 0].max() if (v > 0).any() else 0) for k, v in th_lines.items()}
print("  Theory, maximum distance: " + ", ".join(f"{k}: {v:.0f} km" for k, v in max_km.items()))

Ns = np.logspace(7, 13, 61)
size_th = np.array([theory_key(base, N)["bps"] for N in Ns])
size_asym = theory_key(base, 1e12, asymptotic=True)["bps"]
n_min = Ns[size_th > 0].min() if (size_th > 0).any() else np.nan
print(f"  Theory: at least {n_min:.1e} pulses ({n_min / base.rep_rate_hz:.2g} s) are needed "
      f"for any key at {base.distance_km:g} km")

print("Hardware presets and protocol comparison (theory) ...", flush=True)
km2 = np.linspace(0, 400, 201)
preset_lines = {}
for name, kw in PRESETS.items():
    pl = BB84Link(**kw)
    preset_lines[name] = np.array([theory_key(pl, 1e11, L)["bps"] for L in km2])
    v = preset_lines[name]
    print(f"  {name:<34} 50 km: {fmt(np.interp(50, km2, v), 'bit/s'):>14}   "
          f"max distance ~{km2[v > 0].max() if (v > 0).any() else 0:.0f} km")
proto = protocol_rates(base, km2)
proto = {k: v * base.rep_rate_hz for k, v in proto.items()}
for k, v in proto.items():
    if not k.startswith("PLOB"):
        print(f"  {k:<42} max distance ~{km2[v > 0].max() if (v > 0).any() else 0:.0f} km")
print(f"\nTotal run time: {time.time() - t_start:.0f} s")

# ============================================================================
# 6. Graphs - one per page
# ============================================================================
plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titlesize": 15, "axes.titlelocation": "left", "axes.titleweight": "bold",
})
pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
pct1 = plt.FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.1 else f"{v:.0%}")
bps_fmt = plt.FuncFormatter(lambda v, _: fmt(v, "bit/s") if v > 0 else "0")
figures = []


def new_page(title, subtitle, filename, size=(13, 7.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, pad=32)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=11, color=INK_2)
    figures.append((fig, filename))
    return fig, ax


def alarm_line(ax, x_text, ha="left", vertical=False):
    if vertical:
        ax.axvline(ALARM_QBER, color=CRIT, lw=1.5, ls="--", zorder=1)
        ax.text(ALARM_QBER + 0.003, 0.99, f"QBER alarm {ALARM_QBER:.0%}", rotation=90,
                va="top", color=INK_2, fontsize=10, transform=ax.get_xaxis_transform())
    else:
        ax.axhline(ALARM_QBER, color=CRIT, lw=1.5, ls="--", zorder=1)
        ax.text(x_text, ALARM_QBER, f" QBER alarm {ALARM_QBER:.0%}", color=INK_2,
                fontsize=10, ha=ha, va="bottom")


def short_labels(ax, rows):
    ax.set_yticks(range(len(rows)), [r["short"] for r in rows])
    ax.invert_yaxis()


# ---- Page 1: live error rate per scenario -----------------------------------
fig, ax = new_page(
    "1. Live error rate, as Alice and Bob see it every 0.1 s",
    "Measured on the X-basis test bits. Only the measure-and-resend attacks show up here "
    "- the hacking attacks look perfectly normal.",
    "bb84_01_live_qber.png", size=(16, 7.5))
x0 = 0
for r in results:
    ch = pd.DataFrame(r["chunks"])
    xs = x0 + np.arange(len(ch))
    mk = "o" if not r["countermeasure"] else "D"
    ax.plot(xs, ch.x_qber, mk, ms=7, color=r["colour"], mec=SURFACE, mew=1.2, zorder=3)
    ax.hlines(r["x_qber"], xs[0], xs[-1], color=r["colour"], lw=2, zorder=2)
    ax.text(xs[0], 0.47, r["short"], rotation=40, ha="left", va="bottom",
            fontsize=10, color=INK, rotation_mode="anchor")
    x0 += len(ch) + 2
    ax.axvline(x0 - 1.5, color=GRID, lw=1)
alarm_line(ax, 0)
ax.set(xlabel="Time (each point = 0.1 s of pulses; scenarios run one after another)",
       ylabel="Error rate on X-basis test bits", ylim=(0, 0.66), xlim=(-1, x0 - 1))
ax.set_yticks(np.arange(0, 0.46, 0.05))
ax.set_xticks([])
ax.yaxis.set_major_formatter(pct)
ax.grid(axis="y", color=GRID, lw=0.8)

# ---- Page 2: which defence catches which attack ------------------------------
fig, ax = new_page(
    "2. Which defence catches which attack",
    "Red = that check raised an alarm in this run. The last column is the ground truth "
    "only a simulation can see.",
    "bb84_02_detection_matrix.png", size=(15, 8.5))
cols = list(checks_for(results[0]).keys()) + ["Outcome"]
for i, r in enumerate(results):
    ch = checks_for(r)
    for j, c in enumerate(cols):
        if c == "Outcome":
            o = outcome(r)
            fc = {"ABORTED": "#fdebd0", "BREACH": CRIT, "SECURE KEY": "#d7f0d7"}[o]
            tc = "white" if o == "BREACH" else INK
            txt = o if o != "BREACH" else f"BREACH: Eve knows {r['eve_final_info_frac']:.0%}"
        else:
            fc = CRIT if ch[c] else PASS_CELL
            tc = "white" if ch[c] else MUTED
            txt = "ALARM" if ch[c] else "pass"
        w = 1.6 if c == "Outcome" else 1.0
        xx = j if c != "Outcome" else j
        ax.add_patch(Rectangle((xx + 0.04, i + 0.06), w - 0.08, 0.88, color=fc, lw=0))
        ax.text(xx + w / 2, i + 0.5, txt, ha="center", va="center", color=tc,
                fontsize=10, fontweight="bold" if c == "Outcome" or ch.get(c) else None)
ax.set_xlim(0, len(cols) + 0.6)
ax.set_ylim(len(results), -1.1)
for j, c in enumerate(cols):
    ax.text(j + (0.8 if c == "Outcome" else 0.5), -0.15, c, ha="center", va="bottom",
            fontsize=10, color=INK_2, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([i + 0.5 for i in range(len(results))], [r["short"] for r in results])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

# ---- Page 3: secret key rate per scenario ----------------------------------
fig, ax = new_page(
    "3. Secret key delivered in each scenario",
    "Aborted sessions give no key. A key from a breached session looks fine to Alice and "
    "Bob - but Eve knows it.",
    "bb84_03_key_rate.png")
y = np.arange(len(results))
vals = [r["key_rate_bps"] for r in results]
ax.barh(y, vals, color=[r["colour"] for r in results], height=0.62,
        hatch=[("//" if r["countermeasure"] else "") for r in results], edgecolor=SURFACE)
for yi, r in zip(y, results):
    o = outcome(r)
    txt = (fmt(r["key_rate_bps"], "bit/s") if o == "SECURE KEY" else
           ("aborted - no key" if o == "ABORTED" else
            f"{fmt(r['key_rate_bps'], 'bit/s')} - NOT secret"))
    ax.text(r["key_rate_bps"] + max(vals) * 0.01, yi, txt, va="center", fontsize=10,
            color=CRIT if o == "BREACH" else INK_2,
            fontweight="bold" if o == "BREACH" else None)
short_labels(ax, results)
ax.legend(handles=[Patch(facecolor=MUTED, hatch="//", edgecolor=SURFACE,
                         label="Same attack with the defence switched on")],
          frameon=False, loc="lower right")
ax.set(xlabel="Secret key rate after all post-processing", xlim=(0, max(vals) * 1.35))
ax.xaxis.set_major_formatter(bps_fmt)
ax.grid(axis="x", color=GRID, lw=0.8)

# ---- Page 4: Eve's knowledge, raw vs final ----------------------------------
fig, ax = new_page(
    "4. How much Eve knows: raw key vs final key",
    "Privacy amplification wipes out what the security proof accounts for. "
    "Attacks the proof does not model can survive it.",
    "bb84_04_eve_knowledge.png")
raw = [r["eve_info_frac"] for r in results]
fin = [r["eve_final_info_frac"] for r in results]
ax.barh(y - 0.2, raw, 0.38, color=[r["colour"] for r in results], alpha=0.35,
        edgecolor=[r["colour"] for r in results])
ax.barh(y + 0.2, [0 if not np.isfinite(v) else v for v in fin], 0.38,
        color=[r["colour"] for r in results])
for yi, v, rv in zip(y, fin, raw):
    ax.text(rv + 0.01, yi - 0.2, f"{rv:.0%}", va="center", fontsize=9, color=INK_2)
    ax.text((0 if not np.isfinite(v) else v) + 0.01, yi + 0.2,
            "no key made" if not np.isfinite(v) else f"{v:.0%}", va="center", fontsize=9,
            color=CRIT if np.isfinite(v) and v > 0.01 else INK_2,
            fontweight="bold" if np.isfinite(v) and v > 0.01 else None)
short_labels(ax, results)
ax.set(xlabel="Eve's information (bits per key bit)", xlim=(0, 1.15))
ax.xaxis.set_major_formatter(pct)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.legend(handles=[Patch(facecolor=MUTED, alpha=0.35, edgecolor=MUTED, label="Raw key"),
                   Patch(facecolor=MUTED, label="Final secret key")],
          frameon=False, loc="lower right")

# ---- Page 5: Eve's trade-off ------------------------------------------------
fig, ax = new_page(
    "5. What Eve learns vs the errors she causes",
    "Each line runs from 0% to 100% of pulses attacked. Quantum attacks must cause errors; "
    "hacking attacks dodge the physics.",
    "bb84_05_eve_tradeoff.png")
for attack, d in sweep.groupby("attack", sort=False):
    ax.plot(d.z_qber, d.eve_correct, "o-", color=ATTACK_COLOURS[attack], lw=2, ms=8,
            mec=SURFACE, mew=1.5, label=ATTACK_NAMES[attack], zorder=3)
nb = sweep[sweep.fraction == 0]
ax.plot(nb.z_qber.mean(), nb.eve_correct.mean(), "o", ms=12, color=ATTACK_COLOURS[None],
        mec=SURFACE, mew=1.5, zorder=4, label="No Eve")
alarm_line(ax, 0, vertical=True)
ax.set(xlabel="Error rate in the key (Z basis)", ylabel="Eve's correct guesses on the raw key",
       xlim=(0, max(0.32, sweep.z_qber.max() * 1.05)), ylim=(0.45, 1.03))
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.grid(color=GRID, lw=0.8)
ax.legend(frameon=False, loc="lower right")

# ---- Page 6: polarization drift ---------------------------------------------
fig, ax = new_page(
    "6. Polarization drift on an honest link",
    "The fibre slowly rotates the polarization; active compensation pulls it back. "
    "The error rate follows the drift.",
    "bb84_06_drift.png")
ax.plot(mon_df.t_s, np.sin(mon_df.theta / 2) ** 2, color=MUTED, lw=2,
        label="Error caused by misalignment alone")
ax.plot(mon_df.t_s, mon_df.z_qber, "o-", ms=5, lw=1, color=ATTACK_COLOURS[None],
        mec=SURFACE, mew=1, label="Measured error rate (Z basis)")
alarm_line(ax, 0)
ax.set(xlabel="Time (s)", ylabel="Error rate", ylim=(0, max(0.07, mon_df.z_qber.max() * 1.15)))
ax.yaxis.set_major_formatter(pct1)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right")

# ---- Page 7: QBER vs distance -------------------------------------------------
fig, ax = new_page(
    "7. Error rate vs fibre length (no Eve)",
    "Fewer photons arrive over longer fibre, so dark counts and afterpulses - random "
    "clicks - make up more of the detections.",
    "bb84_07_qber_distance.png")
ax.plot(km, qber_th, color=MUTED, lw=2, ls=":", label="Theory")
ax.plot(dist.km, dist.qber, "o", ms=10, color=ATTACK_COLOURS[None], mec=SURFACE, mew=1.5,
        label="Simulated sessions")
alarm_line(ax, 2)
ax.set(xlabel="Fibre length (km)", ylabel="Error rate (QBER)", ylim=(0, 0.25),
       xlim=(0, 260))
ax.yaxis.set_major_formatter(pct)
ax.grid(color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper left")

# ---- Page 8: key rate vs distance --------------------------------------------
fig, ax = new_page(
    "8. Secret key rate vs fibre length (no Eve)",
    "Longer sessions shrink the statistical safety margins, so they reach further. "
    "Short sessions hit a finite-size wall.",
    "bb84_08_keyrate_distance.png")
styles = ["-", "--", "-.", ":"]
greys = [ATTACK_COLOURS[None], "#1c5cab", "#104281", MUTED]
for (lab, v), ls, c in zip(th_lines.items(), styles, greys):
    ok = v > 0
    ax.plot(km[ok], v[ok], ls, color=c, lw=2, label=f"Theory: {lab} (max ~{max_km[lab]:.0f} km)")
pos = dist[dist.bps > 0]
ax.plot(pos.km, pos.bps, "o", ms=10, color=ATTACK_COLOURS[None], mec=SURFACE, mew=1.5,
        label="Simulated 1 s sessions")
for _, rr in dist[dist.bps <= 0].iterrows():
    ax.annotate("no key", (rr.km, 1.5), ha="center", fontsize=10, color=INK_2)
ax.set_yscale("log")
ax.set(xlabel="Fibre length (km)", ylabel="Secret key rate (bits per second)",
       xlim=(0, 260), ylim=(1, 2e7))
ax.yaxis.set_major_formatter(bps_fmt)
ax.grid(color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right", fontsize=10)

# ---- Page 9: finite-key effect ---------------------------------------------------
fig, ax = new_page(
    f"9. Why sessions must be long: the finite-key effect ({base.distance_km:g} km)",
    "Alice and Bob must leave a safety margin for statistical bad luck. With too few "
    "pulses the margin eats the whole key.",
    "bb84_09_finite_key.png")
ok = size_th > 0
ax.plot(Ns[ok] / base.rep_rate_hz, size_th[ok], color=ATTACK_COLOURS[None], lw=2,
        label="Theory")
ax.axhline(size_asym, color=MUTED, lw=1.5, ls=":")
ax.text(Ns[-1] / base.rep_rate_hz, size_asym * 1.08, "limit for infinitely long sessions",
        ha="right", fontsize=10, color=INK_2)
ok_s = sizes.bps > 0
ax.plot(sizes.N[ok_s] / base.rep_rate_hz, sizes.bps[ok_s], "o", ms=10,
        color=ATTACK_COLOURS[None], mec=SURFACE, mew=1.5, label="Simulated sessions")
for _, rr in sizes[~ok_s].iterrows():
    ax.plot(rr.N / base.rep_rate_hz, size_asym * 0.02, "x", ms=10, color=CRIT)
    ax.annotate("no key", (rr.N / base.rep_rate_hz, size_asym * 0.025), ha="center",
                fontsize=10, color=INK_2)
if np.isfinite(n_min):
    ax.axvline(n_min / base.rep_rate_hz, color=AXIS, lw=1)
    ax.text(n_min / base.rep_rate_hz * 1.1, size_asym * 0.5,
            f"no key below\n~{n_min / base.rep_rate_hz:.2g} s", fontsize=10, color=INK_2)
ax.set_xscale("log")
ax.set(xlabel=f"Session length (seconds at {base.rep_rate_hz / 1e9:g} GHz)",
       ylabel="Secret key rate", ylim=(0, size_asym * 1.2))
ax.yaxis.set_major_formatter(bps_fmt)
ax.grid(color=GRID, lw=0.8)
ax.legend(frameon=False, loc="center right")

# ---- Page 10: hardware presets --------------------------------------------------
fig, ax = new_page(
    "10. Better detectors reach much further",
    "Theory for 100 s sessions. Superconducting detectors have far fewer dark counts and "
    "higher efficiency than avalanche photodiodes.",
    "bb84_10_hardware.png")
pcols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
for (name, v), c in zip(preset_lines.items(), pcols):
    ok = v > 0
    mx = km2[ok].max() if ok.any() else 0
    ax.plot(km2[ok], v[ok], color=c, lw=2, label=f"{name} (max ~{mx:.0f} km)")
ax.set_yscale("log")
ax.set(xlabel="Fibre length (km)", ylabel="Secret key rate", xlim=(0, 400), ylim=(1, 1e8))
ax.yaxis.set_major_formatter(bps_fmt)
ax.grid(color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right")

# ---- Page 11: protocol comparison ----------------------------------------------
fig, ax = new_page(
    "11. BB84 compared with other protocols (same hardware)",
    "Infinitely long sessions. Without decoy states a laser source is limited to short "
    "links; with them it gets close to a perfect single-photon source.",
    "bb84_11_protocols.png")
pc = {"BB84, decoy states (weak laser)": ("#2a78d6", "-"),
      "BB84, no decoy states (weak laser)": ("#eb6834", "-"),
      "BB84, perfect single-photon source": ("#1baf7a", "--"),
      "Six-state, perfect single-photon source": ("#6250d6", "-."),
      "PLOB bound (no protocol can beat this)": (MUTED, ":")}
for name, v in proto.items():
    c, ls = pc[name]
    ok = v > 0
    mx = km2[ok].max() if ok.any() else 0
    lab = name if name.startswith("PLOB") else f"{name} (max ~{mx:.0f} km)"
    ax.plot(km2[ok], v[ok], ls, color=c, lw=2, label=lab)
ax.set_yscale("log")
ax.set(xlabel="Fibre length (km)", ylabel="Secret key rate", xlim=(0, 400), ylim=(1, 3e9))
ax.yaxis.set_major_formatter(bps_fmt)
ax.grid(color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right", fontsize=10)

# ---- Page 12: key budget ---------------------------------------------------------
r0 = results[0]
fk0 = r0["fk"]
budget = [
    ("Sifted key bits\n(Z basis)", fk0["nZ_total"]),
    ("Multi-photon &\nunknown pulses", -(fk0["nZ_total"] - fk0["sZ0"] - fk0["sZ1"])),
    ("Error correction\n(Cascade leakage)", -r0["ec_leaked"]),
    ("Privacy amplification\n(phase errors)", -fk0["sZ1"] * float(h2(fk0["phi"]))),
    ("Security margin\n(eps terms)", -fk0["margin"]),
    ("Authentication", -r0["auth_cost"]),
]
fig, ax = new_page(
    "12. Where the key goes: one honest session",
    f"From {fmt(fk0['nZ_total'])} sifted bits to {fmt(r0['net_key'])} secret bits "
    f"({r0['net_key'] / fk0['nZ_total']:.0%}). Most is spent hiding information from Eve.",
    "bb84_12_key_budget.png", size=(14, 7.5))
level = 0
for i, (lab, v) in enumerate(budget):
    if i == 0:
        ax.bar(i, v, color=ATTACK_COLOURS[None], width=0.6)
        level = v
        ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=10, color=INK)
    else:
        ax.bar(i, -v, bottom=level + v, color=CRIT if v < 0 else GOOD, alpha=0.75, width=0.6)
        ax.text(i, level, f"-{fmt(-v)}", ha="center", va="bottom", fontsize=10, color=INK_2)
        level += v
ax.bar(len(budget), r0["net_key"], color=GOOD, width=0.6)
ax.text(len(budget), r0["net_key"], fmt(r0["net_key"]), ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(range(len(budget) + 1), [b[0] for b in budget] + ["Final\nsecret key"],
              fontsize=10)
ax.set(ylabel="Bits")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
ax.grid(axis="y", color=GRID, lw=0.8)

# ---- Save: one PNG per graph + one PDF with a page per graph ---------------
with PdfPages("bb84_report.pdf") as pdf:
    for fig, name in figures:
        fig.tight_layout()
        fig.savefig(name, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
print("\nSaved " + ", ".join(n for _, n in figures))
print("Saved bb84_report.pdf (one graph per page)")
plt.show()
