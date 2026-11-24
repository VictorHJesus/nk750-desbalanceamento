import json
import os
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

import analise_1x

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("TCC_DATASET", RAIZ / "dataset"))
SINAIS = DATASET / "dados" if (DATASET / "dados").is_dir() else DATASET
SAIDA = Path(os.environ.get("TCC_SAIDA", RAIZ / "saida"))
GERADO = SAIDA / "gerado"
FIGS = SAIDA / "figuras"
R_MM = 100.0
PASSO = 22.5
M_ROTOR = 5.5
G_GRAU = 2.5
COR = {"ai0": "#1f77b4", "ai1": "#d95f02"}
COR_MODO = {"mesmo": "#2a9d8f", "nova": "#e76f51"}
NOME_MODO = {"mesmo": "mesma rotação no treino", "nova": "rotação não vista"}
SINC = [f"a{k}_{c}" for c in ("ai0", "ai1") for k in (1, 2, 3)] + ["rpm"]
ABL = SINC + ["rms_ai0", "curt_ai0", "rms_ai1", "curt_ai1"]
NOME_CAR = {"a1_ai0": "1× AI0", "a2_ai0": "2× AI0", "a3_ai0": "3× AI0", "a1_ai1": "1× AI1", "a2_ai1": "2× AI1",
            "a3_ai1": "3× AI1", "rpm": "rotação"}
OBS_CICLO = {"CF04": "interrompido após o estado B", "CF05": "estado A refeito",
             "CF07": "só o estado A válido"}
MODELOS = ("Floresta aleatória", "SVM (RBF)", "Rede neural (32-16)")
CHAVE = {"Floresta aleatória": "RF", "SVM (RBF)": "SVM", "Rede neural (32-16)": "RN", "Modelo físico": "Fis"}


def virg(x, casas=0):
    return f"{x:.{casas}f}".replace(".", ",")


def parse_parafusos(txt):
    if not isinstance(txt, str):
        return []
    m = re.search(r"furos\(furo:g\)=([\d:;.]+)", txt)
    if not m:
        return []
    return [(int(a), float(b)) for a, b in (p.split(":") for p in m.group(1).split(";"))]


def u_parafusos(lista, s):
    return sum(m * R_MM * np.exp(1j * s * np.radians(PASSO * f)) for f, m in lista)


def grupos_de_rpm(rpms, tol=15.0):
    ordem = sorted(set(round(r, 1) for r in rpms))
    g, ids, ant = -1, {}, None
    for r in ordem:
        if ant is None or r - ant > tol:
            g += 1
        ids[r] = g
        ant = r
    return [ids[round(r, 1)] for r in rpms]


def dif_ang(z1, z2):
    return abs(((np.degrees(np.angle(z1 / z2)) + 180) % 360) - 180)


def carregar():
    if not os.environ.get("PULAR_1X"):
        analise_1x.main(imprimir=False)
    est = pd.read_csv(SAIDA / "vetores_estados.csv")
    est = est[est.status == "valida"].copy()
    est["etapa"] = est.rotulo.str[-1].where(est.rotulo.str[-1].isin(list("ABC")), "Z")
    est["ciclo"] = np.where(est.etapa == "Z", est.rotulo, est.rotulo.str[:-1])
    rep = est.groupby(["rotulo", "canal"]).cumcount()
    est["rotulo"] = est.rotulo + np.where(rep > 0, "*" + rep.astype(str), "")
    est["V"] = est.re1 + 1j * est.im1
    cfg = {}
    excluidos = set()
    arq = DATASET / "ciclos_config.csv"
    if arq.exists():
        tab = pd.read_csv(arq)
        for r in tab.itertuples():
            cfg[r.ciclo] = parse_parafusos("furos(furo:g)=" + str(r.parafuso_fixo))
            if "excluir" in tab.columns and r.excluir == 1:
                excluidos.add(r.ciclo)
    est = est[~est.ciclo.isin(excluidos)].copy()
    est["grupo"] = grupos_de_rpm(est.rpm_nk.tolist())
    est["fixo"] = est.ciclo.map(lambda c: cfg.get(c, []))
    est["telas"] = est.obs.map(parse_parafusos)
    est["parafusos"] = [t + f for t, f in zip(est.telas, est.fixo)]
    sessoes = {d: i + 1 for i, d in enumerate(sorted(est.data.unique()))}
    est["sessao"] = est.data.map(sessoes)
    jan = pd.read_csv(SAIDA / "vetores_janelas.csv")
    jan = jan[jan.status == "valida"].copy()
    jan["rotulo"] = jan.arquivo.map(est.drop_duplicates("arquivo").set_index("arquivo").rotulo)
    jan = jan.dropna(subset=["rotulo"])
    ind = pd.read_csv(DATASET / "indice_coletas.csv")
    ind = ind[ind.status == "valida"].drop_duplicates("rotulo", keep="last").set_index("rotulo")
    return est, jan, ind


def ajustar(d, s, iteracoes=500):
    d = d.copy()
    d["Uadd"] = [u_parafusos(p, s) for p in d.parafusos]
    est0 = []
    for c, g in d[d.canal == "ai0"].groupby("ciclo"):
        a, b = g[g.etapa == "A"], g[g.etapa == "B"]
        if len(a) and len(b) and abs(b.Uadd.iloc[0] - a.Uadd.iloc[0]) > 0:
            al = (b.V.iloc[0] - a.V.iloc[0]) / (b.Uadd.iloc[0] - a.Uadd.iloc[0])
            est0.append(a.V.iloc[0] / al - a.Uadd.iloc[0])
    U0 = np.mean(est0) if est0 else 100 + 0j
    alpha = {}
    for _ in range(iteracoes):
        for (gr, ch), g in d.groupby(["grupo", "canal"]):
            U = U0 + g.Uadd.to_numpy()
            alpha[(gr, ch)] = np.sum(np.conj(U) * g.V.to_numpy()) / np.sum(np.abs(U) ** 2)
        num = den = 0
        for r in d.itertuples():
            a = alpha[(r.grupo, r.canal)]
            num += abs(a) ** 2 * (r.V / a - r.Uadd)
            den += abs(a) ** 2
        novo = num / den
        if abs(novo - U0) < 1e-10:
            U0 = novo
            break
        U0 = novo
    return U0, alpha


def h_ressonancia(n, nn, zeta):
    r = np.asarray(n, dtype=float) / nn
    return r ** 4 / (1 - r ** 2 + 2j * zeta * r)


def ajustar_ressonancia(alpha, rpg):
    gs = sorted({g for g, _ in alpha})
    n = np.array([rpg[g] for g in gs])
    obs = {ch: np.array([alpha[(g, ch)] for g in gs]) for ch in ("ai0", "ai1")}
    melhor = None
    for conj in (False, True):
        for nn0 in np.arange(450, 700, 20):
            for z0 in (0.02, 0.05, 0.1, 0.2):
                def res(p):
                    h = h_ressonancia(n, p[4], p[5])
                    h = np.conj(h) if conj else h
                    e = []
                    for i, ch in enumerate(("ai0", "ai1")):
                        c = p[i] * np.exp(1j * p[2 + i])
                        e.append((c * h - obs[ch]) / np.abs(obs[ch]))
                    e = np.concatenate(e)
                    return np.concatenate([e.real, e.imag])
                p0 = [np.median(np.abs(obs["ai0"])), np.median(np.abs(obs["ai1"])), 0, 0, nn0, z0]
                try:
                    r = least_squares(res, p0, bounds=([0, 0, -10, -10, 300, 0.001], [1, 1, 10, 10, 1200, 1]))
                except Exception:
                    continue
                if melhor is None or r.cost < melhor[0]:
                    melhor = (r.cost, r.x, conj)
    _, p, conj = melhor
    return dict(nn=p[4], zeta=p[5], conj=conj, C={"ai0": p[0] * np.exp(1j * p[2]), "ai1": p[1] * np.exp(1j * p[3])})


def alpha_ressonancia(res, n, ch):
    h = h_ressonancia(n, res["nn"], res["zeta"])
    return res["C"][ch] * (np.conj(h) if res["conj"] else h)


def alpha_linear(alpha, rpm_por_grupo, ch, rpm):
    gs = sorted(g for (g, c) in alpha if c == ch)
    abaixo = [g for g in gs if rpm_por_grupo[g] < rpm]
    acima = [g for g in gs if rpm_por_grupo[g] > rpm]
    if abaixo and acima:
        a, b = abaixo[-1], acima[0]
        w = (rpm - rpm_por_grupo[a]) / (rpm_por_grupo[b] - rpm_por_grupo[a])
        return (1 - w) * alpha[(a, ch)] + w * alpha[(b, ch)]
    n = min(gs, key=lambda g: abs(rpm_por_grupo[g] - rpm))
    return alpha[(n, ch)]


def estimar(d, alpha, rpm_por_grupo):
    linhas = []
    for rot, g in d.groupby("rotulo", sort=False):
        linhas.append((rot, np.mean([r.V / alpha[(r.grupo, r.canal)] for r in g.itertuples()])))
    return pd.DataFrame(linhas, columns=["rotulo", "Uhat"]).set_index("rotulo")


def validar(est, lab, s):
    saida, u0_dobra, alfa_usado, par = [], {}, {}, []
    for c in sorted(est.ciclo.unique()):
        tr, te = est[est.ciclo != c], est[est.ciclo == c]
        U0c, alc = ajustar(tr, s)
        u0_dobra[c] = U0c
        rpg = tr.groupby("grupo").rpm_nk.mean().to_dict()
        res = ajustar_ressonancia(alc, rpg)
        par.append(dict(ciclo=c, nn=res["nn"], zeta=res["zeta"]))
        rmin, rmax = tr.rpm_nk.min(), tr.rpm_nk.max()
        for rot, g in te.groupby("rotulo", sort=False):
            us, us_lin = [], []
            for r in g.itertuples():
                if (r.grupo, r.canal) in alc:
                    a, modo = alc[(r.grupo, r.canal)], "mesmo"
                    a_lin = a
                else:
                    a = alpha_ressonancia(res, r.rpm_nk, r.canal)
                    modo = "interpolado" if rmin < r.rpm_nk < rmax else "extrapolado"
                    a_lin = alpha_linear(alc, rpg, r.canal, r.rpm_nk)
                alfa_usado[(rot, r.canal)] = a
                us.append(r.V / a)
                us_lin.append(r.V / a_lin)
            uh, uh_lin = np.mean(us), np.mean(us_lin)
            ul = U0c + u_parafusos(lab.loc[rot, "parafusos"], s)
            saida.append(dict(rotulo=rot, ciclo=c, etapa=lab.loc[rot, "etapa"], rpm=lab.loc[rot, "rpm_nk"], modo=modo,
                              Ulab=abs(ul), Uhat=abs(uh), Ulab_c=ul, Uhat_c=uh,
                              evec=abs(uh - ul), emod=abs(abs(uh) - abs(ul)), eang=dif_ang(uh, ul),
                              evec_lin=abs(uh_lin - ul)))
    return pd.DataFrame(saida), u0_dobra, alfa_usado, pd.DataFrame(par)


def main():
    GERADO.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    est, jan, ind = carregar()
    rpm_por_grupo = est.groupby("grupo").rpm_nk.mean().to_dict()

    ajustes = {}
    for s in (+1, -1):
        U0, al = ajustar(est, s)
        d = est.assign(Uadd=[u_parafusos(p, s) for p in est.parafusos])
        res = np.array([abs(r.V / al[(r.grupo, r.canal)] - (U0 + r.Uadd)) for r in d.itertuples()])
        ajustes[s] = (U0, al, float(np.sqrt(np.mean(res ** 2))))
    s = min(ajustes, key=lambda k: ajustes[k][2])
    U0, alpha, rmse_in = ajustes[s]
    rmse_outro = ajustes[-s][2]
    est["Uadd"] = [u_parafusos(p, s) for p in est.parafusos]
    est["Ulab"] = U0 + est.Uadd
    ang_furo = lambda z: (np.degrees(np.angle(z)) / s) % 360
    print(f"sentido s={s:+d}  (RMSE s=+1: {ajustes[1][2]:.0f}  s=-1: {ajustes[-1][2]:.0f} g.mm)")
    print(f"U0 = {abs(U0):.1f} g.mm no furo-angulo {ang_furo(U0):.1f} graus")

    ress = ajustar_ressonancia(alpha, rpm_por_grupo)
    print(f"ressonancia: n_n = {ress['nn']:.0f} rpm, zeta = {ress['zeta']:.3f}")

    u0_ses = {}
    for ses, g in est.groupby("sessao"):
        if len(g) >= 4:
            u, _ = ajustar(g, s)
            u0_ses[ses] = u

    lab = est.drop_duplicates("rotulo").set_index("rotulo")
    in_amostra = estimar(est, alpha, rpm_por_grupo)
    in_amostra["Ulab"] = lab.Ulab
    in_amostra["err_in"] = (in_amostra.Uhat - in_amostra.Ulab).abs()

    cv, u0_dobra, alfa_usado, par_dobra = validar(est, lab, s)
    pd.set_option("display.width", 220)
    print(cv.drop(columns=["Ulab_c", "Uhat_c"]).round(1).to_string())
    print(cv.groupby("modo")[["evec", "emod", "eang", "evec_lin"]].agg(["count", "mean", "max"]).round(1))
    print("n_n por dobra:", par_dobra.nn.round(0).tolist(), " zeta:", par_dobra.zeta.round(3).tolist())

    rep = []
    u = est[est.canal == "ai0"].drop_duplicates("rotulo")
    for i, a in u.iterrows():
        for j, b in u.iterrows():
            if a.rotulo < b.rotulo and a.ciclo != b.ciclo and a.grupo == b.grupo and abs(a.Ulab - b.Ulab) < 10 and abs(a.Ulab) > 100:
                rep.append((a.rotulo, b.rotulo, abs(a.V - b.V) / abs(a.V), a.rpm_nk, b.rpm_nk))
    print("repetibilidade:", [(a, b, round(100 * d, 1), ra, rb) for a, b, d, ra, rb in rep])

    nk = []
    ultimos = {}
    for rot in in_amostra.index:
        base = rot.split("*")[0]
        if base[-1] == "C":
            ultimos[base] = rot
    for base, rot in ultimos.items():
        if base in ind.index and pd.notna(ind.loc[base, "nk700_residual_g"]):
            nk.append((rot, ind.loc[base, "nk700_residual_g"] * 100, abs(in_amostra.loc[rot, "Uhat"]), abs(in_amostra.loc[rot, "Ulab"])))
    nk = pd.DataFrame(nk, columns=["rotulo", "nk_res", "nosso_res", "rotulo_res"])

    ml = ml_classico(jan, est, lab, cv, u0_dobra, alfa_usado, s)

    figuras(est, jan, alpha, rpm_por_grupo, ress, cv, ml, ang_furo)
    escrever_tex(est, lab, in_amostra, cv, rep, nk, ml, U0, u0_ses, rmse_in, rmse_outro, s, ang_furo, alpha,
                 rpm_por_grupo, ress, par_dobra)


def tabela_janelas(jan, lab):
    linhas = []
    for rot, g in jan.groupby("rotulo"):
        if rot not in lab.index:
            continue
        f = {}
        for ch in ("ai0", "ai1"):
            x = g[g.canal == ch].set_index("janela")
            for k in (1, 2, 3):
                f[f"a{k}_{ch}"] = np.hypot(x[f"re{k}"], x[f"im{k}"])
            f[f"V_{ch}"] = x.re1 + 1j * x.im1
            f[f"rms_{ch}"] = x.rms_g
            f[f"curt_{ch}"] = x.curtose
        f = pd.DataFrame(f).dropna()
        f["rpm"] = lab.loc[rot, "rpm_nk"]
        f["rotulo"] = rot
        f["ciclo"] = lab.loc[rot, "ciclo"]
        linhas.append(f)
    F = pd.concat(linhas).reset_index(drop=True)
    F["Uper"] = 9549 * G_GRAU * M_ROTOR / F.rpm
    return F


def ml_classico(jan, est, lab, cv, u0_dobra, alfa_usado, s):
    F = tabela_janelas(jan, lab)
    modo = cv.set_index("rotulo").modo.map(lambda m: "mesmo" if m == "mesmo" else "nova")
    F["modo"] = F.rotulo.map(modo)
    ciclos = sorted(F.ciclo.unique())
    mk_cls = {"Floresta aleatória": lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1),
              "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC()),
              "Rede neural (32-16)": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=0))}
    mk_reg = {"Floresta aleatória": lambda: RandomForestRegressor(200, random_state=0, n_jobs=-1),
              "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVR(C=1000)),
              "Rede neural (32-16)": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=0))}
    preds, imp_reg, imp_cls = [], [], []
    for c in ciclos:
        tr, te = (F.ciclo != c).to_numpy(), (F.ciclo == c).to_numpy()
        y = np.array([abs(u0_dobra[c] + u_parafusos(lab.loc[r, "parafusos"], s)) for r in F.rotulo])
        yc = (y > F.Uper.to_numpy()).astype(int)
        base = F.loc[te, ["rotulo", "ciclo", "modo", "rpm"]].copy()
        base["y"], base["yc"] = y[te], yc[te]

        def junta(nome, conj, yhat, ychat):
            d = base.copy()
            d["modelo"], d["conj"], d["yhat"], d["ychat"] = nome, conj, yhat, ychat
            preds.append(d)

        maj = int(yc[tr].mean() >= 0.5)
        junta("Referência", "-", np.full(te.sum(), y[tr].mean()), np.full(te.sum(), maj))
        Ft = F[te]
        uh = np.mean([Ft[f"V_{ch}"].to_numpy() / np.array([alfa_usado[(r, ch)] for r in Ft.rotulo]) for ch in ("ai0", "ai1")], axis=0)
        junta("Modelo físico", "-", np.abs(uh), (np.abs(uh) > Ft.Uper.to_numpy()).astype(int))
        for conj, cols in (("sinc", SINC), ("abl", ABL)):
            X = F[cols]
            for nome in MODELOS:
                mc = mk_cls[nome]().fit(X[tr], yc[tr])
                mr = mk_reg[nome]().fit(X[tr], y[tr])
                junta(nome, conj, mr.predict(X[te]), mc.predict(X[te]))
                if conj == "sinc" and nome == "Floresta aleatória":
                    pr = permutation_importance(mr, X[te], y[te], scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
                    pc = permutation_importance(mc, X[te], yc[te], scoring="accuracy", n_repeats=5, random_state=0)
                    imp_reg.append(pr.importances_mean)
                    imp_cls.append(pc.importances_mean)
    P = pd.concat(preds, ignore_index=True)
    P["e"] = (P.yhat - P.y).abs()

    def metricas(d):
        por_dobra = d.groupby("ciclo").apply(lambda g: (g.ychat == g.yc).mean())
        return dict(acc=por_dobra.mean(), acc_min=por_dobra.min(), pior=por_dobra.idxmin(),
                    acc_bal=balanced_accuracy_score(d.yc, d.ychat), mae=d.e.mean(),
                    mae_mesmo=d[d.modo == "mesmo"].e.mean(), mae_nova=d[d.modo == "nova"].e.mean())

    linhas = []
    for (nome, conj), d in P.groupby(["modelo", "conj"], sort=False):
        linhas.append(dict(modelo=nome, conj=conj, **metricas(d)))
    tab = pd.DataFrame(linhas)
    print(tab.round(3).to_string())
    return dict(tab=tab, P=P, F=F, janelas=len(F), estados=F.rotulo.nunique(), ciclos=len(ciclos),
                prevalencia=P[P.modelo == "Referência"].yc.mean(),
                imp_reg=pd.Series(np.mean(imp_reg, axis=0), index=SINC),
                imp_cls=pd.Series(np.mean(imp_cls, axis=0), index=SINC))


def figuras(est, jan, alpha, rpm_por_grupo, ress, cv, ml, ang_furo):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
    cands = est[(est.canal == "ai0") & (est.etapa == "B")].sort_values("rms_g")
    if len(cands):
        rot = cands.iloc[-1].rotulo if "CF05B" not in set(cands.rotulo) else "CF05B"
        arq = next(SINAIS.rglob(f"{rot}_*.csv"))
        d = pd.read_csv(arq)
        meta = json.loads(arq.with_suffix(".json").read_text(encoding="utf-8"))
        fs = meta["fs_real_hz"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
        for ch in ("ai0", "ai1"):
            x = d[f"{ch}_g"].to_numpy()
            x = x - x.mean()
            w = np.hanning(len(x))
            esp = 2 * np.abs(np.fft.rfft(x * w)) / w.sum()
            f = np.fft.rfftfreq(len(x), 1 / fs)
            ax[0].plot(f, esp, color=COR[ch], lw=0.6, label=ch.upper())
            fr = est[est.rotulo == rot].rpm_tac.iloc[0] / 60
            m = (f > 0.6 * fr) & (f < 3.4 * fr)
            ax[1].plot(f[m], esp[m] * 1000, color=COR[ch], lw=1.0, label=ch.upper())
        ax[0].set_xlim(0, 2000)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Frequência (Hz)")
        ax[0].set_ylabel("Amplitude (g)")
        ax[0].set_title(f"Coleta {rot}, 0 a 2000 Hz")
        ax[0].legend(loc="upper right")
        for k in (1, 2, 3):
            ax[1].axvline(k * fr, color="gray", ls="--", lw=0.8)
            ax[1].text(k * fr, ax[1].get_ylim()[1] * 0.95, f"{k}×", ha="center", va="top", fontsize=9)
        ax[1].set_xlabel("Frequência (Hz)")
        ax[1].set_ylabel("Amplitude (mg)")
        ax[1].set_title(f"Detalhe em torno de $f_r$ = {fr:.2f} Hz")
        fig.savefig(FIGS / "fig_espectro.png")
        plt.close(fig)

    completos = [c for c, g in est[est.canal == "ai0"].groupby("ciclo") if set("ABC") <= set(g.etapa)]
    if completos:
        c = "CF05" if "CF05" in completos else completos[-1]
        fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
        marc = {"A": "o", "B": "s", "C": "^"}
        for ch in ("ai0", "ai1"):
            for r in est[(est.ciclo == c) & (est.canal == ch)].itertuples():
                jj = jan[(jan.rotulo == r.rotulo) & (jan.canal == ch)]
                z = ((jj.re1 + 1j * jj.im1) * 1000).to_numpy()
                ax.scatter(z.real, z.imag, s=6, color=COR[ch], alpha=0.25)
                ax.scatter([r.V.real * 1000], [r.V.imag * 1000], marker=marc[r.etapa], s=70, color=COR[ch],
                           edgecolor="k", zorder=3, label=f"{ch.upper()} {r.etapa}")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_aspect("equal")
        lim = ax.get_xlim() + ax.get_ylim()
        m = max(abs(v) for v in lim)
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_xlabel("Parte real do vetor 1× (mg)")
        ax.set_ylabel("Parte imaginária (mg)")
        ax.set_title(f"Ciclo {c}: A, B e C")
        ax.legend(fontsize=7, ncol=2, loc="lower left")
        fig.savefig(FIGS / "fig_vetores_ciclo.png")
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    nc = np.linspace(330, 900, 400)
    for ch in ("ai0", "ai1"):
        gs = sorted(g for (g, c) in alpha if c == ch)
        rp = [rpm_por_grupo[g] for g in gs]
        mod = [abs(alpha[(g, ch)]) * 1e6 for g in gs]
        fase = [np.degrees(np.angle(alpha[(g, ch)])) % 360 for g in gs]
        ac = alpha_ressonancia(ress, nc, ch)
        ax[0].plot(nc, np.abs(ac) * 1e6, "-", color=COR[ch], lw=1.0, alpha=0.8)
        ax[1].plot(nc, np.degrees(np.angle(ac)) % 360, "-", color=COR[ch], lw=1.0, alpha=0.8)
        ax[0].plot(rp, mod, "o", color=COR[ch], label=f"{ch.upper()} (ajuste por rotação)")
        ax[1].plot(rp, fase, "o", color=COR[ch], label=ch.upper())
    for a_ in ax:
        a_.axvline(ress["nn"], color="gray", ls=":", lw=0.9)
    ax[0].plot([], [], "-", color="k", lw=1.0, label="modelo de ressonância")
    ax[0].set_xlabel("Rotação (rpm)")
    ax[0].set_ylabel(r"$|\alpha|$ ($\mu$g por g·mm)")
    ax[1].set_xlabel("Rotação (rpm)")
    ax[1].set_ylabel(r"$\arg\alpha$ (°)")
    ax[1].set_ylim(0, 360)
    ax[0].legend(fontsize=7)
    fig.savefig(FIGS / "fig_alpha_rpm.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5.0), constrained_layout=True)
    cv = cv.assign(m2=cv.modo.map(lambda m: "mesmo" if m == "mesmo" else "nova"))
    for m2, g in cv.groupby("m2"):
        ax[0].scatter(g.Ulab, g.Uhat, s=40, color=COR_MODO[m2], edgecolor="k", label=NOME_MODO[m2])
    mx = max(cv.Ulab.max(), cv.Uhat.max()) * 1.05
    ax[0].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[0].set_xlim(0, mx)
    ax[0].set_ylim(0, mx)
    ax[0].set_xlabel(r"$|U_{liq}|$ do rótulo (g·mm)")
    ax[0].set_ylabel(r"$|\hat U_{liq}|$ estimado (g·mm)")
    ax[0].legend(fontsize=8)
    para_furo = lambda z: abs(z) * np.exp(1j * np.radians(ang_furo(z)))
    for r in cv.itertuples():
        a_, b_ = para_furo(r.Ulab_c), para_furo(r.Uhat_c)
        ax[1].annotate("", xy=(b_.real, b_.imag), xytext=(a_.real, a_.imag),
                       arrowprops=dict(arrowstyle="->", color=COR_MODO[r.m2], lw=1.0))
        ax[1].plot(a_.real, a_.imag, "o", ms=4, mfc="white", mec="k", zorder=3)
    ax[1].plot([], [], "o", ms=4, mfc="white", mec="k", label="rótulo")
    for m2 in ("mesmo", "nova"):
        ax[1].plot([], [], "-", color=COR_MODO[m2], label=f"estimativa ({NOME_MODO[m2]})")
    pts = np.array([para_furo(z) for z in list(cv.Ulab_c) + list(cv.Uhat_c)])
    lim = 1.1 * max(np.abs(pts.real).max(), np.abs(pts.imag).max())
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$|U|\cos\theta$ (g·mm)")
    ax[1].set_ylabel(r"$|U|\sin\theta$ (g·mm)")
    ax[1].set_title("Rótulo → estimativa (ângulo dos furos)", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")
    fig.savefig(FIGS / "fig_validacao_ciclos.png")
    plt.close(fig)

    if not ml:
        return
    P = ml["P"]
    nomes = ["Modelo físico"] + list(MODELOS)
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.6), constrained_layout=True, sharex=True, sharey=True)
    mx = 0
    for nome, ax in zip(nomes, axs.flat):
        d = P[(P.modelo == nome) & (P.conj.isin(["-", "sinc"]))]
        g = d.groupby(["rotulo", "modo"]).agg(y=("y", "first"), med=("yhat", "median"),
                                              q1=("yhat", lambda v: v.quantile(0.25)), q3=("yhat", lambda v: v.quantile(0.75))).reset_index()
        for m2, gg in g.groupby("modo"):
            ax.errorbar(gg.y, gg.med, yerr=[gg.med - gg.q1, gg.q3 - gg.med], fmt="o", ms=5, color=COR_MODO[m2],
                        mec="k", ecolor=COR_MODO[m2], elinewidth=1, capsize=2, label=NOME_MODO[m2])
        mx = max(mx, g.y.max(), g.q3.max())
        ax.set_title(f"{nome}: EAM = {d.e.mean():.0f} g·mm".replace(".", ","), fontsize=9)
    for ax in axs.flat:
        ax.plot([0, 1.05 * mx], [0, 1.05 * mx], "k--", lw=0.8)
        ax.set_xlim(0, 1.05 * mx)
        ax.set_ylim(0, 1.05 * mx)
    for ax in axs[1]:
        ax.set_xlabel(r"$|U_{liq}|$ do rótulo (g·mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$|\hat U_{liq}|$ previsto (g·mm)")
    axs[0, 0].legend(fontsize=8, loc="upper left")
    fig.savefig(FIGS / "fig_ml_regressao.png")
    plt.close(fig)

    nomes_c = ["Referência"] + nomes
    fig, axs = plt.subplots(1, 5, figsize=(11, 2.9), constrained_layout=True)
    for nome, ax in zip(nomes_c, axs):
        d = P[(P.modelo == nome) & (P.conj.isin(["-", "sinc"]))]
        cm = confusion_matrix(d.yc, d.ychat, labels=[0, 1])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{100 * cmn[i, j]:.0f}%\n({cm[i, j]})".replace(".", ","), ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["dentro", "fora"])
        ax.set_yticks([0, 1], ["dentro", "fora"])
        ax.grid(False)
        ax.set_xlabel("Previsto")
        tit = "Classe majoritária" if nome == "Referência" else nome
        ab = virg(balanced_accuracy_score(d.yc, d.ychat), 2)
        ax.set_title(f"{tit}\nac. bal. = {ab}", fontsize=8)
    axs[0].set_ylabel("Real (rótulo)")
    fig.savefig(FIGS / "fig_ml_confusao.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), constrained_layout=True)
    for ax, imp, tit, un in ((axs[0], ml["imp_cls"], "Classificação", "queda da acurácia (p.p.)"),
                             (axs[1], ml["imp_reg"], "Regressão", "aumento do EAM (g·mm)")):
        v = imp * (100 if tit == "Classificação" else 1)
        v = v.sort_values()
        ax.barh([NOME_CAR[k] for k in v.index], v.to_numpy(), color="#457b9d")
        ax.set_xlabel(un)
        ax.set_title(f"Floresta aleatória: {tit.lower()}", fontsize=9)
    fig.savefig(FIGS / "fig_ml_importancia.png")
    plt.close(fig)

    F, Pr = ml["F"], P[P.modelo == "Referência"]
    cls = Pr.groupby("rotulo").yc.first()
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    for k, cor, nm in ((0, "#2a9d8f", "dentro da tolerância"), (1, "#e76f51", "fora da tolerância")):
        d = F[F.rotulo.map(cls) == k]
        ax.scatter(d.rpm + np.random.default_rng(0).normal(0, 3, len(d)), d.a1_ai0 * 1000, s=6, alpha=0.4, color=cor, label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("Rotação (rpm)")
    ax.set_ylabel("Amplitude 1× no AI0 (mg)")
    ax.legend(fontsize=8, markerscale=2)
    fig.savefig(FIGS / "fig_ml_espaco.png")
    plt.close(fig)


def tabela(nome, cabecalho, colunas, linhas):
    t = ["% Gerado por src/modelo_tcc.py", f"\\begin{{tabular}}{{{colunas}}}", "\\toprule"] + cabecalho + ["\\midrule"]
    t += [l if l.startswith(("\\midrule", "\\cmidrule")) else l + "\\\\" for l in linhas]
    t += ["\\bottomrule", "\\end{tabular}"]
    (GERADO / nome).write_text("\n".join(t) + "\n", encoding="utf-8")


def escrever_tex(est, lab, ina, cv, rep, nk, ml, U0, u0_ses, rmse_in, rmse_outro, s, ang_furo, alpha, rpg, ress, par_dobra):
    m = {}
    m["NColetas"] = str(est.rotulo.nunique())
    m["NCiclos"] = str(est.ciclo.nunique())
    m["NRotacoes"] = str(est.grupo.nunique())
    m["UZero"] = virg(abs(U0))
    m["UZeroG"] = virg(abs(U0) / R_MM, 2)
    m["UZeroAng"] = virg(ang_furo(U0), 1)
    m["CorrAng"] = virg((ang_furo(U0) + 180) % 360, 1)
    m["UZeroEsp"] = virg(abs(U0) / M_ROTOR, 0)
    ses = sorted(u0_ses)
    m["NSessoes"] = str(len(ses))
    for i, k in enumerate(ses):
        L = "ABCD"[i]
        m[f"UZeroSes{L}"], m[f"UZeroAngSes{L}"] = virg(abs(u0_ses[k])), virg(ang_furo(u0_ses[k]), 1)
    m["UZeroDifPct"] = virg(100 * max(abs(a - b) for a in u0_ses.values() for b in u0_ses.values()) / abs(U0), 1)
    m["RMSEAmostra"] = virg(rmse_in)
    m["RMSEOutro"] = virg(rmse_outro)
    m["Sentido"] = "oposto ao" if s == -1 else "igual ao"
    m["Nn"] = virg(ress["nn"])
    m["NnHz"] = virg(ress["nn"] / 60, 1)
    m["Zeta"] = virg(ress["zeta"], 3)
    m["NnMin"], m["NnMax"] = virg(par_dobra.nn.min()), virg(par_dobra.nn.max())
    m["ZetaMin"], m["ZetaMax"] = virg(par_dobra.zeta.min(), 3), virg(par_dobra.zeta.max(), 3)
    cv = cv.copy()
    cv["m2"] = cv.modo.map(lambda x: "mesmo" if x == "mesmo" else "nova")
    for modo, nome in (("mesmo", "Mesmo"), ("interpolado", "Interp"), ("extrapolado", "Extrap"), ("nova", "Nova")):
        g = cv[cv.m2 == modo] if modo == "nova" else cv[cv.modo == modo]
        m[f"N{nome}"] = str(len(g))
        m[f"Err{nome}Media"] = virg(g.evec.mean()) if len(g) else "--"
        m[f"Err{nome}Max"] = virg(g.evec.max()) if len(g) else "--"
        m[f"ErrMod{nome}"] = virg(g.emod.mean()) if len(g) else "--"
        m[f"ErrLin{nome}"] = virg(g.evec_lin.mean()) if len(g) else "--"
    mesmo = cv[cv.modo == "mesmo"]
    m["AngErrAB"] = virg(mesmo[mesmo.etapa.isin(["A", "B"])].eang.mean(), 1)
    m["AngErrABMax"] = virg(mesmo[mesmo.etapa.isin(["A", "B"])].eang.max(), 1)
    m["AngErrC"] = virg(mesmo[mesmo.etapa == "C"].eang.mean(), 1)
    m["AngErrCMax"] = virg(mesmo[mesmo.etapa == "C"].eang.max(), 1)
    m["AngErrNovaAB"] = virg(cv[(cv.m2 == "nova") & cv.etapa.isin(["A", "B"])].eang.mean(), 1)
    m["ErrMesmoPctUZero"] = virg(100 * mesmo.evec.mean() / abs(U0), 0)
    for rot in ("CF03A", "CF03B"):
        if rot in set(cv.rotulo):
            r = cv[cv.rotulo == rot].iloc[0]
            k = rot.replace("CF03", "CFzeroTres")
            m[f"Err{k}"], m[f"ErrLin{k}"] = virg(r.evec), virg(r.evec_lin)
    mesma = [r for r in rep if abs(r[3] - r[4]) < 5]
    m["RepDifPct"] = virg(100 * np.mean([r[2] for r in mesma]), 1) if mesma else "--"
    m["RepDifMax"] = virg(100 * np.max([r[2] for r in mesma]), 0) if mesma else "--"
    m["NRepMesma"] = str(len(mesma))
    m["NRep"] = str(len(rep))
    lr = ["\\multicolumn{4}{l}{\\textit{Mesma rotação}}"]
    lr += [f"{a} & {b} & {virg(ra)} & {virg(100 * d, 1)}" for a, b, d, ra, rb in sorted(mesma, key=lambda r: r[2])]
    outras = [r for r in rep if abs(r[3] - r[4]) >= 5]
    if outras:
        lr += ["\\midrule", "\\multicolumn{4}{l}{\\textit{Rotações diferentes dentro do mesmo grupo}}"]
        lr += [f"{a} & {b} & {virg(ra)} e {virg(rb)} & {virg(100 * d, 1)}" for a, b, d, ra, rb in sorted(outras, key=lambda r: r[2])]
    tabela("tab_rep.tex", ["Estado 1 & Estado 2 & Rotação & Diferença entre\\\\",
                           " & & (rpm) & os vetores $1\\times$ (\\%)\\\\"], "llcr", lr)
    if len(nk):
        m["NKResMedio"] = virg(nk.nk_res.mean())
        m["NossoResMedio"] = virg(nk.nosso_res.mean())
    if ml:
        tab = ml["tab"].set_index(["modelo", "conj"])
        m["MLJanelas"], m["MLEstados"], m["MLCiclos"] = str(ml["janelas"]), str(ml["estados"]), str(ml["ciclos"])
        m["MLPrev"] = virg(100 * ml["prevalencia"], 0)
        ref = tab.loc[("Referência", "-")]
        m["MLMaj"], m["MLBase"] = virg(100 * ref.acc, 0), virg(ref.mae)
        m["MLBaseMesmo"], m["MLBaseNova"] = virg(ref.mae_mesmo), virg(ref.mae_nova)
        for (nome, conj), r in tab.iterrows():
            if nome == "Referência":
                continue
            k = CHAVE[nome] + ("Abl" if conj == "abl" else "")
            m[f"MLAcc{k}"], m[f"MLAccMin{k}"], m[f"MLAccBal{k}"] = virg(100 * r.acc, 0), virg(100 * r.acc_min, 0), virg(r.acc_bal, 2)
            m[f"MLPior{k}"] = r.pior
            m[f"MLMae{k}"], m[f"MLMaeMesmo{k}"], m[f"MLMaeNova{k}"] = virg(r.mae), virg(r.mae_mesmo), virg(r.mae_nova)
        linhas = []
        rot = {"Referência": "Referência", "Modelo físico": "Modelo físico ($V_1/\\alpha$)"}
        def lin(nome, conj, r):
            nm = rot.get(nome, nome)
            bal = "0,50" if nome == "Referência" else virg(r.acc_bal, 2)
            return (f"{nm} & {virg(100 * r.acc, 0)} & {virg(100 * r.acc_min, 0)} & {bal} & "
                    f"{virg(r.mae)} & {virg(r.mae_mesmo)} & {virg(r.mae_nova)}")
        for nome, conj in (("Referência", "-"), ("Modelo físico", "-")):
            linhas.append(lin(nome, conj, tab.loc[(nome, conj)]))
        linhas.append("\\midrule")
        linhas.append("\\multicolumn{7}{l}{\\textit{Componentes síncronas e rotação}}")
        for nome in MODELOS:
            linhas.append(lin(nome, "sinc", tab.loc[(nome, "sinc")]))
        linhas.append("\\midrule")
        linhas.append("\\multicolumn{7}{l}{\\textit{Ablação: com RMS e curtose}}")
        for nome in MODELOS:
            linhas.append(lin(nome, "abl", tab.loc[(nome, "abl")]))
        tabela("tab_ml.tex", ["Modelo & \\multicolumn{3}{c}{Classificação (maior é melhor)} & \\multicolumn{3}{c}{Regressão: EAM em g·mm (menor é melhor)}\\\\",
                              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                              " & Ac. média (\\%) & Ac. pior ciclo (\\%) & Ac. bal. & Todos & Rot. vista & Rot. não vista\\\\"],
               "lrrrrrr", linhas)
        print(ml["tab"].round(3).to_string())

    linhas = ["% Gerado por src/modelo_tcc.py - NAO editar a mao."]
    linhas += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (GERADO / "numeros.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    t = []
    for c, g in est[est.canal == "ai0"].groupby("ciclo", sort=True):
        fx = "; ".join(f"{m_:g} g no furo {f}".replace(".", ",") for f, m_ in g.fixo.iloc[0])
        obs = [OBS_CICLO[c]] if c in OBS_CICLO else []
        nrec = int(g.rotulo.str.contains("*", regex=False).sum())
        if nrec:
            obs.append(f"estado C recoletado ({nrec + 1} coletas válidas)")
        if fx:
            obs.append(f"parafuso fixo de {fx}")
        t.append(f"{c} & {g.sessao.iloc[0]} & {virg(g.rpm_nk.mean())} & {''.join(sorted(set(g.etapa)))} & {g.rotulo.nunique()} & {'; '.join(obs) or '--'}")
    tabela("tab_coletas.tex", ["Ciclo & Sessão & Rotação (rpm) & Estados & Coletas & Observação\\\\"], "lcrccl", t)

    t = []
    for rot in ina.index:
        a = ina.loc[rot]
        t.append(f"{rot} & {virg(lab.loc[rot, 'rpm_nk'])} & {virg(abs(a.Ulab))} & {virg(abs(a.Uhat))} & {virg(abs(a.err_in))} & "
                 f"{virg(ang_furo(a.Ulab))} & {virg(ang_furo(a.Uhat))}")
    tabela("tab_estados.tex", ["Coleta & Rot. (rpm) & $|U_{liq}|$ rótulo & $|\\hat U_{liq}|$ & Erro vetorial & Ângulo rótulo (°) & Ângulo est. (°)\\\\",
                               " & & (g·mm) & (g·mm) & (g·mm) & & \\\\"], "llrrrrr", t)

    origem = {"mesmo": "mesma rotação", "interpolado": "ressonância (interp.)", "extrapolado": "ressonância (extrap.)"}
    t = [f"{r.rotulo} & {virg(r.rpm)} & {virg(r.Ulab)} & {virg(r.Uhat)} & {virg(r.evec)} & {virg(r.emod)} & {virg(r.eang)} & {origem[r.modo]}"
         for r in cv.itertuples()]
    tabela("tab_validacao.tex", ["Coleta & Rotação & $|\\vec{U}_{liq}|$ & $|\\vec{U}_{liq}|$ & Erro & Erro de & Erro de & Origem de $\\alpha$\\\\",
                                 " & (rpm) & rótulo (g·mm) & estimado (g·mm) & vetorial (g·mm) & módulo (g·mm) & ângulo (°) & \\\\"], "lrrrrrrl", t)

    t = []
    cv2 = cv.assign(vista=cv.modo == "mesmo", ab=cv.etapa.isin(["A", "B"]))
    for vista, nv in ((True, "vista no treino"), (False, "não vista")):
        for ab, ne in ((True, "A e B"), (False, "C (corrigido)")):
            g = cv2[(cv2.vista == vista) & (cv2.ab == ab)]
            if not len(g):
                continue
            lin_ = "--" if vista else virg(g.evec_lin.mean())
            t.append(f"{nv if ab else ''} & {ne} & {len(g)} & {virg(g.Ulab.min())} a {virg(g.Ulab.max())} & "
                     f"{virg(g.evec.mean())} & {virg(g.evec.max())} & {virg(g.eang.mean())} & {lin_}")
        if vista:
            t.append("\\midrule")
    tabela("tab_validacao_resumo.tex",
           ["Rotação do & Estados & Coletas & $|\\vec{U}_{liq}|$ & \\multicolumn{2}{c}{Erro vetorial (g·mm)} & Erro de & Erro médio se $\\alpha$\\\\",
            "\\cmidrule(lr){5-6}",
            "ciclo de fora & & & rótulo (g·mm) & médio & máximo & ângulo médio (°) & fosse interpolado (g·mm)\\\\"],
           "llcrrrrr", t)

    t = []
    for modo, nm in (("mesmo", "Mesma rotação ($\\alpha$ do grupo)"), ("interpolado", "Rotação interpolada"), ("extrapolado", "Rotação extrapolada")):
        g = cv[cv.modo == modo]
        if len(g):
            lin_ = virg(g.evec_lin.mean()) if modo != "mesmo" else "--"
            t.append(f"{nm} & {len(g)} & {lin_} & {virg(g.evec.mean())}")
    tabela("tab_alpha_modo.tex", ["Situação do ciclo de fora & Coletas & $\\alpha$ interpolado linearmente & $\\alpha$ de ressonância\\\\"], "lrrr", t)

    t = []
    for (g, ch), a in sorted(alpha.items(), key=lambda kv: rpg[kv[0][0]]):
        if ch != "ai0":
            continue
        nc = est[est.grupo == g].ciclo.nunique()
        ar = alpha_ressonancia(ress, rpg[g], ch)
        fa, fr = np.degrees(np.angle(a)) % 360, np.degrees(np.angle(ar)) % 360
        t.append(f"{virg(rpg[g])} & {nc} & {virg(abs(a) * 1e6, 2)} & {virg(abs(ar) * 1e6, 2)} & "
                 f"{virg(100 * (abs(ar) - abs(a)) / abs(a))} & {virg(fa)} & {virg(fr)} & {virg(((fr - fa + 180) % 360) - 180)}")
    tabela("tab_alpha.tex", ["Rotação & Ciclos & \\multicolumn{3}{c}{$|\\alpha|$ ($\\mu$g por g·mm)} & \\multicolumn{3}{c}{Fase de $\\alpha$ (°)}\\\\",
                             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
                             "(rpm) & & medido & modelo & diferença (\\%) & medida & modelo & diferença\\\\"], "rcrrrrrr", t)

    if len(nk):
        tabela("tab_nk.tex", ["Estado & \\multicolumn{3}{c}{Residual após a correção (g·mm)}\\\\",
                              "\\cmidrule(lr){2-4}",
                              "corrigido & \\textit{software} NK700 & rótulo & estimado pelo vetor $1\\times$\\\\"], "lrrr",
               [f"{r.rotulo} & {virg(r.nk_res)} & {virg(r.rotulo_res)} & {virg(r.nosso_res)}" for r in nk.itertuples()])


if __name__ == "__main__":
    main()
