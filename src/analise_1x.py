import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from coleta_teste import bordas_tacometro

RAIZ = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("TCC_DATASET", RAIZ / "dataset"))
SINAIS = DATASET / "dados" if (DATASET / "dados").is_dir() else DATASET
SAIDA = Path(os.environ.get("TCC_SAIDA", RAIZ / "saida"))
JANELA_S = 1.0
CANAIS = ("ai0", "ai1")
HARMONICOS = (1, 2, 3)


def pulsos_limpos(t, idx):
    te = t[idx]
    te = te[te > 0.01]
    if len(te) < 5:
        return np.array([]), np.array([])
    med = np.median(np.diff(te))
    aceitos = [te[0]]
    voltas = [0]
    for x in te[1:]:
        passos = (x - aceitos[-1]) / med
        n = int(round(passos))
        if n >= 1 and abs(passos - n) < 0.3:
            aceitos.append(x)
            voltas.append(voltas[-1] + n)
    return np.array(aceitos), np.array(voltas, dtype=float)


def angulo_rotor(t, tp, voltas):
    return 2 * np.pi * np.interp(t, tp, voltas, left=np.nan, right=np.nan)

def vetor(x, theta, k=1):
    ok = ~np.isnan(theta)
    xx, th = x[ok] - np.mean(x[ok]), theta[ok]
    return 2 * np.mean(xx * np.exp(-1j * k * th))


def processar(csv):
    meta = json.loads(csv.with_suffix(".json").read_text(encoding="utf-8"))
    d = pd.read_csv(csv)
    fs = meta["fs_real_hz"]
    t = d["tempo_s"].to_numpy()
    idx, _ = bordas_tacometro(d["ai3_V"].to_numpy(), fs)
    tp, voltas = pulsos_limpos(t, idx)
    if len(tp) < 10:
        return [], []
    theta = angulo_rotor(t, tp, voltas)
    rpm_tac = 60.0 * voltas[-1] / (tp[-1] - tp[0])
    n = int(JANELA_S * fs)
    estados, janelas = [], []
    base = dict(rotulo=meta["rotulo"], arquivo=csv.name, status=meta["status"], rpm_nk=meta["rpm_anotada"],
                rpm_tac=rpm_tac, data=meta["data_hora"][:10], obs=meta.get("obs", ""))
    for canal in CANAIS:
        x = d[f"{canal}_g"].to_numpy()
        reg = dict(base, canal=canal, rms_g=float(np.sqrt(np.mean((x - x.mean()) ** 2))))
        for k in HARMONICOS:
            c = vetor(x, theta, k)
            reg[f"re{k}"], reg[f"im{k}"] = c.real, c.imag
        cs = []
        for i in range(0, len(x) - n + 1, n):
            xj, tj = x[i:i + n], theta[i:i + n]
            if np.isnan(tj).any():
                continue
            lin = dict(base, canal=canal, janela=i // n, rms_g=float(np.sqrt(np.mean((xj - xj.mean()) ** 2))),
                       curtose=float(pd.Series(xj).kurt()))
            for k in HARMONICOS:
                c = vetor(xj, tj, k)
                lin[f"re{k}"], lin[f"im{k}"] = c.real, c.imag
            if k == HARMONICOS[0]:
                pass
            cs.append(complex(lin["re1"], lin["im1"]))
            janelas.append(lin)
        cs = np.array(cs)
        m = cs.mean()
        reg["disp_amp"] = float(np.std(np.abs(cs)) / abs(m))
        reg["disp_fase_deg"] = float(np.degrees(np.std(np.angle(cs / m))))
        reg["n_pulsos"] = len(tp)
        estados.append(reg)
    return estados, janelas


def main(imprimir=True):
    est, jan = [], []
    for csv in sorted(p for p in SINAIS.rglob("*.csv") if p.with_suffix(".json").is_file()):
        e, j = processar(csv)
        est += e
        jan += j
    df = pd.DataFrame(est)
    SAIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA / "vetores_estados.csv", index=False)
    pd.DataFrame(jan).to_csv(SAIDA / "vetores_janelas.csv", index=False)
    if not imprimir:
        return
    df["amp1_g"] = np.hypot(df.re1, df.im1)
    df["fase1_deg"] = np.degrees(np.arctan2(df.im1, df.re1)) % 360
    pd.set_option("display.width", 200)
    print(df[["rotulo", "canal", "status", "rpm_nk", "amp1_g", "fase1_deg", "disp_amp", "disp_fase_deg", "rms_g"]].round(4).to_string())


if __name__ == "__main__":
    main()
