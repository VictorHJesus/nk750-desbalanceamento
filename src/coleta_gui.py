import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from coleta_teste import bordas_tacometro, rpm_do_tacometro

RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "coletas"
CONFIG = Path(__file__).with_name("coleta_gui_config.json")

TIPOS = ["Acelerômetro", "Tacômetro", "Desligado"]
ACEL, TACO, OFF = TIPOS
CLOCK_HZ = 51200.0
JANELA_BUF_S = 4.0
JANELA_PLOT_S = 2.0
EXCITACAO_IEPE_A = 0.002
RPM_MIN, RPM_MAX = 800, 2000

PADRAO = {
    "fs_n": 5,
    "duracao": 30.0,
    "rotulo": "teste",
    "rpm": "",
    "rep": 1,
    "obs": "",
    "massa": "",
    "raio": "",
    "angulo": "",
    "plano": "",
    "furos": "",
    "ciclo": "C01",
    "etapa": "A",
    "canais": [
        {"tipo": ACEL, "descricao": "viga do pedestal 1 (sensor de carga 1)", "serie": "", "sens": "10", "iepe": True, "acopl": "AC", "ppr": "1"},
        {"tipo": ACEL, "descricao": "viga do pedestal 2 (sensor de carga 2)", "serie": "", "sens": "10", "iepe": True, "acopl": "AC", "ppr": "1"},
        {"tipo": OFF, "descricao": "(livre)", "serie": "", "sens": "10", "iepe": False, "acopl": "AC", "ppr": "1"},
        {"tipo": TACO, "descricao": "tacômetro", "serie": "", "sens": "10", "iepe": False, "acopl": "AC", "ppr": "1"},
    ],
}


def num(texto, nome):
    try:
        return float(str(texto).replace(",", "."))
    except ValueError:
        raise ValueError(f"{nome}: valor inválido ({texto!r})")


def curto(txt, n=34):
    return txt if len(txt) <= n else txt[: n - 1] + "…"


def fmt_hz(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def carregar_config():
    cfg = json.loads(json.dumps(PADRAO))
    try:
        salvo = json.loads(CONFIG.read_text(encoding="utf-8"))
        for k, v in salvo.items():
            if k == "canais":
                for i, c in enumerate(v[:4]):
                    cfg["canais"][i].update(c)
            elif k in cfg:
                cfg[k] = v
    except (OSError, ValueError):
        pass
    return cfg


PERFIL = Path(__file__).with_name("perfil_sessao.json")
PERFIL_CAMPOS = [
    ("operador", "Operador"),
    ("massa_rotor_kg", "Massa do rotor (kg)"),
    ("grau_G", "Grau G escolhido (mm/s, opcional)"),
    ("e_mar_gmm_kg", "e_mar da NK750 (g·mm/kg)"),
    ("dist_mancais_mm", "Distância entre mancais (mm)"),
    ("pos_planos_mm", "Posição dos planos de massa (mm)"),
    ("relacao_polias", "Relação das polias (D motora / D movida)"),
    ("sentido_rotacao", "Sentido de rotação (visto do lado do motor)"),
    ("referencia_angular", "Referência angular 0° (onde está marcada)"),
    ("fixacao_sensores", "Fixação dos sensores"),
    ("balanca", "Balança (modelo e resolução)"),
    ("paquimetro", "Paquímetro (resolução)"),
    ("temperatura_C", "Temperatura da sala (°C)"),
    ("umidade_pct", "Umidade (%)"),
    ("notas", "Notas da sessão"),
]
PERFIL_PADRAO = {k: "" for k, _ in PERFIL_CAMPOS}
PERFIL_PADRAO["e_mar_gmm_kg"] = "2"

INDICE_COLUNAS = [
    "arquivo", "data_hora", "rotulo", "repeticao", "status", "motivo_descarte", "rpm_anotada", "rpm_tacometro", "rpm_estimado_espectro",
    "massa_g", "raio_mm", "angulo_deg", "plano", "remontada", "U_gmm", "e_gmm_kg", "U_sobre_U_mar", "U_sobre_U_per",
    "fs_real_hz", "duracao_s", "rms_ai0_g", "rms_ai1_g", "rms_ai2_g", "alertas", "operador", "temperatura_C",
    "umidade_pct", "obs", "csv_sha256",
    "nk700_status", "nk700_rpm", "nk700_amplitude", "nk700_fase_deg", "nk700_massa_g", "nk700_angulo_deg",
    "nk700_residual_g", "nk700_limite_g",
]
NK700_CAMPOS = [("rpm", "rpm no software"), ("amplitude", "Amplitude (aba Sem Massa)"), ("fase_deg", "Fase ° (aba Sem Massa)"),
                ("massa_g", "Massa de correção (g)"), ("angulo_deg", "Ângulo de correção (°)"),
                ("residual_g", "Atual / residual (g)"), ("limite_g", "Limite (g)")]
NK700_STATUS = ["balanceado (dentro do limite ISO)", "fora do limite", "sem leitura"]


def carregar_perfil():
    perfil = dict(PERFIL_PADRAO)
    try:
        perfil.update({k: str(v) for k, v in json.loads(PERFIL.read_text(encoding="utf-8")).items() if k in perfil})
    except (OSError, ValueError):
        pass
    return perfil


def salvar_perfil(perfil):
    PERFIL.write_text(json.dumps(perfil, ensure_ascii=False, indent=2), encoding="utf-8")


def num_ou_none(txt):
    try:
        return float(str(txt).replace(",", ".")) if str(txt).strip() else None
    except ValueError:
        return None


def rpm_por_espectro(x, fs, f_min=8.0, f_max=50.0):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if x.size < 256:
        return None
    esp = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1 / fs)
    idx = np.where((f >= f_min) & (f <= min(f_max, 0.4 * fs)))[0]
    if idx.size < 3:
        return None
    k = idx[np.argmax(esp[idx])]
    d = 0.0
    if 0 < k < esp.size - 1:
        a, b, c = np.log(esp[k - 1] + 1e-30), np.log(esp[k] + 1e-30), np.log(esp[k + 1] + 1e-30)
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
    return {"rpm": float((f[k] + d * (f[1] - f[0])) * 60.0), "razao_pico_mediana": float(esp[k] / (np.median(esp[idx]) + 1e-30))}


def amplitude_1x(x, fs, rpm_ref, tol_hz=0.5):
    if not rpm_ref:
        return None
    w = np.hanning(len(x))
    a = np.abs(np.fft.rfft(x * w)) * 2 / np.sum(w)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    m = (f >= rpm_ref / 60.0 - tol_hz) & (f <= rpm_ref / 60.0 + tol_hz)
    return float(a[m].max()) if m.any() else None


def controle_qualidade(canais_ativos, dados, fs, tacho_res, rpm_ref):
    qc = {"canais": {}, "alertas": []}
    estimativas = []
    for (idx, c), linha in zip(canais_ativos, dados):
        if c["tipo"] != ACEL:
            continue
        lim = 4.9 / (c["sens_v"] / 1000.0)
        x = linha - linha.mean()
        rms = float(np.sqrt(np.mean(x**2)))
        sat = float(np.mean(np.abs(linha) >= 0.98 * lim))
        est = rpm_por_espectro(linha, fs)
        if est:
            estimativas.append(est["rpm"])
        qc["canais"][f"ai{idx}"] = {
            "rms_g": rms,
            "pico_g": float(np.abs(linha).max()),
            "offset_dc_g": float(linha.mean()),
            "fracao_saturada": sat,
            "rpm_estimado_espectro": est["rpm"] if est else None,
            "razao_pico_mediana": est["razao_pico_mediana"] if est else None,
            "amp_1x_g": amplitude_1x(x, fs, rpm_ref),
        }
        if rms < 1e-4:
            qc["alertas"].append(f"AI{idx}: canal parado ou desconectado (RMS < 0,0001 g)")
        if sat > 0:
            qc["alertas"].append(f"AI{idx}: saturacao em {100 * sat:.2f}% das amostras")
    rpm_est = float(np.median(estimativas)) if estimativas else None
    qc["rpm_estimado_espectro"] = rpm_est
    if tacho_res is not None and not tacho_res["rpm"]:
        qc["alertas"].append("tacometro sem pulsos")
    elif tacho_res is not None and tacho_res["rpm"] and tacho_res["rpm_desvio"] and tacho_res["rpm_desvio"] > 0.02 * tacho_res["rpm"]:
        qc["alertas"].append(f"rpm instavel durante a coleta (desvio {tacho_res['rpm_desvio']:.1f} rpm)")
    if rpm_ref is None:
        qc["alertas"].append("rpm nao informado (use a estimativa do espectro apenas como conferencia)")
    elif rpm_est and abs(rpm_est - rpm_ref) / rpm_ref > 0.05:
        qc["alertas"].append(f"rpm estimado pelo espectro ({rpm_est:.0f}) difere mais de 5% do rpm de referencia ({rpm_ref:.0f}); com o rotor balanceado o 1x some e o estimador pode pegar um harmonico: confira amp_1x_g")
    return qc, rpm_est


def ciclo_do_rotulo(rotulo):
    import re

    m = re.fullmatch(r"(.+)[ABC]", (rotulo or "").strip())
    return m.group(1) if m else (rotulo or "").strip()


def resultante_furos(txt, raio_mm=100.0, n_furos=16):
    import cmath, math
    u = 0j
    for item in filter(None, (t.strip() for t in txt.split(";"))):
        try:
            k, m = item.split(":")
            u += float(m.replace(",", ".")) * cmath.exp(1j * math.radians(360.0 / n_furos * int(k)))
        except ValueError:
            raise ValueError(f"'{item}' (use furo:massa_g)")
    mag = abs(u)
    return mag, (math.degrees(cmath.phase(u)) % 360.0 if mag > 1e-9 else 0.0)


def calcular_condicao(p, perfil, rpm_ref):
    m, r = p["massa_g"], p["raio_mm"]
    u = m * r if (m is not None and r is not None) else None
    m_rotor = num_ou_none(perfil["massa_rotor_kg"])
    g_alvo = num_ou_none(perfil["grau_G"])
    e_mar = num_ou_none(perfil["e_mar_gmm_kg"])
    u_mar = e_mar * m_rotor if (e_mar is not None and m_rotor) else None
    u_per = 9549.0 * g_alvo * m_rotor / rpm_ref if (g_alvo and m_rotor and rpm_ref) else None
    cond = {
        "massa_g": m, "raio_mm": r, "angulo_deg": p["angulo_deg"], "plano": p["plano"], "remontada": p["remontada"],
        "U_gmm": u,
        "e_gmm_kg": u / m_rotor if (u is not None and m_rotor) else None,
        "U_sobre_U_mar": u / u_mar if (u is not None and u_mar) else None,
        "U_sobre_U_per": u / u_per if (u is not None and u_per) else None,
    }
    ref = {"massa_rotor_kg": m_rotor, "grau_G": g_alvo, "e_mar_gmm_kg": e_mar, "U_mar_gmm": u_mar, "U_per_gmm": u_per}
    return cond, ref


def info_ambiente(modulo):
    import platform
    from importlib import metadata

    amb = {"python": platform.python_version(), "so": platform.platform(), "computador": platform.node()}
    try:
        amb["nidaqmx_python"] = metadata.version("nidaqmx")
    except Exception:
        pass
    if modulo:
        try:
            from nidaqmx.system import Device, System

            v = System.local().driver_version
            amb["driver_nidaqmx"] = f"{v.major_version}.{v.minor_version}.{v.update_version}"
            dev = Device(modulo)
            amb["modulo_produto"], amb["modulo_serie"] = dev.product_type, str(dev.dev_serial_num)
            try:
                chassi = dev.compact_daq_chassis_device
                amb["chassi_produto"], amb["chassi_serie"] = chassi.product_type, str(chassi.dev_serial_num)
            except Exception:
                pass
        except Exception:
            pass
    return amb


def linha_indice(meta, arquivo):
    cond, perfil, qc = meta.get("condicao") or {}, meta.get("perfil") or {}, meta.get("qc") or {}
    rms = lambda ch: (qc.get("canais", {}).get(ch) or {}).get("rms_g")
    return {
        "arquivo": arquivo, "data_hora": meta["data_hora"], "rotulo": meta["rotulo"], "repeticao": meta["repeticao"],
        "status": meta.get("status", "valida"), "motivo_descarte": meta.get("motivo_descarte", ""), "rpm_anotada": meta["rpm_anotada"], "rpm_tacometro": meta["rpm_tacometro"],
        "rpm_estimado_espectro": meta.get("rpm_estimado_espectro"),
        "massa_g": cond.get("massa_g"), "raio_mm": cond.get("raio_mm"), "angulo_deg": cond.get("angulo_deg"),
        "plano": cond.get("plano"), "remontada": cond.get("remontada"), "U_gmm": cond.get("U_gmm"),
        "e_gmm_kg": cond.get("e_gmm_kg"), "U_sobre_U_mar": cond.get("U_sobre_U_mar"), "U_sobre_U_per": cond.get("U_sobre_U_per"),
        "fs_real_hz": meta["fs_real_hz"], "duracao_s": meta["duracao_s"],
        "rms_ai0_g": rms("ai0"), "rms_ai1_g": rms("ai1"), "rms_ai2_g": rms("ai2"),
        "alertas": " | ".join(qc.get("alertas", [])), "operador": perfil.get("operador"),
        "temperatura_C": perfil.get("temperatura_C"), "umidade_pct": perfil.get("umidade_pct"),
        "obs": meta["obs"], "csv_sha256": meta.get("csv_sha256"),
        **{"nk700_" + k: v for k, v in (meta.get("nk700") or {}).items()},
    }


def gravar_indice(caminho, linha):
    import csv

    if caminho.exists():
        with open(caminho, newline="", encoding="utf-8") as f:
            cab = next(csv.reader(f), [])
        if cab != INDICE_COLUNAS:
            atualizar_indice(caminho, "")
    novo = not caminho.exists()
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDICE_COLUNAS)
        if novo:
            w.writeheader()
        w.writerow(linha)


def atualizar_indice(caminho, arquivo, **campos):
    import csv

    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for ln in linhas:
        if ln["arquivo"] == arquivo:
            ln.update({k: v for k, v in campos.items() if k in INDICE_COLUNAS})
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDICE_COLUNAS)
        w.writeheader()
        w.writerows(linhas)


class LeitorNI:

    def __init__(self, modulo, canais, fs, finito, n_total):
        self.modulo, self.canais, self.fs, self.finito, self.n_total = modulo, canais, fs, finito, n_total

    def __enter__(self):
        import nidaqmx
        from nidaqmx.constants import (
            AccelSensitivityUnits,
            AccelUnits,
            AcquisitionType,
            Coupling,
            ExcitationSource,
            VoltageUnits,
        )

        self.task = nidaqmx.Task()
        try:
            for idx, c in self.canais:
                fisico = f"{self.modulo}/ai{idx}"
                if c["tipo"] == ACEL:
                    sens = c["sens_v"]
                    lim = 4.9 / (sens / 1000.0)
                    canal = self.task.ai_channels.add_ai_accel_chan(
                        fisico,
                        min_val=-lim,
                        max_val=lim,
                        units=AccelUnits.G,
                        sensitivity=sens,
                        sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                        current_excit_source=ExcitationSource.INTERNAL if c["iepe"] else ExcitationSource.NONE,
                        current_excit_val=EXCITACAO_IEPE_A,
                    )
                    canal.ai_coupling = Coupling.AC if c["acopl"] == "AC" else Coupling.DC
                else:
                    canal = self.task.ai_channels.add_ai_voltage_chan(
                        fisico, min_val=-5.0, max_val=5.0, units=VoltageUnits.VOLTS
                    )
                    canal.ai_coupling = Coupling.AC
            if self.finito:
                self.task.timing.cfg_samp_clk_timing(
                    rate=self.fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=self.n_total
                )
            else:
                self.task.timing.cfg_samp_clk_timing(
                    rate=self.fs, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(self.fs * 10)
                )
            self.fs_real = float(self.task.timing.samp_clk_rate)
        except Exception:
            self.task.close()
            raise
        return self

    def ler(self, n):
        dado = self.task.read(number_of_samples_per_channel=n, timeout=15.0)
        return np.array(dado, dtype=float).reshape(len(self.canais), n)

    def __exit__(self, *a):
        self.task.close()


class LeitorSimulado:

    def __init__(self, canais, fs, rpm):
        self.canais, self.fs, self.rpm, self.i = canais, fs, rpm, 0
        self.fs_real = fs

    def __enter__(self):
        return self

    def ler(self, n):
        t = (self.i + np.arange(n)) / self.fs
        self.i += n
        f = self.rpm / 60.0
        saida = []
        for k, (idx, c) in enumerate(self.canais):
            if c["tipo"] == ACEL:
                amp = 0.05 * (1 + 0.3 * k) * (self.rpm / 1000.0) ** 2
                x = amp * np.sin(2 * np.pi * f * t + k) + 0.3 * amp * np.sin(2 * np.pi * 2 * f * t) + np.random.normal(0, 0.003, n)
            else:
                x = (np.mod(t * f * c["ppr_v"], 1) < 0.12) * 3.3 + np.random.normal(0, 0.02, n)
            saida.append(x)
        time.sleep(n / self.fs)
        return np.array(saida)

    def __exit__(self, *a):
        pass


def trabalhador(leitor, gravar, n_total, fila, parar):
    try:
        with leitor:
            fs = leitor.fs_real
            fila.put(("inicio", fs))
            passo = max(1, int(fs * 0.25))
            blocos, total = [], 0
            while not parar.is_set():
                n = passo if not gravar else min(passo, n_total - total)
                if n <= 0:
                    break
                bloco = leitor.ler(n)
                total += n
                if gravar:
                    blocos.append(bloco)
                fila.put(("bloco", bloco, total))
        if gravar and not parar.is_set():
            fila.put(("fim", np.hstack(blocos), fs))
        else:
            fila.put(("parado",))
    except Exception as e:
        fila.put(("erro", f"{type(e).__name__}: {e}"))


class App(tk.Tk):
    def __init__(self, simulado=False, auto=False):
        super().__init__()
        self.title("Coleta de vibração: NI 9234" + (" [SIMULADO]" if simulado else ""))
        self.geometry(f"{min(1500, self.winfo_screenwidth() - 40)}x{min(900, self.winfo_screenheight() - 110)}+10+10")
        self.simulado, self.auto = simulado, auto
        self.cfg = carregar_config()
        self.perfil = carregar_perfil()
        self.ambiente = {}
        self.fila, self.parar, self.thread = queue.Queue(), threading.Event(), None
        self.modo = None
        self.canais_ativos, self.buf, self.fs_atual = [], None, None
        self.ultimo_csv, self.ultimo_desenho, self.sujo = None, 0.0, False
        self.n_total = 0
        self._montar()
        self._atualizar_dispositivos()
        self._atualizar_info()
        self._atualizar_cond()
        self._ciclo()
        self._ctl_iniciar()
        self.protocol("WM_DELETE_WINDOW", self._fechar)

    def _montar(self):
        esq = ttk.Frame(self, padding=8)
        esq.pack(side="left", fill="y")
        topo = ttk.Frame(esq)
        topo.pack(fill="x")
        av = ttk.Frame(esq)
        dir_ = ttk.Frame(self, padding=4)
        dir_.pack(side="left", fill="both", expand=True)

        f = ttk.LabelFrame(av, text="Dispositivo", padding=6)
        f.pack(fill="x")
        self.v_disp = tk.StringVar()
        self.cb_disp = ttk.Combobox(f, textvariable=self.v_disp, state="readonly", width=34)
        self.cb_disp.pack(side="left")
        ttk.Button(f, text="Atualizar", command=self._atualizar_dispositivos).pack(side="left", padx=4)

        f = ttk.LabelFrame(av, text="Aquisição", padding=6)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="Taxa de amostragem:").grid(row=0, column=0, sticky="w")
        self.taxas = [f"{fmt_hz(CLOCK_HZ / n)} Hz  (n={n})" for n in range(1, 32)]
        self.v_fs = tk.StringVar()
        self.cb_fs = ttk.Combobox(f, textvariable=self.v_fs, values=self.taxas, state="readonly", width=20)
        self.cb_fs.current(int(self.cfg["fs_n"]) - 1)
        self.cb_fs.grid(row=0, column=1, sticky="w", padx=4)
        self.cb_fs.bind("<<ComboboxSelected>>", lambda e: self._atualizar_info())
        ttk.Label(f, text="Duração (s):").grid(row=1, column=0, sticky="w", pady=3)
        self.v_dur = tk.StringVar(value=str(self.cfg["duracao"]))
        e = ttk.Entry(f, textvariable=self.v_dur, width=8)
        e.grid(row=1, column=1, sticky="w", padx=4)
        self.v_dur.trace_add("write", lambda *a: self._atualizar_info())
        self.l_info = ttk.Label(f, text="", foreground="#444", wraplength=760, justify="left")
        self.l_info.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        f = ttk.LabelFrame(av, text="Canais (NI 9234)", padding=6)
        f.pack(fill="x")
        cab = ["Canal", "Tipo", "Descrição / posição", "Nº série", "Sens. (mV/g)", "IEPE", "Acopl.", "Pulsos/volta"]
        for j, t in enumerate(cab):
            ttk.Label(f, text=t, font=("", 8, "bold")).grid(row=0, column=j, padx=2, sticky="w")
        self.lin = []
        for i in range(4):
            c = self.cfg["canais"][i]
            v = {
                "tipo": tk.StringVar(value=c["tipo"]),
                "descricao": tk.StringVar(value=c["descricao"]),
                "serie": tk.StringVar(value=c["serie"]),
                "sens": tk.StringVar(value=str(c["sens"])),
                "iepe": tk.BooleanVar(value=bool(c["iepe"])),
                "acopl": tk.StringVar(value=c["acopl"]),
                "ppr": tk.StringVar(value=str(c["ppr"])),
            }
            w = {}
            ttk.Label(f, text=f"AI{i}").grid(row=i + 1, column=0, padx=2)
            w["tipo"] = ttk.Combobox(f, textvariable=v["tipo"], values=TIPOS, state="readonly", width=12)
            w["descricao"] = ttk.Entry(f, textvariable=v["descricao"], width=30)
            w["serie"] = ttk.Entry(f, textvariable=v["serie"], width=9)
            w["sens"] = ttk.Entry(f, textvariable=v["sens"], width=7)
            w["iepe"] = ttk.Checkbutton(f, variable=v["iepe"])
            w["acopl"] = ttk.Combobox(f, textvariable=v["acopl"], values=["AC", "DC"], state="readonly", width=4)
            w["ppr"] = ttk.Entry(f, textvariable=v["ppr"], width=6)
            for j, k in enumerate(["tipo", "descricao", "serie", "sens", "iepe", "acopl", "ppr"], start=1):
                w[k].grid(row=i + 1, column=j, padx=2, pady=2)
            w["tipo"].bind("<<ComboboxSelected>>", lambda e, i=i: self._tipo_mudou(i))
            self.lin.append((v, w))
            self._tipo_mudou(i, salvar=False)
        ttk.Label(
            f,
            text="Acelerômetro: IEPE + sensibilidade. Tacômetro: saída do condicionador 482C (que alimenta o LaserTach), ±5 V, AC, sem IEPE.",
            foreground="#444",
            wraplength=760,
            justify="left",
        ).grid(row=5, column=0, columnspan=8, sticky="w", pady=(2, 0))

        f = ttk.LabelFrame(av, text="Condição manual (use o painel acima)", padding=6)
        f.pack(fill="x", pady=3)
        self.v_rot = tk.StringVar(value=self.cfg["rotulo"])
        self.v_rpm = tk.StringVar(value=str(self.cfg["rpm"]))
        self.v_rep = tk.StringVar(value=str(self.cfg["rep"]))
        self.v_obs = tk.StringVar(value=self.cfg["obs"])
        self.v_massa = tk.StringVar(value=str(self.cfg["massa"]))
        self.v_raio = tk.StringVar(value=str(self.cfg["raio"]))
        self.v_ang = tk.StringVar(value=str(self.cfg["angulo"]))
        self.v_plano = tk.StringVar(value=str(self.cfg["plano"]))
        self.v_furos = tk.StringVar(value=str(self.cfg.get("furos", "")))
        self.v_remont = tk.BooleanVar(value=False)
        campos = [
            [("Rótulo (condição):", self.v_rot, 22), ("rpm anotado:", self.v_rpm, 8)],
            [("Repetição:", self.v_rep, 6), ("Observações:", self.v_obs, 26)],
            [("Massa de teste (g):", self.v_massa, 8), ("Raio (mm):", self.v_raio, 8)],
            [("Ângulo (°):", self.v_ang, 8), ("Plano:", self.v_plano, 8)],
        ]
        for r, linha in enumerate(campos):
            for j, (txt, var, larg) in enumerate(linha):
                ttk.Label(f, text=txt).grid(row=r, column=2 * j, sticky="w", pady=2, padx=(8 if j else 0, 0))
                ttk.Entry(f, textvariable=var, width=larg).grid(row=r, column=2 * j + 1, sticky="w", padx=4)
        ttk.Checkbutton(f, text="Massa remontada desde a coleta anterior", variable=self.v_remont).grid(row=5, column=0, columnspan=4, sticky="w")
        for var in (self.v_massa, self.v_raio, self.v_furos):
            var.trace_add("write", lambda *a: self._atualizar_cond())

        self._montar_topo(topo)

        self.prog = ttk.Progressbar(esq, maximum=100)
        self.prog.pack(fill="x", pady=2)
        self.log = tk.Text(esq, height=5, width=70, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self.b_avanc = ttk.Button(esq, text="▸ Configuração avançada (canais, taxa, perfil, condição manual)", command=lambda: self._alternar_avanc(av))
        self.b_avanc.pack(fill="x", pady=(4, 0))
        self.av = av

        top = ttk.Frame(dir_)
        top.pack(fill="x")
        ttk.Label(top, text="Monitor ao vivo:").pack(side="left")
        self.v_vista = tk.StringVar(value="Tempo")
        for t in ("Tempo", "Espectro"):
            ttk.Radiobutton(top, text=t, value=t, variable=self.v_vista, command=lambda: setattr(self, "sujo", True)).pack(side="left", padx=4)
        self.fig = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=dir_)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._grafico_vazio("Preencha os parâmetros e clique em «Coletar e salvar».\nO gráfico da coleta aparece aqui.")

    def _tipo_mudou(self, i, salvar=True):
        v, w = self.lin[i]
        t = v["tipo"].get()
        acel, taco = t == ACEL, t == TACO
        w["descricao"].config(state="normal" if t != OFF else "disabled")
        w["serie"].config(state="normal" if acel else "disabled")
        w["sens"].config(state="normal" if acel else "disabled")
        w["iepe"].config(state="normal" if acel else "disabled")
        w["acopl"].config(state="readonly" if acel else "disabled")
        w["ppr"].config(state="normal" if taco else "disabled")
        if taco:
            v["iepe"].set(False)
            v["acopl"].set("AC")
            if v["descricao"].get().strip() in ("", "mancal 1 (lado do motor), radial horizontal"):
                v["descricao"].set("tacômetro")
        elif acel:
            v["iepe"].set(True)
            v["acopl"].set("AC")
        self._atualizar_info()

    def _atualizar_info(self):
        try:
            n = self.cb_fs.current() + 1
            fs = CLOCK_HZ / n
            dur = num(self.v_dur.get(), "Duração")
            ncan = sum(1 for v, _ in self.lin if v["tipo"].get() != OFF)
            amostras = int(fs * dur)
            mb = amostras * (ncan + 1) * 9.5 / 1e6
            banda = 0.45 * fs
            f_max = RPM_MAX / 60
            self.l_info.config(
                text=f"{amostras} amostras/canal · CSV ≈ {mb:.0f} MB · resolução em frequência = {1/dur:.3f} Hz\n"
                f"Banda útil ≈ 0,45·fs = {banda:.0f} Hz (valor aproximado do filtro do 9234) = {banda/f_max:.0f}× a rotação máxima prevista "
                f"({RPM_MIN}–{RPM_MAX} rpm = {RPM_MIN/60:.1f}–{f_max:.1f} Hz)"
            )
        except (ValueError, AttributeError):
            pass

    def _atualizar_cond(self):
        try:
            m, r = num_ou_none(self.v_massa.get()), num_ou_none(self.v_raio.get())
            if self.v_furos.get().strip():
                try:
                    m, ang_res = resultante_furos(self.v_furos.get(), r or 100.0)
                    r = r or 100.0
                    self.l_cond.config(text=f"Resultante dos furos: {m:.3f} g a {ang_res:.1f}° (r = {r:g} mm)")
                    self.l_cond.config(text=self.l_cond.cget("text") + f" · U = {m * r:.1f} g·mm")
                except ValueError as e:
                    self.l_cond.config(text=f"Furos inválidos: {e}")
                return
            if m is None or r is None:
                self.l_cond.config(text=" ")
                return
            txt = f"U = m·r = {m * r:.1f} g·mm"
            m_rotor, e_mar = num_ou_none(self.perfil["massa_rotor_kg"]), num_ou_none(self.perfil["e_mar_gmm_kg"])
            if m_rotor:
                txt += f" · e = {m * r / m_rotor:.2f} g·mm/kg"
                if e_mar:
                    txt += f" · {m * r / (e_mar * m_rotor):.1f}× U_mar da NK750"
            else:
                txt += " (informe a massa do rotor no perfil para ver e e U/U_mar)"
            self.l_cond.config(text=txt)
        except (AttributeError, tk.TclError):
            pass

    def _abrir_perfil(self):
        win = tk.Toplevel(self)
        win.title("Perfil da sessão")
        win.transient(self)
        vars_ = {}
        for r, (k, rotulo) in enumerate(PERFIL_CAMPOS):
            ttk.Label(win, text=rotulo).grid(row=r, column=0, sticky="w", padx=6, pady=2)
            vars_[k] = tk.StringVar(value=self.perfil[k])
            ttk.Entry(win, textvariable=vars_[k], width=48).grid(row=r, column=1, padx=6, pady=2)

        def salvar():
            self.perfil = {k: v.get().strip() for k, v in vars_.items()}
            try:
                salvar_perfil(self.perfil)
            except OSError as e:
                messagebox.showerror("Perfil", f"Não foi possível salvar: {e}")
                return
            self._atualizar_cond()
            self._log("Perfil da sessão salvo.")
            win.destroy()

        ttk.Button(win, text="Salvar", command=salvar).grid(row=len(PERFIL_CAMPOS), column=0, columnspan=2, pady=8)

    def _descartar_ultima(self):
        if not self.ultimo_csv:
            return
        motivo = simpledialog.askstring("Descartar coleta", f"Motivo para descartar {self.ultimo_csv.name}:", parent=self)
        if motivo is None:
            return
        try:
            self._descartar(self.ultimo_csv, motivo)
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("Descartar", f"Não foi possível atualizar: {e}")

    def _descartar(self, csv, motivo):
        js = Path(csv).with_suffix(".json")
        meta = json.loads(js.read_text(encoding="utf-8-sig"))
        meta["status"], meta["motivo_descarte"] = "descartada", motivo.strip()
        js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        atualizar_indice(self._caminho_indice(), Path(csv).name, status="descartada", motivo_descarte=motivo.strip())
        self._log(f"Descartada (arquivos mantidos): {Path(csv).name}")

    def _ctl_iniciar(self):
        self._ctl_dir = Path(os.environ.get("COLETA_CTL", Path(__file__).resolve().parent / "_ctl"))
        self._ctl_dir.mkdir(parents=True, exist_ok=True)
        self._ctl_seq, self._ctl_erro, self._ctl_ultimo_estado = "", "", ""
        self._ctl_poll()

    def _ctl_poll(self):
        try:
            cmd = self._ctl_dir / "cmd.json"
            if cmd.exists():
                try:
                    dados = json.loads(cmd.read_text(encoding="utf-8-sig"))
                except ValueError:
                    dados = None
                if dados is not None:
                    os.replace(cmd, self._ctl_dir / "cmd.feito.json")
                    self._ctl_erro = ""
                    try:
                        self._ctl_exec(dados)
                    except Exception as e:
                        self._ctl_erro = f"{type(e).__name__}: {e}"
                    self._ctl_seq = str(dados.get("id", "")) if isinstance(dados, dict) else ""
            self._ctl_escrever_estado()
        except Exception:
            pass
        self.after(250, self._ctl_poll)

    def _ctl_exec(self, c):
        rodando = bool(self.thread and self.thread.is_alive())
        if rodando and any(k in c for k in ("coletar", "novo_ciclo", "ciclo", "etapa", "furos", "obs")):
            raise RuntimeError("coleta em andamento; comando ignorado")
        if c.get("novo_ciclo"):
            self._novo_ciclo()
        if "ciclo" in c:
            self.v_ciclo.set(str(c["ciclo"]))
        if "etapa" in c:
            self.v_etapa.set(str(c["etapa"]).upper())
            self._etapa_escolhida(self.v_etapa.get())
        if "rpm" in c:
            self.v_rpm.set(str(c["rpm"]))
        if "furos" in c:
            self.v_furos.set(str(c["furos"]))
        if "obs" in c:
            self.v_obs.set(str(c["obs"]))
        if "nk" in c:
            nk = dict(c["nk"])
            for k in ("amplitude", "fase_deg", "massa_g", "angulo_deg", "residual_g", "limite_g"):
                self.v_nk[k].set("")
            self.v_nk_status.set(str(nk.pop("status", "")))
            for k, v in nk.items():
                if k == "rpm":
                    self.v_rpm.set(str(v))
                elif k in self.v_nk:
                    self.v_nk[k].set(str(v))
        if "descartar" in c:
            d = c["descartar"]
            if isinstance(d, str):
                d = {"motivo": d}
            alvo = None
            if d.get("arquivo"):
                alvo = next(iter(PASTA_DADOS.glob(f"coleta_*/{d['arquivo']}")), None)
            elif self.ultimo_csv:
                alvo = self.ultimo_csv
            if alvo is None:
                raise FileNotFoundError("coleta a descartar nao encontrada")
            self._descartar(alvo, d.get("motivo", "descartada por comando"))
            self._atualizar_alvos()
        if c.get("salvar_nk"):
            alvo = c["salvar_nk"]
            col = [(js, m) for js, m in self._coletas_do_ciclo() if m.get("status", "valida") != "descartada"]
            if isinstance(alvo, str):
                col = [(js, m) for js, m in col if m.get("rotulo", "").upper() == alvo.upper()]
            if not col:
                raise FileNotFoundError("nenhuma coleta valida do ciclo para gravar o NK700")
            js_alvo = col[-1][0]
            self._atualizar_alvos(marcar_ultima=False)
            for js, v in self.alvos.items():
                v.set(js == js_alvo)
            self._salvar_nk700()
            for k in ("amplitude", "fase_deg", "massa_g", "angulo_deg", "residual_g", "limite_g"):
                self.v_nk[k].set("")
            self.v_nk_status.set("")
        if c.get("coletar"):
            p = self._parametros()
            if p["rpm"] is None:
                raise ValueError("rpm vazio")
            if self.v_etapa.get() == "A" and self.v_furos.get().strip():
                raise ValueError("etapa A com parafusos preenchidos")
            self._iniciar("coleta")

    def _ctl_escrever_estado(self):
        rodando = bool(self.thread and self.thread.is_alive())
        ult = {}
        if self.ultimo_csv:
            try:
                m = json.loads(self.ultimo_csv.with_suffix(".json").read_text(encoding="utf-8-sig"))
                ult = {"arquivo": self.ultimo_csv.name, "rotulo": m.get("rotulo"), "rpm_tacometro": m.get("rpm_tacometro"),
                       "status": m.get("status"), "alertas": (m.get("qc") or {}).get("alertas", []), "nk700": m.get("nk700")}
            except (OSError, ValueError):
                pass
        est = {
            "id_ultimo_comando": self._ctl_seq, "erro": self._ctl_erro, "rodando": rodando,
            "progresso_pct": float(self.prog["value"]) if hasattr(self, "prog") else None,
            "ciclo": self.v_ciclo.get(), "etapa": self.v_etapa.get(), "rpm": self.v_rpm.get(), "furos": self.v_furos.get(),
            "obs": self.v_obs.get(), "rotulo": self.v_rot.get(), "proxima_rep": self.v_rep.get(),
            "status_texto": self.l_status.cget("text"), "ultima_coleta": ult,
        }
        txt = json.dumps(est, ensure_ascii=False, indent=1)
        if txt != self._ctl_ultimo_estado:
            tmp = self._ctl_dir / "estado.tmp"
            tmp.write_text(txt, encoding="utf-8")
            os.replace(tmp, self._ctl_dir / "estado.json")
            self._ctl_ultimo_estado = txt

    def _alternar_avanc(self, av):
        if av.winfo_ismapped():
            av.pack_forget()
            self.b_avanc.config(text=self.b_avanc.cget("text").replace("▾", "▸"))
        else:
            av.pack(fill="x", before=self.prog)
            self.b_avanc.config(text=self.b_avanc.cget("text").replace("▸", "▾"))

    def _montar_topo(self, topo):
        for v in (self.v_massa, self.v_raio, self.v_ang):
            v.set("")
        self.v_ciclo = tk.StringVar(value=str(self.cfg.get("ciclo", "C01")))
        self.v_etapa = tk.StringVar(value=str(self.cfg.get("etapa", "A")))
        f = ttk.LabelFrame(topo, text="1. Coleta", padding=6)
        f.pack(fill="x")
        ttk.Label(f, text="Ciclo:").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_ciclo, width=7).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(f, text="Novo ciclo", command=self._novo_ciclo).grid(row=0, column=2, padx=4)
        ttk.Label(f, text="rpm do software:").grid(row=0, column=3, sticky="e", padx=(14, 0))
        ttk.Entry(f, textvariable=self.v_rpm, width=8).grid(row=0, column=4, sticky="w", padx=4)
        ttk.Label(f, text="Etapa:").grid(row=1, column=0, sticky="w", pady=4)
        fe = ttk.Frame(f)
        fe.grid(row=1, column=1, columnspan=4, sticky="w")
        for letra, txt in (("A", "A  sem massa"), ("B", "B  massa de prova"), ("C", "C  correção aplicada")):
            ttk.Radiobutton(fe, text=txt, value=letra, variable=self.v_etapa, command=lambda l=letra: self._etapa_escolhida(l)).pack(side="left", padx=6)
        ttk.Label(f, text="Parafusos (furo:g):").grid(row=2, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_furos, width=30).grid(row=2, column=1, columnspan=3, sticky="we", padx=4)
        ttk.Label(f, text="ex.: 0:5.4   ou   3:4.66;11:2.1", foreground="#666").grid(row=2, column=4, sticky="w")
        ttk.Label(f, text="Observações:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(f, textvariable=self.v_obs, width=30).grid(row=3, column=1, columnspan=3, sticky="we", padx=4)
        self.l_rotulo = ttk.Label(f, text="", font=("", 11, "bold"))
        self.l_rotulo.grid(row=4, column=0, columnspan=5, sticky="w", pady=(4, 0))
        self.l_cond = ttk.Label(f, text=" ", foreground="#444", wraplength=520, justify="left")
        self.l_cond.grid(row=5, column=0, columnspan=5, sticky="w")
        for v in (self.v_ciclo, self.v_etapa, self.v_rep):
            v.trace_add("write", lambda *a: self._atualizar_rotulo())
        self._atualizar_rotulo()

        fb = ttk.Frame(topo)
        fb.pack(fill="x", pady=4)
        self.b_coletar = ttk.Button(fb, text="▶  COLETAR E SALVAR", command=lambda: self._iniciar("coleta"))
        self.b_coletar.pack(side="left", padx=2, ipady=6, ipadx=8)
        self.b_monitor = ttk.Button(fb, text="Monitorar", command=lambda: self._iniciar("monitor"))
        self.b_monitor.pack(side="left", padx=2)
        self.b_parar = ttk.Button(fb, text="Parar", command=self.parar.set, state="disabled")
        self.b_parar.pack(side="left", padx=2)
        self.b_descartar = ttk.Button(fb, text="Descartar última…", command=self._descartar_ultima, state="disabled")
        self.b_descartar.pack(side="left", padx=2)
        self.b_pasta = ttk.Button(fb, text="Pasta", command=self._abrir_pasta)
        self.b_pasta.pack(side="left", padx=2)
        ttk.Button(fb, text="Perfil…", command=self._abrir_perfil).pack(side="left", padx=2)
        self.l_status = ttk.Label(topo, text="Nenhuma coleta ainda.", wraplength=560, justify="left")
        self.l_status.pack(fill="x", pady=(0, 4))

        f = ttk.LabelFrame(topo, text="2. Leitura do software NK700 (rótulo do ML)", padding=6)
        f.pack(fill="x")
        self.v_nk_status = tk.StringVar(value="")
        ttk.Label(f, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.v_nk_status, values=NK700_STATUS, width=32).grid(row=0, column=1, columnspan=3, sticky="w", padx=4)
        self.v_nk = {k: tk.StringVar() for k, _ in NK700_CAMPOS}
        self.v_nk["rpm"] = self.v_rpm
        pos = {"residual_g": (1, 0), "limite_g": (1, 2), "massa_g": (2, 0), "angulo_deg": (2, 2), "amplitude": (3, 0), "fase_deg": (3, 2)}
        for k, nome in NK700_CAMPOS:
            if k == "rpm":
                continue
            r, c = pos[k]
            ttk.Label(f, text=nome + ":").grid(row=r, column=c, sticky="w", pady=2)
            ttk.Entry(f, textvariable=self.v_nk[k], width=9).grid(row=r, column=c + 1, sticky="w", padx=4)
        self.l_furo_sug = ttk.Label(f, text="", foreground="#444")
        self.l_furo_sug.grid(row=4, column=0, columnspan=4, sticky="w")
        self.v_nk["angulo_deg"].trace_add("write", lambda *a: self._furo_sugerido())
        ttk.Label(f, text="Gravar esta leitura em:", font=("", 9, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.f_alvos = ttk.Frame(f)
        self.f_alvos.grid(row=6, column=0, columnspan=4, sticky="w")
        self.alvos = {}
        ttk.Button(f, text="Salvar leitura", command=self._salvar_nk700).grid(row=7, column=3, sticky="e", pady=(4, 0))
        ttk.Label(f, text="Campos em branco não apagam o que já foi gravado.", foreground="#666").grid(row=7, column=0, columnspan=3, sticky="w")
        self.v_ciclo.trace_add("write", lambda *a: self._atualizar_alvos())
        self._atualizar_alvos()

    def _coletas_do_ciclo(self):
        ciclo = self.v_ciclo.get().strip().upper()
        out = []
        for js in sorted(PASTA_DADOS.glob("coleta_*/*.json"), key=lambda p: p.stat().st_mtime):
            try:
                m = json.loads(js.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if ciclo_do_rotulo(m.get("rotulo", "")).upper() == ciclo:
                out.append((js, m))
        return out

    def _atualizar_alvos(self, marcar_ultima=True):
        try:
            for w in self.f_alvos.winfo_children():
                w.destroy()
            self.alvos = {}
            col = self._coletas_do_ciclo()
            if not col:
                ttk.Label(self.f_alvos, text="(nenhuma coleta neste ciclo ainda)", foreground="#666").pack(anchor="w")
                return
            for i, (js, m) in enumerate(col):
                v = tk.BooleanVar(value=marcar_ultima and i == len(col) - 1)
                self.alvos[js] = v
                nk = m.get("nk700") or {}
                tem = [k for k in ("amplitude", "fase_deg", "massa_g", "residual_g") if nk.get(k) is not None]
                txt = f"{m.get('rotulo', '')} · nº {m.get('repeticao', '')}" + (f"  (já tem: {', '.join(tem)})" if tem else "")
                ttk.Checkbutton(self.f_alvos, text=txt, variable=v).pack(anchor="w")
            fb = ttk.Frame(self.f_alvos)
            fb.pack(anchor="w")
            ttk.Button(fb, text="Todas", width=7, command=lambda: [v.set(True) for v in self.alvos.values()]).pack(side="left")
            ttk.Button(fb, text="Nenhuma", width=8, command=lambda: [v.set(False) for v in self.alvos.values()]).pack(side="left", padx=4)
        except (AttributeError, tk.TclError):
            pass

    def _atualizar_rotulo(self):
        try:
            rot = self.v_ciclo.get().strip().upper() + self.v_etapa.get()
            self.v_rot.set(rot)
            self.l_rotulo.config(text=f"Vai salvar como:  {rot}  ·  coleta nº {self.v_rep.get()}")
        except (AttributeError, tk.TclError):
            pass

    def _etapa_escolhida(self, letra):
        if letra == "A":
            self.v_furos.set("")

    def _novo_ciclo(self):
        import re

        m = re.fullmatch(r"([A-Za-z]*)(\d+)", self.v_ciclo.get().strip())
        if m:
            self.v_ciclo.set(f"{m.group(1).upper()}{int(m.group(2)) + 1:0{len(m.group(2))}d}")
        self.v_etapa.set("A")
        self.v_furos.set("")
        self.v_nk_status.set("")
        for k in ("amplitude", "fase_deg", "massa_g", "angulo_deg", "residual_g", "limite_g"):
            self.v_nk[k].set("")

    def _furo_sugerido(self):
        try:
            a = float(self.v_nk["angulo_deg"].get().replace(",", ".")) % 360.0
            self.l_furo_sug.config(text=f"ângulo {a:g}° ≈ furo {round(a / 22.5) % 16} (se a numeração dos furos seguir o ângulo do software)")
        except ValueError:
            self.l_furo_sug.config(text="")

    def _salvar_nk700(self):
        alvo = [js for js, v in self.alvos.items() if v.get()]
        if not alvo:
            messagebox.showinfo("NK700", "Marque em qual(is) coleta(s) gravar a leitura.")
            return
        try:
            novo = {}
            if self.v_nk_status.get().strip():
                novo["status"] = self.v_nk_status.get().strip()
            for k, nome in NK700_CAMPOS:
                txt = self.v_nk[k].get().strip()
                if txt:
                    novo[k] = num(txt, nome)
            if not novo:
                messagebox.showinfo("NK700", "Nenhum campo preenchido.")
                return
            ciclo = self.v_ciclo.get().strip().upper()
            for js in alvo:
                meta = json.loads(js.read_text(encoding="utf-8"))
                blk = {**(meta.get("nk700") or {}), **novo, "ciclo": ciclo}
                meta["nk700"] = blk
                js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                atualizar_indice(self._caminho_indice(), js.with_suffix(".csv").name,
                                 **{"nk700_" + k: ("" if blk.get(k) is None else blk[k]) for k, _ in [("status", 0)] + NK700_CAMPOS})
            nomes = ", ".join(js.stem.split("_")[0] for js in alvo)
            self._log(f"NK700 gravado em: {nomes}")
            self.l_status.config(text=f"✔ Leitura do NK700 gravada em {nomes}.", foreground="#1a7f37")
            self._atualizar_alvos(marcar_ultima=False)
            for js in alvo:
                if js in self.alvos:
                    self.alvos[js].set(True)
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("NK700", str(e))

    def _caminho_indice(self):
        return PASTA_DADOS / ("indice_coletas_simulado.csv" if self.simulado else "indice_coletas.csv")

    def _atualizar_dispositivos(self):
        if self.simulado:
            self.cb_disp["values"] = ["(simulado)"]
            self.cb_disp.current(0)
            return
        try:
            from nidaqmx.system import System

            devs = [f"{d.name}: {d.product_type}" for d in System.local().devices if "9234" in d.product_type]
        except Exception as e:
            self._log(f"Não foi possível consultar o driver NI: {e}")
            devs = []
        self.cb_disp["values"] = devs
        if devs:
            self.cb_disp.current(0)
            self._log(f"NI 9234 encontrado: {devs[0]}")
        else:
            self.v_disp.set("")
            self._log("Nenhum NI 9234 encontrado. Conecte o USB e clique em Atualizar.")

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {msg}\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _parametros(self):
        canais = []
        for i, (v, _) in enumerate(self.lin):
            t = v["tipo"].get()
            if t == OFF:
                continue
            c = {"tipo": t, "descricao": v["descricao"].get().strip(), "serie": v["serie"].get().strip()}
            if t == ACEL:
                c["sens_v"] = num(v["sens"].get(), f"AI{i} sensibilidade")
                if c["sens_v"] <= 0:
                    raise ValueError(f"AI{i}: sensibilidade deve ser maior que zero")
                c["iepe"], c["acopl"] = bool(v["iepe"].get()), v["acopl"].get()
            else:
                c["ppr_v"] = int(num(v["ppr"].get(), f"AI{i} pulsos/volta"))
                if c["ppr_v"] < 1:
                    raise ValueError(f"AI{i}: pulsos por volta deve ser pelo menos 1")
            canais.append((i, c))
        if not canais:
            raise ValueError("Ative pelo menos um canal.")
        n = self.cb_fs.current() + 1
        dur = num(self.v_dur.get(), "Duração")
        if not 1 <= dur <= 600:
            raise ValueError("Duração deve estar entre 1 e 600 s.")
        rpm_txt = self.v_rpm.get().strip()
        p = {
            "canais": canais,
            "fs_n": n,
            "fs": CLOCK_HZ / n,
            "duracao": dur,
            "rotulo": self.v_rot.get().strip() or "teste",
            "rpm": num(rpm_txt, "rpm anotado") if rpm_txt else None,
            "rep": int(num(self.v_rep.get() or 1, "Repetição")),
            "obs": self.v_obs.get().strip(),
            "massa_g": None, "raio_mm": None, "angulo_deg": None,
            "plano": self.v_plano.get().strip(),
            "remontada": bool(self.v_remont.get()),
        }
        for chave, var, nome in (("massa_g", self.v_massa, "Massa de teste"), ("raio_mm", self.v_raio, "Raio"), ("angulo_deg", self.v_ang, "Ângulo")):
            txt = var.get().strip()
            if txt:
                p[chave] = num(txt, nome)
                if p[chave] < 0:
                    raise ValueError(f"{nome} não pode ser negativo.")
        furos_txt = self.v_furos.get().strip()
        if furos_txt:
            if p["raio_mm"] is None:
                p["raio_mm"] = 100.0
            p["massa_g"], p["angulo_deg"] = resultante_furos(furos_txt, p["raio_mm"])
            p["obs"] = (p["obs"] + " | " if p["obs"] else "") + "furos(furo:g)=" + furos_txt
        if p["massa_g"] is not None and p["raio_mm"] is None:
            raise ValueError("Informe o raio da massa de teste (ou deixe massa e raio vazios para a condição sem massa).")
        if p["angulo_deg"] is not None:
            p["angulo_deg"] %= 360.0
        if not self.simulado:
            disp = self.v_disp.get()
            if not disp:
                raise ValueError("Nenhum NI 9234 selecionado. Conecte o equipamento e clique em Atualizar.")
            p["modulo"] = disp.split(":")[0]
        return p

    def _salvar_config(self):
        if self.auto:
            return
        try:
            cfg = {
                "fs_n": self.cb_fs.current() + 1,
                "duracao": self.v_dur.get(),
                "rotulo": self.v_rot.get(),
                "rpm": self.v_rpm.get(),
                "rep": self.v_rep.get(),
                "obs": self.v_obs.get(),
                "massa": self.v_massa.get(),
                "raio": self.v_raio.get(),
                "angulo": self.v_ang.get(),
                "plano": self.v_plano.get(),
                "furos": self.v_furos.get(),
                "ciclo": self.v_ciclo.get(),
                "etapa": self.v_etapa.get(),
                "canais": [
                    {k: var.get() for k, var in v.items()} for v, _ in self.lin
                ],
            }
            CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _iniciar(self, modo):
        if self.thread and self.thread.is_alive():
            return
        try:
            p = self._parametros()
        except ValueError as e:
            messagebox.showerror("Parâmetros inválidos", str(e))
            return
        if modo == "coleta" and p["rpm"] is None and not self.auto:
            if not messagebox.askyesno("rpm", "O rpm do software está vazio. Ele é necessário para o ML.\nColetar mesmo assim?"):
                return
        if modo == "coleta" and self.v_etapa.get() == "A" and self.v_furos.get().strip() and not self.auto:
            if not messagebox.askyesno("Etapa A", "A etapa A é 'sem massa', mas há parafusos preenchidos.\nColetar mesmo assim?"):
                return
        self._salvar_config()
        self.p, self.modo = p, modo
        self.canais_ativos = p["canais"]
        self.fs_atual = p["fs"]
        self.buf = np.zeros((len(p["canais"]), 0))
        self.n_total = int(round(p["fs"] * p["duracao"]))
        self.parar.clear()
        gravar = modo == "coleta"
        if self.simulado:
            leitor = LeitorSimulado(p["canais"], p["fs"], p["rpm"] or 1200.0)
        else:
            leitor = LeitorNI(p["modulo"], p["canais"], p["fs"], gravar, self.n_total)
        self.thread = threading.Thread(target=trabalhador, args=(leitor, gravar, self.n_total, self.fila, self.parar), daemon=True)
        self.thread.start()
        self._set_executando(True)
        self._montar_eixos_vivo()
        self._log(("Coletando" if gravar else "Monitorando") + f": {len(p['canais'])} canais, {fmt_hz(p['fs'])} Hz" + (f", {p['duracao']:g} s" if gravar else ""))

    def _set_executando(self, rodando):
        s = "disabled" if rodando else "normal"
        self.b_coletar.config(state=s)
        self.b_monitor.config(state=s)
        self.b_parar.config(state="normal" if rodando else "disabled")

    def _ciclo(self):
        try:
            while True:
                msg = self.fila.get_nowait()
                tipo = msg[0]
                if tipo == "inicio":
                    self.fs_atual = msg[1]
                    if abs(msg[1] - self.p["fs"]) > 1e-6:
                        self._log(f"Taxa real do hardware: {msg[1]:.3f} Hz")
                elif tipo == "bloco":
                    L = int(self.fs_atual * JANELA_BUF_S)
                    self.buf = np.concatenate([self.buf, msg[1]], axis=1)[:, -L:]
                    self.sujo = True
                    if self.modo == "coleta":
                        self.prog["value"] = 100 * msg[2] / self.n_total
                elif tipo == "fim":
                    self._finalizar(msg[1], msg[2])
                elif tipo == "parado":
                    self._set_executando(False)
                    self._log("Parado (nada foi salvo).")
                    self.prog["value"] = 0
                elif tipo == "erro":
                    self._set_executando(False)
                    self.prog["value"] = 0
                    self._log("ERRO: " + msg[1])
                    messagebox.showerror("Erro na aquisição", msg[1])
        except queue.Empty:
            pass
        if self.sujo and self.buf is not None and self.thread and self.thread.is_alive() and time.time() - self.ultimo_desenho > 0.3:
            self._desenhar_vivo()
        self.after(80, self._ciclo)

    def _grafico_vazio(self, texto):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=12, color="#555")
        self.canvas.draw_idle()

    def _montar_eixos_vivo(self):
        self.fig.clear()
        self.eixos = [self.fig.add_subplot(len(self.canais_ativos), 1, k + 1) for k in range(len(self.canais_ativos))]

    def _desenhar_vivo(self):
        self.sujo, self.ultimo_desenho = False, time.time()
        fs = self.fs_atual
        espectro = self.v_vista.get() == "Espectro"
        rpm_ref = self.p["rpm"]
        for (idx, c), linha in zip(self.canais_ativos, self.buf):
            if c["tipo"] == TACO and linha.size > fs:
                r = rpm_do_tacometro(linha, fs, c["ppr_v"])
                if r["rpm"]:
                    rpm_ref = r["rpm"]
        for ax, (idx, c), linha in zip(self.eixos, self.canais_ativos, self.buf):
            ax.clear()
            x = linha[-int(fs * JANELA_PLOT_S):]
            if x.size < 10:
                continue
            if c["tipo"] == ACEL:
                x = x - x.mean()
                if espectro and x.size > 64:
                    esp = 2 * np.abs(np.fft.rfft(x * np.hanning(x.size))) / (x.size * 0.5)
                    f = np.fft.rfftfreq(x.size, 1 / fs)
                    ax.semilogy(f, esp, lw=0.6)
                    ax.set_xlim(0, min(fs / 2, 1000))
                    ax.set_xlabel("frequência (Hz)")
                    if rpm_ref:
                        ax.axvline(rpm_ref / 60, color="r", ls="--", lw=0.8, label=f"rpm/60 = {rpm_ref/60:.1f} Hz")
                        ax.legend(fontsize=8, loc="upper right")
                else:
                    ax.plot(np.arange(x.size) / fs - x.size / fs, x, lw=0.5)
                    ax.set_xlabel("tempo (s)")
                ax.set_title(f"AI{idx} · {curto(c['descricao'])}\nRMS {np.sqrt(np.mean(x**2)):.4f} g | pico {np.abs(x).max():.3f} g", fontsize=8)
                ax.set_ylabel("g" if not espectro else "amplitude (g)")
            else:
                r = rpm_do_tacometro(linha, fs, c["ppr_v"])
                ax.plot(np.arange(x.size) / fs - x.size / fs, x, lw=0.6, color="tab:green")
                ax.set_xlabel("tempo (s)")
                ax.set_ylabel("V")
                ax.set_title(
                    f"AI{idx} · tacômetro\n" + (f"{r['rpm']:.0f} rpm | {r['pulsos']} pulsos | " if r["rpm"] else "SEM PULSOS | ") + f"{r['amplitude_V']:.2f} V pico a pico",
                    fontsize=8,
                    color="black" if r["rpm"] else "red",
                )
        self.canvas.draw_idle()

    def _desenhar_resultado(self, dados, fs, tacho_res):
        self.fig.clear()
        n = len(self.canais_ativos)
        t = np.arange(dados.shape[1]) / fs
        rpm_ref = self.p["rpm"] or (tacho_res["rpm"] if tacho_res else None)
        for k, ((idx, c), linha) in enumerate(zip(self.canais_ativos, dados)):
            a1 = self.fig.add_subplot(n, 2, 2 * k + 1)
            a2 = self.fig.add_subplot(n, 2, 2 * k + 2)
            if c["tipo"] == ACEL:
                x = linha - linha.mean()
                a1.plot(t, x, lw=0.4)
                a1.set_title(f"AI{idx} · {curto(c['descricao'])}\nRMS {np.sqrt(np.mean(x**2)):.4f} g | pico {np.abs(x).max():.3f} g", fontsize=8)
                a1.set_ylabel("g")
                esp = 2 * np.abs(np.fft.rfft(x * np.hanning(x.size))) / (x.size * 0.5)
                f = np.fft.rfftfreq(x.size, 1 / fs)
                a2.semilogy(f, esp, lw=0.5)
                a2.set_xlim(0, min(fs / 2, 1000))
                a2.set_title("espectro (FFT, Hann), 0 a 1000 Hz", fontsize=8)
                a2.set_ylabel("amplitude (g)")
                if rpm_ref:
                    a2.axvline(rpm_ref / 60, color="r", ls="--", lw=0.8, label=f"rpm/60 = {rpm_ref/60:.1f} Hz")
                    a2.legend(fontsize=8, loc="upper right")
                a2.set_xlabel("frequência (Hz)") if k == n - 1 else None
            else:
                bordas, amp = bordas_tacometro(linha, fs)
                a1.plot(t, linha, lw=0.4, color="tab:green")
                a1.set_title(f"AI{idx} · tacômetro\n{amp:.2f} V pico a pico, {len(bordas)} pulsos", fontsize=8)
                a1.set_ylabel("V")
                if len(bordas) >= 3:
                    rpm_i = 60.0 / (np.diff(bordas) / fs * c["ppr_v"])
                    a2.plot(t[bordas[1:]], rpm_i, ".-", ms=2, lw=0.5, color="tab:green")
                    a2.set_title(f"rpm instantâneo\nmédia {rpm_i.mean():.1f}, desvio {rpm_i.std():.1f}", fontsize=8)
                    a2.set_ylabel("rpm")
                    a2.set_xlabel("tempo (s)")
                else:
                    a2.text(0.5, 0.5, "sem pulsos válidos", ha="center", va="center", color="red", transform=a2.transAxes)
                    a2.axis("off")
            if k == n - 1:
                a1.set_xlabel("tempo (s)")
        self.fig.suptitle(f"{self.p['rotulo']} | {fmt_hz(fs)} Hz | {self.p['duracao']:g} s", fontsize=10)
        self.canvas.draw_idle()

    def _finalizar(self, dados, fs):
        p = self.p
        agora = datetime.now()
        pasta = PASTA_DADOS / (f"coleta_simulada_{agora:%Y%m%d}" if self.simulado else f"coleta_{agora:%Y%m%d}")
        pasta.mkdir(parents=True, exist_ok=True)
        colunas, tacho_res, tacho_cfg = [], None, None
        for (idx, c), linha in zip(self.canais_ativos, dados):
            colunas.append(f"ai{idx}_g" if c["tipo"] == ACEL else f"ai{idx}_V")
            if c["tipo"] == TACO and tacho_res is None:
                tacho_res = rpm_do_tacometro(linha, fs, c["ppr_v"])
                tacho_cfg = {"canal": f"ai{idx}", "ppr": c["ppr_v"], "acoplamento": "AC", "iepe": False, **tacho_res}
        rpm_tac = tacho_res["rpm"] if tacho_res else None
        rpm_ref = p["rpm"] or rpm_tac
        base = f"{p['rotulo']}_{int(round(rpm_ref)) if rpm_ref else 'rpmND'}rpm_rep{p['rep']:02d}_{agora:%H%M%S}"
        t = np.arange(dados.shape[1]) / fs
        csv = pasta / f"{base}.csv"
        np.savetxt(csv, np.column_stack([t, dados.T]), delimiter=",", header="tempo_s," + ",".join(colunas), comments="", fmt="%.6g")
        meta = {
            "data_hora": agora.isoformat(timespec="seconds"),
            "software": "coleta_gui.py",
            "rotulo": p["rotulo"],
            "repeticao": p["rep"],
            "rpm_anotada": p["rpm"],
            "rpm_tacometro": rpm_tac,
            "obs": p["obs"],
            "modulo": p.get("modulo", "simulado"),
            "fs_pedida_hz": p["fs"],
            "fs_real_hz": fs,
            "duracao_s": p["duracao"],
            "tacometro": tacho_cfg,
            "posicoes": {f"ai{i}": c["descricao"] for i, c in self.canais_ativos if c["tipo"] == ACEL},
            "canais": {
                f"ai{i}": (
                    {"tipo": "acelerometro", "descricao": c["descricao"], "serie": c["serie"], "sensibilidade_mV_g": c["sens_v"],
                     "iepe": c["iepe"], "excitacao_IEPE_A": EXCITACAO_IEPE_A if c["iepe"] else 0, "acoplamento": c["acopl"]}
                    if c["tipo"] == ACEL
                    else {"tipo": "tacometro", "descricao": c["descricao"], "pulsos_por_volta": c["ppr_v"], "acoplamento": "AC", "iepe": False}
                )
                for i, c in self.canais_ativos
            },
        }
        import hashlib

        qc, rpm_est = controle_qualidade(self.canais_ativos, dados, fs, tacho_res, rpm_ref)
        cond, ref = calcular_condicao(p, self.perfil, rpm_ref)
        modulo = p.get("modulo")
        if modulo not in self.ambiente:
            self.ambiente = {modulo: info_ambiente(None if self.simulado else modulo)}
        meta.update({
            "status": "valida",
            "motivo_descarte": "",
            "rpm_estimado_espectro": rpm_est,
            "condicao": cond,
            "referencias": ref,
            "perfil": dict(self.perfil),
            "ambiente": self.ambiente[modulo],
            "qc": qc,
            "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        })
        csv.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            gravar_indice(self._caminho_indice(), linha_indice(meta, csv.name))
        except OSError as e:
            self._log(f"Aviso: índice não atualizado ({e}). Feche o indice_coletas.csv se estiver aberto no Excel.")
        self.b_descartar.config(state="normal")
        self._atualizar_alvos()
        self.v_remont.set(False)
        self.ultimo_csv = csv
        self.prog["value"] = 100
        self._set_executando(False)
        self._log(f"Salvo: {csv.name}")
        for (idx, c), linha in zip(self.canais_ativos, dados):
            if c["tipo"] == ACEL:
                x = linha - linha.mean()
                self._log(f"  AI{idx}: RMS {np.sqrt(np.mean(x**2)):.4f} g | pico {np.abs(x).max():.3f} g")
        if tacho_res:
            self._log("  Tacômetro: " + (f"{rpm_tac:.1f} rpm ({tacho_res['pulsos']} pulsos)" if rpm_tac else f"SEM PULSOS (amplitude {tacho_res['amplitude_V']:.2f} V)"))
        if rpm_est:
            self._log(f"  rpm estimado pelo espectro: {rpm_est:.0f} (conferência; não substitui a medição)")
        if cond["U_gmm"] is not None:
            self._log(f"  U = {cond['U_gmm']:.1f} g·mm" + (f" | e = {cond['e_gmm_kg']:.2f} g·mm/kg" if cond["e_gmm_kg"] else ""))
        for a in qc["alertas"]:
            self._log("  ALERTA: " + a)
        try:
            self.v_rep.set(str(p["rep"] + 1))
        except tk.TclError:
            pass
        rms = [float(np.sqrt(np.mean((l - l.mean()) ** 2))) for (idx, c), l in zip(self.canais_ativos, dados) if c["tipo"] == ACEL]
        aviso = [a for a in qc["alertas"] if "tacometro" not in a]
        self.l_status.config(
            text=f"✔ Salva: {p['rotulo']} · coleta nº {p['rep']} · RMS " + " / ".join(f"{r:.3f} g" for r in rms)
            + ("   ⚠ " + "; ".join(aviso) if aviso else "")
            + "\nPróximo: mude a etapa/massa se for o caso e colete de novo; ao fim do ciclo, registre a leitura do NK700.",
            foreground="#b35900" if aviso else "#1a7f37",
        )
        self._desenhar_resultado(dados, fs, tacho_res)
        if self.auto:
            self.after(600, self._autoteste_fim)

    def _abrir_pasta(self):
        pasta = self.ultimo_csv.parent if self.ultimo_csv else PASTA_DADOS
        pasta.mkdir(parents=True, exist_ok=True)
        os.startfile(pasta) if sys.platform == "win32" else subprocess.Popen(["xdg-open", str(pasta)])

    def _autoteste_fim(self):
        from PIL import ImageGrab

        self.update()
        x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        saida = Path(os.environ.get("GUI_SHOT", "gui_shot.png"))
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(saida)
        print("screenshot:", saida, "| csv:", self.ultimo_csv)
        self._fechar()

    def _fechar(self):
        self.parar.set()
        self._salvar_config()
        self.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--simulado", action="store_true", help="usa dados sinteticos (sem hardware)")
    ap.add_argument("--auto", action="store_true", help="(teste) coleta 4 s simulados, tira print e fecha")
    a = ap.parse_args()
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    if a.auto:
        global PASTA_DADOS
        PASTA_DADOS = Path(os.environ.get("GUI_DADOS", "gui_dados_teste"))
    app = App(simulado=a.simulado or a.auto, auto=a.auto)
    if a.auto:
        app.v_dur.set("4")
        app.v_rot.set("auto")
        app.v_rpm.set("1200")
        app.after(500, lambda: app._iniciar("coleta"))
    app.mainloop()


if __name__ == "__main__":
    main()
