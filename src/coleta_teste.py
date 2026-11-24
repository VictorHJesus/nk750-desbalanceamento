import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

POSICOES = {
    "ai0": "mancal 1 (lado do motor), radial horizontal",
    "ai1": "mancal 1 (lado do motor), radial vertical",
    "ai2": "mancal 2, radial horizontal",
}
TACOMETRO = "ai3"
SENSIBILIDADE_MV_G = 10.0
EXCITACAO_IEPE_A = 0.002


def achar_modulo(nome=None):
    from nidaqmx.system import System

    dispositivos = list(System.local().devices)
    if nome:
        return nome
    for d in dispositivos:
        if "9234" in d.product_type:
            return d.name
    nomes = [f"{d.name} ({d.product_type})" for d in dispositivos] or ["nenhum dispositivo"]
    raise SystemExit("NI 9234 nao encontrado. Dispositivos vistos: " + "; ".join(nomes))


def listar():
    from nidaqmx.system import System

    sistema = System.local()
    print("Driver NI-DAQmx:", sistema.driver_version)
    for d in sistema.devices:
        print(f"- {d.name}: {d.product_type}")
    if not list(sistema.devices):
        print("Nenhum dispositivo. Verifique o NI MAX.")


def bordas_tacometro(v, fs):
    p5, p95 = np.percentile(v, [0.1, 99.9])
    amplitude = float(p95 - p5)
    if amplitude < 0.2:
        return np.array([], dtype=int), amplitude
    baixo, alto = p5 + 0.3 * amplitude, p5 + 0.7 * amplitude
    estado = np.full(len(v), -1)
    estado[v > alto] = 1
    estado[v < baixo] = 0
    valido = np.where(estado >= 0, np.arange(len(v)), 0)
    estado = estado[np.maximum.accumulate(valido)]
    estado[estado < 0] = 0
    subida = np.where((estado[1:] == 1) & (estado[:-1] == 0))[0] + 1
    return subida, amplitude


def rpm_do_tacometro(v, fs, ppr=1):
    bordas, amplitude = bordas_tacometro(v, fs)
    vazio = {"rpm": None, "rpm_desvio": None, "pulsos": int(len(bordas)), "amplitude_V": amplitude}
    if len(bordas) < 3:
        return vazio
    periodos = np.diff(bordas) / fs
    rpm_inst = 60.0 / (periodos * ppr)
    rpm_medio = 60.0 * (len(bordas) - 1) / ((bordas[-1] - bordas[0]) / fs) / ppr
    return {
        "rpm": float(rpm_medio),
        "rpm_desvio": float(np.std(rpm_inst)),
        "pulsos": int(len(bordas)),
        "amplitude_V": amplitude,
    }


def coletar(modulo, fs, duracao, tacometro=True):
    import nidaqmx
    from nidaqmx.constants import (
        AccelSensitivityUnits,
        AccelUnits,
        AcquisitionType,
        Coupling,
        ExcitationSource,
        VoltageUnits,
    )

    n = int(round(fs * duracao))
    with nidaqmx.Task() as task:
        for ch in POSICOES:
            canal = task.ai_channels.add_ai_accel_chan(
                f"{modulo}/{ch}",
                min_val=-50.0,
                max_val=50.0,
                units=AccelUnits.G,
                sensitivity=SENSIBILIDADE_MV_G,
                sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                current_excit_source=ExcitationSource.INTERNAL,
                current_excit_val=EXCITACAO_IEPE_A,
            )
            canal.ai_coupling = Coupling.AC
        if tacometro:
            tacho = task.ai_channels.add_ai_voltage_chan(
                f"{modulo}/{TACOMETRO}", min_val=-5.0, max_val=5.0, units=VoltageUnits.VOLTS
            )
            tacho.ai_coupling = Coupling.DC
        task.timing.cfg_samp_clk_timing(
            rate=fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=n
        )
        fs_real = task.timing.samp_clk_rate
        print(f"Taxa pedida: {fs} Hz | taxa real do hardware: {fs_real} Hz")
        print(f"Coletando {duracao} s ({n} amostras por canal)...")
        dados = np.array(task.read(number_of_samples_per_channel=n, timeout=duracao + 15))
    return dados, fs_real


def resumo(dados, fs, rpm):
    print("\nResumo rapido (confira se faz sentido):")
    for i, ch in enumerate(POSICOES):
        x = dados[i] - dados[i].mean()
        rms = float(np.sqrt(np.mean(x**2)))
        espectro = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        banda = freqs > 2.0
        pico = float(freqs[banda][np.argmax(espectro[banda])])
        msg = f"  {ch}: RMS = {rms:.4f} g | pico espectral em {pico:.2f} Hz"
        if rpm:
            msg += f" (rotacao de referencia = {rpm / 60:.2f} Hz)"
        print(msg)
    print("Dica: com rotor girando, o pico deve cair perto da frequencia de rotacao.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--listar", action="store_true", help="lista dispositivos NI e sai")
    p.add_argument("--modulo", help="nome do modulo no NI MAX (padrao: acha o NI 9234)")
    p.add_argument("--fs", type=float, default=10240.0, help="taxa de amostragem em Hz (padrao 10240)")
    p.add_argument("--duracao", type=float, default=30.0, help="segundos (padrao 30)")
    p.add_argument("--rpm", type=float, help="rotacao anotada (inversor ou tacometro de mao); opcional se houver tacometro no ai3")
    p.add_argument("--sem-tacometro", action="store_true", help="nao le o ai3 (so os 3 acelerometros)")
    p.add_argument("--ppr", type=int, default=1, help="pulsos do tacometro por volta (padrao 1)")
    p.add_argument("--rotulo", default="teste", help="nome da condicao, ex.: sem_massa")
    p.add_argument("--obs", default="", help="observacoes livres")
    a = p.parse_args()

    if a.listar:
        listar()
        return

    modulo = achar_modulo(a.modulo)
    usa_tacho = not a.sem_tacometro
    dados, fs_real = coletar(modulo, a.fs, a.duracao, usa_tacho)
    n_acel = len(POSICOES)
    tacho = rpm_do_tacometro(dados[n_acel], fs_real, a.ppr) if usa_tacho else None
    if tacho:
        if tacho["rpm"] is None:
            print(f"Tacometro: sem pulsos validos (amplitude pico a pico = {tacho['amplitude_V']:.2f} V, pulsos = {tacho['pulsos']}).")
        else:
            print(
                f"Tacometro: {tacho['rpm']:.1f} rpm (desvio {tacho['rpm_desvio']:.1f}) "
                f"| {tacho['pulsos']} pulsos | amplitude {tacho['amplitude_V']:.2f} V pico a pico"
            )
    rpm_ref = a.rpm or (tacho["rpm"] if tacho else None)

    agora = datetime.now()
    pasta = Path(__file__).resolve().parents[1] / "testes" / f"teste_{agora:%Y%m%d}"
    pasta.mkdir(parents=True, exist_ok=True)
    base = f"{a.rotulo}_{int(round(rpm_ref)) if rpm_ref else 'rpmND'}rpm_{agora:%H%M%S}"

    t = np.arange(dados.shape[1]) / fs_real
    np.savetxt(
        pasta / f"{base}.csv",
        np.column_stack([t, dados.T]),
        delimiter=",",
        header="tempo_s," + ",".join(f"{c}_g" for c in POSICOES) + ("," + TACOMETRO + "_V" if usa_tacho else ""),
        comments="",
        fmt="%.6g",
    )
    meta = {
        "data_hora": agora.isoformat(timespec="seconds"),
        "rotulo": a.rotulo,
        "rpm_anotada": a.rpm,
        "rpm_tacometro": tacho["rpm"] if tacho else None,
        "tacometro": (
            {"canal": TACOMETRO, "ppr": a.ppr, "acoplamento": "DC", "iepe": False, **tacho} if tacho else None
        ),
        "obs": a.obs,
        "modulo": modulo,
        "fs_pedida_hz": a.fs,
        "fs_real_hz": fs_real,
        "duracao_s": a.duracao,
        "sensibilidade_mV_g_catalogo": SENSIBILIDADE_MV_G,
        "excitacao_IEPE_A": EXCITACAO_IEPE_A,
        "acoplamento": "AC",
        "posicoes": POSICOES,
    }
    (pasta / f"{base}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo: {pasta / (base + '.csv')}")
    resumo(dados[:n_acel], fs_real, rpm_ref)


if __name__ == "__main__":
    main()
