# Detecção de desbalanceamento em eixo rotativo: código e dados

Código e conjunto de dados do TCC **"Detecção de desbalanceamento em eixo rotativo por análise de vibração e aprendizado de máquina"**
(Victor Hugo Alves de Jesus, Engenharia Mecânica, Universidade Federal de Goiás, 2026).

Os dados foram coletados em uma bancada de balanceamento Teknikao NK750, com dois acelerômetros, módulo NI 9234 e tacômetro a laser.
O texto do TCC explica o método e os resultados. Aqui está só o que é preciso para usar os dados e reproduzir a análise.

## Conteúdo

```
dataset/    37 coletas (um .csv de sinais e um .json de metadados cada), índice e configuração dos ciclos
src/
  analise_1x.py    extrai o vetor síncrono 1×, 2× e 3× de cada coleta, usando o pulso do tacômetro como fase
  modelo_tcc.py    modelo físico, validação deixando um ciclo de fora, aprendizado de máquina, tabelas e figuras
  coleta_gui.py    interface de coleta com o NI 9234 (tem modo --simulado, sem hardware)
  coleta_teste.py  funções de apoio à coleta e coleta simples por linha de comando
```

## Instalação

Python 3.13 (testado no Windows 11).

```bash
pip install -r requirements.txt
```

Só para coletar com o hardware, instale também o driver NI-DAQmx da National Instruments e `pip install nidaqmx`.

## Reproduzir a análise

```bash
python src/modelo_tcc.py
```

Primeiro extrai os vetores 1× (leva alguns minutos) e depois ajusta os modelos. Grava em `saida/`:

- `vetores_estados.csv` e `vetores_janelas.csv`: vetores síncronos por coleta e por janela de 1 s;
- `gerado/`: números e tabelas em LaTeX do capítulo de resultados;
- `figuras/`: figuras do capítulo de resultados.

Para reaproveitar os vetores já calculados, defina `PULAR_1X=1`. Para usar outras pastas, defina `TCC_DATASET` (dados) e `TCC_SAIDA` (saída).

## Dados

```
dataset/
  dados/               <rótulo>_<rpm>rpm_rep<n>_<hhmmss>.csv e .json
  indice_coletas.csv   uma linha por coleta
  ciclos_config.csv    parafusos fixos de cada ciclo e ciclos excluídos
  LEIAME.md            descrição completa
```

Cada coleta tem 30 s a 5120 Hz, com as colunas `tempo_s`, `ai0_g` e `ai1_g` (acelerômetros, em g) e `ai3_V` (pulso do tacômetro, em V).
Leia o [`dataset/LEIAME.md`](dataset/LEIAME.md) para os rótulos, os ciclos excluídos e os avisos sobre cada ciclo.
O nome do operador, o nome do computador e os números de série do equipamento foram anonimizados nos metadados.

## Como citar

> JESUS, Victor Hugo Alves de. *Detecção de desbalanceamento em eixo rotativo por análise de vibração e aprendizado de máquina*. Trabalho de Conclusão de Curso (Engenharia Mecânica) — Universidade Federal de Goiás, Goiânia, 2026.

## Licença

Código: MIT ([LICENSE](LICENSE)). Dados em `dataset/`: CC BY 4.0, citando o TCC.
