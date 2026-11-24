# Conjunto de dados de vibração: bancada de balanceamento Teknikao NK750 (UFG)

Dados coletados para o Trabalho de Conclusão de Curso *Detecção de desbalanceamento em eixo rotativo por análise de vibração e aprendizado de máquina*, Engenharia Mecânica, Universidade Federal de Goiás, 2026.
Este conjunto é publicado junto com o código de análise, sob a licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.pt-br).
Ao usar estes dados, cite o TCC.

**Anonimização:** o nome do operador, o nome do computador e os números de série dos sensores, do módulo e do chassi foram substituídos por `anonimizado` nos `.json` e no índice. Os sinais (`.csv`) não foram alterados.

---

## Conteúdo

| Item | O que é |
|---|---|
| `dados/` | 37 coletas válidas, cada uma com um `.csv` (sinais) e um `.json` (metadados) |
| `indice_coletas.csv` | uma linha por coleta: rótulo, rotação, massas adicionadas e leitura do software da bancada |
| `ciclos_config.csv` | parafusos que já faziam parte do rotor no início de cada ciclo, e ciclos excluídos do ajuste |

**Sessões:** 15/10, 31/10 e 01/11/2026. A correia foi trocada entre a primeira e a segunda.
**Coletas:** 30 s cada, 5120 Hz, rotação medida pelo tacômetro de 366 a 856 rpm.

---

## Sinais (`.csv`)

| Coluna | Unidade | Descrição |
|---|---|---|
| `tempo_s` | s | tempo desde o início da coleta |
| `ai0_g` | g | acelerômetro na viga do pedestal 1 (lado esquerdo), direção horizontal radial |
| `ai1_g` | g | acelerômetro na viga do pedestal 2 (lado direito), direção horizontal radial |
| `ai3_V` | V | pulso do tacômetro, 1 por volta, na referência angular do rotor |

Acelerômetros PCB 352C03 (sensibilidade nominal 10 mV/g), fixados com cera de abelha, e módulo NI 9234 com acoplamento AC. As sensibilidades individuais estão no `.json` de cada coleta.

A amostra 0 do NI 9234 às vezes vem com uma borda falsa no tacômetro. Os scripts do TCC ignoram bordas nos primeiros 10 ms.

---

## Nome dos arquivos e estrutura do ensaio

`CF<ciclo><estado>_<rpm>rpm_rep<n>_<hhmmss>`, por exemplo `CF05B_480rpm_rep27_115702`.

Cada **ciclo** é um procedimento de balanceamento completo, feito em uma única rotação:

| Estado | O que é |
|---|---|
| A | condição inicial do ciclo |
| B | condição inicial + massa de prova em um furo |
| C | correção indicada pelo software da bancada |

Ajustes finos do estado C são coletas separadas com o mesmo rótulo C (CF06: 2 coletas C; CF11: 4 coletas C).

**Histórico dos ciclos:**
- CF04 foi interrompido e tem só os estados A e B.
- CF07 tem só o estado A válido. A leitura do software foi feita na aba errada, e o ciclo foi repetido como CF08.
- CF09 foi refeito do zero como CF10 e não está no conjunto.
- CF05 começa com um parafuso fixo de 5,24 g no furo 2 (ver `ciclos_config.csv`).

---

## Rótulos: como obter o desbalanceamento de cada coleta

O rotor tem **16 furos (0 a 15) a 22,5° um do outro, no raio de 100 mm**. O furo 0 fica na referência angular marcada no rotor.
**A numeração dos furos cresce no sentido oposto ao da rotação.**

O desbalanceamento **adicionado** em uma coleta é conhecido pelos parafusos pesados:

$$\vec{U}_{add} = \sum_k m_k \cdot 100\ \text{mm} \cdot e^{\,j\theta_k}, \qquad \theta_k = -22{,}5° \cdot f_k$$

em que $m_k$ é a massa (g) e $f_k$ o número do furo de cada parafuso. Os parafusos de cada coleta estão na coluna `obs` do índice, no formato `furos(furo:g)=3:4.7;4:1.23`. A eles somam-se os parafusos fixos do ciclo, listados em `ciclos_config.csv`.

O desbalanceamento **total** é $\vec{U}_{liq} = \vec{U}_0 + \vec{U}_{add}$, em que $\vec{U}_0$ é o desbalanceamento próprio do rotor limpo. $\vec{U}_0$ não foi medido diretamente: o TCC o estima a partir dos ensaios (Seção 3.5.2) em cerca de **500 g·mm a 252,9°**, no referencial dos furos. Esse valor é uma estimativa do modelo, não uma medição independente.

---

## Índice (`indice_coletas.csv`): colunas principais

| Coluna | Descrição |
|---|---|
| `rotulo` | ciclo e estado, por exemplo `CF05B` |
| `rpm_anotada` | rotação lida no software da bancada no momento da coleta |
| `rpm_tacometro` | rotação calculada pelos pulsos do tacômetro |
| `massa_g`, `raio_mm`, `angulo_deg` | massa de prova do estado B |
| `obs` | parafusos adicionados (`furo:g`) e observações da coleta |
| `nk700_*` | leitura da tela do software da bancada (ver abaixo) |
| `uso_no_modelo` | `ajuste` (31 coletas) ou `fora do ajuste` (6 coletas, CF12 e CF13) |

**Leitura do software da bancada:** nem toda coleta tem todos os campos. A massa e o ângulo de correção (`nk700_massa_g`, `nk700_angulo_deg`) só existem nas 8 coletas em que o software os indicou. O residual (`nk700_residual_g`) aparece em 15, a amplitude e a fase em 22, e a rotação em 35 das 37.

---

## Avisos

- **CF12 e CF13:** uma arruela de cerca de 3 g, sem furo, foi colada ao rotor com fita crepe, em posição e ângulo desconhecidos. Por isso o desbalanceamento inicial é diferente dos outros ciclos, e o rótulo desses ciclos não pode ser calculado com o $\vec{U}_0$ do rotor limpo. Eles estão no conjunto, mas ficam fora das análises do TCC.
- **CF13:** a indicação do software (15,57 g) não é coerente com os vetores medidos; os furos podem ter sido anotados errados.
- **CF03 (367 rpm):** o sinal em 1× é fraco (menos de 1 mg), e é a pior relação sinal-ruído do conjunto.
- **Não incluídos:** 6 coletas descartadas durante os ensaios e os testes anteriores a 15/10.

---

## Processamento usado no TCC

Os scripts estão em `src/` deste repositório:
- `analise_1x.py` extrai o vetor síncrono 1×, 2× e 3× de cada coleta e de cada janela de 1 s;
- `modelo_tcc.py` faz o ajuste do modelo físico, a validação deixando um ciclo de fora e os modelos de aprendizado de máquina.
