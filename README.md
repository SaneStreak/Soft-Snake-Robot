# Hierarchical CPG-Regulated RL for a Continuum Soft Snake Robot

A bio-inspired, multi-rate hierarchical control framework combining **Proximal Policy Option-Critic (PPOC)** reinforcement learning, **coupled Matsuoka Central Pattern Generator (CPG)** neural ODEs, a **200 Hz capacitive self-sensing PI inner loop**, and a **12-module bilateral Dielectric Elastomer Actuator (DEA)** continuum plant operating under **Hu et al. (2009) anisotropic Coulomb friction** — paired with a **SOFA v26.06.00 3D hyperelastic FEM simulation** and an **interactive multi-region waypoint arena**.

---

## What Has Been Implemented

1. **High-Level Learning Brain (`src/rl/`)**:
   - Multi-rate **PPOC** agent (`PPOCAgent`, `OptionCriticNetwork`) choosing discrete macro-options $K_f \in \{0.6, 0.8, 1.0, 1.3, 1.6\}$ via bidirectional option termination $\beta_\omega(s)$ and continuous intra-option steering biases $\vec{\alpha} \in [-2.0, 2.0]^4$ (smoothly mapped across 4 spatial body zones to 12 CPG modules via `map_action_to_cpg_alpha`).
   - **Behavior-Cloning Warm-Start + Glass Stall Recovery (`scripts/train_ppoc_agent.py`)**: Markovian lead-compensated teacher (`_markov_cpg_teacher`) with `--warmstart-bc` distillation across multi-angle targets, $L/2$ slick tile crossings, and isotropic glass pads ($\eta = 1.0$), using zero-mean harmonic augmentation, $K_f = 0.80$ low-speed recovery (`v_fwd < 0.12 m/s` or large turn), and low-speed C-arc symmetry breaking.
2. **Low-Level Spinal Layer (`src/spine/`)**:
   - 12-module coupled Matsuoka CPG (`MatsuokaCPG`) integrated with vectorized RK4 at 200 Hz ($\Delta t = 0.005\text{ s}$).
   - Free-Response Oscillation Constraint ($c = 0.75$) preventing amplitude decay and sim-to-real locomotion collapse.
   - Describing-function verification (`src/spine/describing_function.py`) proving linear steering offset vs. complementary tonic bias ($R^2 \ge 0.99$).
3. **200 Hz Inner Loop, Baseline & Physical Plant (`src/controllers/`, `src/plant/`)**:
   - Capacitive self-sensing PI curvature tracker (`CapacitiveInnerLoopPI`) driving 24 bilateral DEA strips (12 left/right pairs).
   - Pull-in safety governor (`DEAActuatorPlant`) clamping voltages to $\le 450\text{ V}$ ($45\text{ V}/\mu\text{m}$ on $10\ \mu\text{m}$ elastomer film, $\ge 20\%$ below $58\text{ V}/\mu\text{m}$ breakdown) and $|dV/dt| \le 20\text{ V/ms}$.
   - Planar continuum kinematics (`PlanarSnakePlant`, $L = 600\text{ mm}, M = 35.6\text{ g}$) under orthogonal anisotropic Coulomb friction ($\eta = \mu_n / \mu_t$).
   - Classical PI+ILOS navigation baseline (`ClassicalPIILOSController`) used for comparative benchmarking and live UI toggling.
4. **Gymnasium Env & 3D SOFA FEM IPC (`src/envs/`, `scripts/sofa_scene_server.py`)**:
   - `SnakeGymEnv` with Stage A / B / C terrain curriculum and domain randomization (`configs/ppoc_training.yaml`).
   - TCP IPC (`SofaConnector`, `SofaLocalFEMBridge`) connecting the 20 Hz / 200 Hz Python controller to a live **SOFA v26.06.00 (`SofaPython3`)** 3D hyperelastic FEM scene (`HexahedronFEMForceField`, 150-node planar floor constraint in `PlanarBellySkate`, 144 bilateral muscle springs, dynamic multi-region floor patches, and synced 3D target beacons).
5. **Interactive Multi-Region 3D Arena (`scripts/interactive_sofa_arena.py`)**:
   - Live companion overhead radar + control panel synced over TCP IPC to SOFA 3D (or standalone `--local` embedded FEM bridge):
     - **Left-Click** anywhere on the arena floor to queue amber target waypoints; **Right-Click** to drop/override an instant green beacon target.
     - **Arena Presets**: `Multi-Region Arena` (Sand $\eta=3.0$, `Tile Trap (L/2, eta=1.0)`, `Upper Tile Pad (eta=1.0)`, `Rough Gravel (eta=4.0)`), `Tile Corridor (L/2)`, and `Open Sand`.
     - **Live Controls**: Switch live between `PPOC-CPG` and `Classical PI+ILOS`, launch a 5-point `Zig-Zag Tour`, `Reset Snake`, `Clear Queue`, `Pause / Run`, and monitor zone, forward speed, frequency $K_f$, peak voltage $V_{\max}$, and goal count.

---

## Environments Required (Teammate Setup)

The project uses **one main Python environment (`soft_snake`)** for all training, testing, plots, and the Arena UI, plus an **optional second environment (`sofa_py312`)** only if you want to run the full 3D SOFA v26.06.00 FEM simulator/GUI.

### 1. Main Environment — `soft_snake` (Required)

- **Python**: `>= 3.10` (`Python 3.11` recommended/tested; must include standard-library `tkinter` for `interactive_sofa_arena.py`)
- **Core Packages** (see `requirements.txt` and `setup.py`):
  - `numpy >= 1.24.0`
  - `scipy >= 1.10.0`
  - `pyyaml >= 6.0`
  - `gymnasium >= 0.29.0`
  - `torch >= 2.1.0`
  - `matplotlib >= 3.7.0`
  - `tensorboard >= 2.14.0`
  - `pytest >= 7.4.0`

```bash
conda create -n soft_snake python=3.11 tk -y
conda activate soft_snake
pip install -r requirements.txt
pip install -e . --no-deps
```

> Everything offline — including `pytest -v`, training, comparative plots, baseline benchmarks, `run_sofa_eval.py` (embedded bridge), and `interactive_sofa_arena.py --local` — runs completely inside `soft_snake` without needing SOFA installed.

### 2. Optional 3D SOFA FEM Environment — `sofa_py312` + SOFA `v26.06.00` (For Full 3D Simulation / GUI)

SOFA `v26.06.00` (`SofaPython3`) is compiled against **Python 3.12**, so SOFA runs in its own lightweight **Python 3.12** environment (`sofa_py312`) and communicates with `soft_snake` over TCP IPC (`127.0.0.1:5555`):

- **External Software**: [SOFA v26.06.00](https://www.sofa-framework.org/) with `SofaPython3` plugin
- **Python Environment (`sofa_py312`)**:
  ```bash
  conda create -n sofa_py312 python=3.12 numpy pyyaml -y
  ```
- **Path Configuration**: If launching via `scripts/launch_arena.ps1` or `scripts/launch_sofa_gui.ps1` on Windows, update `$env:SOFA_ROOT` (currently `C:\Users\sanch\SOFA\v26.06.00`) and the `conda` environment paths to match your local installation.

---

## Repository Structure

```text
├── configs/
│   ├── cpg_matsuoka.yaml            # Matsuoka CPG parameters & FOC settings
│   ├── plant_rev_c.yaml             # DEA actuator, elastomer film & friction specs
│   └── ppoc_training.yaml           # Gymnasium curriculum & PPOC hyperparameters
├── checkpoints/
│   └── ppoc_best.pt                 # Pretrained PPOC-CPG checkpoint tracked for evaluation
├── papers/                          # Reference PDFs (Hu et al. 2009, CPG/RL literature)
├── results/                         # Generated comparative study plots & 3D eval visuals
├── scripts/
│   ├── interactive_sofa_arena.py    # Companion Click-to-Target Arena UI (SOFA IPC or --local)
│   ├── launch_arena.ps1             # Launches SOFA 3D GUI + Companion Arena UI together
│   ├── launch_sofa_gui.ps1          # Launches SOFA v26.06.00 GUI server standalone
│   ├── plot_comparative_study.py    # Classical PI+ILOS vs. PPOC-CPG plots
│   ├── run_baseline_benchmark.py    # Sand vs. L/4, L/2, L tile band classical benchmark
│   ├── run_sofa_eval.py             # Zero-shot 1D -> 3D SOFA FEM transfer & GIF script
│   ├── sofa_scene_server.py         # SofaPython3 3D FEM scene + TCP IPC server
│   └── train_ppoc_agent.py          # Parallel PPOC + BC warm-start trainer
├── src/
│   ├── controllers/                 # Capacitive PI inner loop & Classical PI+ILOS baseline
│   ├── envs/                        # SnakeGymEnv & SofaConnector / SofaLocalFEMBridge
│   ├── plant/                       # Bilateral DEA actuator, Coulomb friction, 1D kinematics
│   ├── rl/                          # OptionCriticNetwork, PPOCAgent, RewardShaper
│   └── spine/                       # Coupled Matsuoka CPG ODE & describing function
├── tests/                           # 20 unit & physics verification tests
├── requirements.txt
└── setup.py
```

---

## Quickstart & Commands

```bash
conda activate soft_snake

# 1. Run unit & physics verification tests (20 tests)
pytest -v

# 2. Run classical PI+ILOS baseline benchmark (sand vs. L/4, L/2, L tile bands)
python scripts/run_baseline_benchmark.py

# 3. Train PPOC-CPG with Behavior-Cloning warm-start & parallel environments
python scripts/train_ppoc_agent.py --warmstart-bc --num-envs 8 --total-timesteps 16384

# 4. Generate comparative trajectory, velocity, and DEA safety plots
python scripts/plot_comparative_study.py

# 5. Run zero-shot 1D -> 3D transfer evaluation (local bridge by default)
python scripts/run_sofa_eval.py

# 6. Run Interactive Arena WITHOUT SOFA installed (embedded 3D physics bridge)
python scripts/interactive_sofa_arena.py --local
# Headless Arena smoke test:
python scripts/interactive_sofa_arena.py --smoke-test
```

### Running With Full SOFA v26.06.00 3D GUI (Windows)

```powershell
# Launch both SOFA 3D GUI and Companion Click-to-Target Arena UI:
.\scripts\launch_arena.ps1

# Or run SOFA GUI & headless evaluation separately:
# Terminal 1:
.\scripts\launch_sofa_gui.ps1
# Terminal 2:
conda activate soft_snake
python scripts/run_sofa_eval.py --external --host 127.0.0.1 --port 5555 --no-animate
```
