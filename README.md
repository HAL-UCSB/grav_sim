# GraV Sim

Forward kinematics simulator for grasping hand single-finger motion.

## Setup
from the repository root folder:
```bash
uv sync
uv run
uv run streamlit run .\src\grav_sim\ui\verify_assets.py
```
Follow the instructions on the provided in verify_assets page.

## Running

```bash
uv run grav_sim
```

## Cite

```latex
@inproceedings{aponte2024grav,
  title={Grav: Grasp volume data for the design of one-handed xr interfaces},
  author={Aponte, Alejandro and Caetano, Arthur and Luo, Yunhao and Sra, Misha},
  booktitle={Proceedings of the 2024 ACM designing interactive systems conference},
  pages={151--167},
  year={2024}
}
```