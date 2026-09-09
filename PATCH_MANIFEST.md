# Prompt Decommitment Patch Manifest

Overlay these paths onto `Yunbo-max/gr-ktc-minecraft`:

- `gr_ktc/prompt_decommitment.py`
- `gr_ktc/local_qwen_vl_provider.py`
- `scripts/run_mineexplorer_prompt_decommitment.py`
- `scripts/analyze_prompt_decommitment.py`
- `scripts/__init__.py` (only if the repository does not already contain it)
- `configs/prompt_decommitment_16gb.yaml`
- `docs/PROMPT_DECOMMITMENT_EXPERIMENT.md`
- `docs/superpowers/specs/2026-09-08-prompt-decommitment-design.md`
- `docs/superpowers/plans/2026-09-08-prompt-decommitment.md`
- `tests/test_prompt_decommitment.py`
- `tests/test_local_qwen_vl_provider.py`
- `tests/test_prompt_decommitment_analysis.py`

Runtime extras:

```bash
pip install -e '.[train,test]' pillow gymnasium requests loguru python-dotenv typer fastapi uvicorn pydantic imageio imageio-ffmpeg
```

Official benchmark checkout:

```bash
git clone https://github.com/meituan-longcat/MineExplorer third_party/MineExplorer
git -C third_party/MineExplorer checkout cca4c5ed4b857ecf554bd51902ca9e01d359f383
```

The patch intentionally does not change existing GR-KTC/KV/MetaPlastic code. It is a new, isolated prompt-only experiment path.
