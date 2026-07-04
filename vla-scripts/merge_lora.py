"""
merge_lora.py

Utility script to merge LoRA adapter weights into a base OpenVLA model from saved checkpoints.
Use this if finetune.py crashes before completing post-hoc merging.

Run with:
    python vla-scripts/merge_lora.py \
        --vla_path openvla/openvla-7b \
        --adapter_dir <PATH/TO/ADAPTER/WEIGHTS> \
        --output_dir <PATH/TO/SAVE/MERGED/MODEL>

Optional:
    --dataset_statistics_path <PATH/TO/dataset_statistics.pkl>  (to copy into output dir)
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def merge_lora(
    vla_path: str,
    adapter_dir: Path,
    output_dir: Path,
    dataset_statistics_path: Path | None = None,
) -> None:
    print(f"Loading base model from `{vla_path}`...")

    # Register OpenVLA to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    base_vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from `{adapter_dir}`...")
    merged_vla = PeftModel.from_pretrained(base_vla, str(adapter_dir))
    merged_vla = merged_vla.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)

    # Derive run_dir from adapter_dir: they share the same exp_id suffix.
    # e.g. adapter-tmp/<exp_id> -> runs/<exp_id>
    adapter_parent = adapter_dir.parent.name  # "adapter-tmp" or custom
    exp_id = adapter_dir.name
    candidate_run_dirs = []

    # Check common parent directory siblings (runs/ next to adapter-tmp/)
    sibling_runs = adapter_dir.parent.parent / "runs" / exp_id
    if sibling_runs.exists():
        candidate_run_dirs.append(sibling_runs)
    # Also check the default run_root_dir pattern
    for runs_parent in [adapter_dir.parent.parent, Path("runs")]:
        candidate = runs_parent / exp_id
        if candidate.exists() and "tokenizer_config.json" in os.listdir(candidate):
            candidate_run_dirs.append(candidate)

    if candidate_run_dirs:
        source_dir = candidate_run_dirs[0]
        print(f"Copying processor from `{source_dir}`...")
        processor = AutoProcessor.from_pretrained(str(source_dir), trust_remote_code=True)
        processor.save_pretrained(str(output_dir))

    print(f"Saving merged model to `{output_dir}`...")
    merged_vla.save_pretrained(str(output_dir))

    if dataset_statistics_path and Path(dataset_statistics_path).exists():
        import shutil
        shutil.copy2(dataset_statistics_path, output_dir / "dataset_statistics.pkl")
        print(f"Copied dataset statistics to `{output_dir}`.")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into base OpenVLA model.")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b", help="Path to base OpenVLA model on HuggingFace Hub.")
    parser.add_argument("--adapter_dir", type=Path, required=True, help="Path to directory containing saved LoRA adapter weights.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to save the merged model.")
    parser.add_argument(
        "--dataset_statistics_path",
        type=Path,
        default=None,
        help="Optional path to dataset_statistics.pkl to copy into output directory.",
    )
    args = parser.parse_args()

    merge_lora(args.vla_path, args.adapter_dir, args.output_dir, args.dataset_statistics_path)
