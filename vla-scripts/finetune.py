"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import gc
import os
from collections import deque
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor

try:
    import wandb
except ImportError:
    wandb = None

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# cuDNN benchmark - picks optimal algorithm once and caches it (safe with fixed batch sizes)
torch.backends.cudnn.benchmark = True


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    use_gradient_checkpointing: bool = False                        # Trade VRAM for ~20% slower steps; enable if OOM
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: Optional[str] = None                         # W&B project name (if specified, wandb logging is enabled)
    wandb_entity: Optional[str] = None                          # W&B entity/account name
    run_id_note: Optional[str] = None                           # Extra note for logging, Weights & Biases
    csv_logging: bool = False                                   # Enable logging in csv file
    log_freq: int = 10                                          # How many steps between log entries
    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if cfg.use_gradient_checkpointing:
        base_vla.gradient_checkpointing_enable()

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        base_vla = prepare_model_for_kbit_training(base_vla)
    else:
        base_vla = base_vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(base_vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    if cfg.use_lora:
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True)
    else:
        vla = DDP(base_vla, device_ids=[device_id], find_unused_parameters=True)

    # Enable static graph for compatibility with gradient checkpointing + DDP
    if cfg.use_gradient_checkpointing:
        vla._set_static_graph()

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        pin_memory=True,
    )

    # Initialize Logging =>> W&B and CSV
    _wandb_enabled = distributed_state.is_main_process and cfg.wandb_project is not None and wandb is not None

    if _wandb_enabled:
        run_name = f"ft+{exp_id}"
        experiment = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_name)
        print(f"\n{'='*60}")
        print("W&B logging enabled.")
        print(f"  Project: {cfg.wandb_project}")
        print(f"  Entity:  {cfg.wandb_entity}")
        print(f"  Run ID:  {run_name}")
        print("Metrics below would be written to W&B:")
        print('='*60)

    # Initialize CSV logging
    if cfg.csv_logging and distributed_state.is_main_process:
        with open(run_dir / "log.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["gradient_step", "train_loss", "action_accuracy", "l1_loss"])

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Gradient step counter (increments only when optimizer.step() is called)
    gradient_step_idx = 0

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch["input_ids"].to(device_id, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device_id, non_blocking=True),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id, non_blocking=True),
                    labels=batch["labels"].to(device_id, non_blocking=True),
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Only compute action accuracy, L1 loss, and loss scalar at logging intervals (these are for metrics only)
            if distributed_state.is_main_process and gradient_step_idx % cfg.log_freq == 0:
                recent_losses.append(loss.item())
                # Compute Accuracy and L1 Loss for Logging (detach to break graph refs early)
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1].detach()
                action_preds = action_logits.argmax(dim=2).cpu()
                action_gt_cpu = batch["labels"][:, 1:].cpu()
                mask = action_gt_cpu > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt_cpu) & mask
                mask_sum = mask.sum().float()
                if mask_sum > 0:
                    action_accuracy = correct_preds.sum().float() / mask_sum
                else:
                    action_accuracy = torch.tensor(0.0)

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt_cpu[mask].numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())
                smoothened_loss = sum(recent_losses) / len(recent_losses)

            # Push Metrics to W&B or write CSV log
            if distributed_state.is_main_process and gradient_step_idx % cfg.log_freq == 0:
                metrics_to_log = {"train_loss": smoothened_loss}

                if recent_action_accuracies:
                    smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                    smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
                    metrics_to_log["action_accuracy"] = smoothened_action_accuracy
                    metrics_to_log["l1_loss"] = smoothened_l1_loss

                if _wandb_enabled:
                    wandb.log(metrics_to_log, step=gradient_step_idx)
                if cfg.csv_logging:
                    with open(run_dir / "log.csv", "a") as csvf:
                        writer = csv.writer(csvf)
                        writer.writerow([gradient_step_idx] + [metrics_to_log.get(k, 0.0) for k in ["train_loss", "action_accuracy", "l1_loss"]])

            # Promoter Step (every grad accumulation steps)
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                gradient_step_idx += 1
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, save only adapter weights (merging done post-hoc after training)
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                dist.barrier()
                # Force GC at checkpoint boundaries to prevent system RAM leaks during long runs
                gc.collect()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                # Ensure final checkpoint is saved even if max_steps doesn't align with save_steps
                if gradient_step_idx % cfg.save_steps != 0 and distributed_state.is_main_process:
                    print(f"Saving final Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                dist.barrier()

                break

    # Merge LoRA weights into model backbone post-hoc
    if cfg.use_lora and distributed_state.is_main_process:
        print("Merging LoRA weights into base model...")
        # Free everything from training to avoid OOM — without this, we'd have the full
        # training model + optimizer state + fresh base model + merged model all in memory.
        del vla, optimizer, dataloader, vla_dataset
        torch.cuda.empty_cache()
        gc.collect()

        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if cfg.save_latest_checkpoint_only:
            merged_vla.save_pretrained(run_dir)
            print(f"Saved merged model at: {run_dir}")
        else:
            checkpoint_dir = run_dir / f"{run_dir.name}--{gradient_step_idx}_chkpt"
            os.makedirs(checkpoint_dir, exist_ok=True)

            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            merged_vla.save_pretrained(checkpoint_dir)

            print(f"Saved merged model at: {checkpoint_dir}")


if __name__ == "__main__":
    finetune()
