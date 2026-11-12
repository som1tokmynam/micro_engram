"""Config keys of the micro_engram Axolotl port plugin (merged into Axolotl's config schema)."""
from typing import Optional

from pydantic import BaseModel


class QLoRAPortArgs(BaseModel):
    # item 5: profile this optimizer step (1-based; 3 = after the optimizer state exists).
    qlora_port_profile_step: Optional[int] = None
    # item 1: global layers probe each call for packed rows (not Axolotl's static sample_packing).
    qlora_port_dynamic_packing: Optional[bool] = True
    # item 1: right padding continues the last document's position_ids instead of restarting at 0.
    qlora_port_continue_pad_positions: Optional[bool] = True
    # item 3 (minor): a packed document's first token is not the previous document's target.
    qlora_port_mask_cross_doc_targets: Optional[bool] = True
    # item 2: no model-built mask for the global layers, so packed rows reach the Triton kernel.
    qlora_port_unmask_global_layers: Optional[bool] = True
    # item 3: per-token loss over the whole step (Trainer gets num_items_in_batch).
    qlora_port_token_loss: Optional[bool] = True
    # item 6: a failing rank prints its traceback and exits instead of hanging in NCCL.
    qlora_port_fail_fast: Optional[bool] = True
