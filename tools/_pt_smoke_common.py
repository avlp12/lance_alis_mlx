"""STAGE 9+ PT smoke 공용 헬퍼 — 환경 셋업 + Lesson E 재발화 원천 차단.

배경: STAGE 7 §3 Lesson E 가 STAGE 9 §0 PT smoke 에서 재발화. 두 시간 디버깅.
STAGE 7 fix 가 그 harness 한정이었음 — 새 PT smoke 가 *같은 코드* 사용해도
*mask 전달 방식*만 다르게 해서 silent 발화. 미래 STAGE 9+ PT smoke
(video_edit, x2t_video, t2v fine-tune) 작성 시 동일 재발화 위험. → 공용 헬퍼화.

사용 패턴:
    from tools._pt_smoke_common import install_pt_smoke_env, pt_layer_mask
    install_pt_smoke_env()    # env shim — must call BEFORE any PT Lance import
    # ... build PT model ...
    for layer in pt.layers:
        h = layer(..., attention_mask=pt_layer_mask(attn_dense_bool), ...)
"""
from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import types


__all__ = ["install_pt_smoke_env", "pt_layer_mask"]


def install_pt_smoke_env() -> None:
    """모든 STAGE 9+ PT smoke 가 호출하는 단일 환경 셋업.

    포함:
      1. flash_attn stub (single-sequence shim, SDPA fallback)
      2. transformers utils 의 is_flash_attn_*_available → False (transformers ≥ 5.x 의
         flash_attention import 경로 차단)
      3. modeling.lance namespace stub (refs/Lance/modeling/lance 직접 import 허용)
      4. flex_attention SDPA patch (★ Lesson E 처리 위치 — 자세한 설명은
         `pt_layer_mask` docstring 참조)

    호출 시점: 다른 어떤 PT Lance / transformers import 보다 *먼저*.
    """
    import torch
    import torch.nn.functional as F

    # --- 1. flash_attn stub ----------------------------------------------------
    def _flash_shim(q, k, v, cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k, causal=True, **_kw):
        if cu_seqlens_q.numel() != 2 or cu_seqlens_k.numel() != 2:
            raise NotImplementedError("flash_attn shim handles single-sequence only")
        n_heads = q.shape[1]
        n_kv = k.shape[1]
        if n_kv < n_heads:
            rep = n_heads // n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=bool(causal))
        return out.squeeze(0).transpose(0, 1).contiguous()

    fa = types.ModuleType("flash_attn")
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    fa.flash_attn_varlen_func = _flash_shim
    sys.modules["flash_attn"] = fa

    # --- 2. transformers flash-attn availability → False -----------------------
    import transformers.utils.import_utils as _imp_utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_imp_utils, fn, lambda: False)
    import transformers.utils as _utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_utils, fn, lambda: False)

    # --- 3. modeling.lance namespace stub --------------------------------------
    lance_dir = os.path.abspath("refs/Lance/modeling/lance")
    pkg = types.ModuleType("modeling.lance")
    pkg.__path__ = [lance_dir]
    sys.modules["modeling.lance"] = pkg

    # --- 4. flex_attention SDPA patch (★ Lesson E containment) -----------------
    import torch.nn.attention.flex_attention as _fa
    from torch.nn.attention.flex_attention import BlockMask

    def _dense_from_block_mask(bm: "BlockMask", L: int) -> torch.Tensor:
        q = torch.arange(L)[:, None]
        k = torch.arange(L)[None, :]
        b = torch.tensor(0)
        h = torch.tensor(0)
        return bm.mask_mod(b, h, q, k)

    def _patched_flex_attention(query, key, value, block_mask, enable_gqa=True,
                                return_lse=False, kernel_options=None, **kw):
        """SDPA fallback for flex_attention.

        ★ Lesson E (STAGE 7 §3, re-discovered at STAGE 9 §0):
          When `block_mask` is a *floating-dtype* additive mask (0 = attend,
          -inf = block), the `dense.to(torch.bool)` conversion below would
          invert polarity (`-inf → True (truthy)`, `0 → False`).  Down-stream
          attention then attends to the *blocked* positions and ignores the
          attended ones — silently producing wrong-output that's only
          detectable via "different inputs → same output" cross-checks.

          Defence (contract): callers MUST pass `bool` dense or a `BlockMask`.
          `pt_layer_mask(dense_bool)` enforces this with an `assert`.
        """
        assert query.dim() == 4
        n_h = query.shape[1]
        n_kv = key.shape[1]
        q4, k4, v4 = query, key, value
        if enable_gqa and n_kv < n_h:
            rep = n_h // n_kv
            k4 = k4.repeat_interleave(rep, dim=1)
            v4 = v4.repeat_interleave(rep, dim=1)
        L_q = query.shape[2]
        if isinstance(block_mask, BlockMask):
            dense = _dense_from_block_mask(block_mask, L_q)
        else:
            dense = block_mask
        if dense.dtype != torch.bool:
            # Lesson E trap — caller bypassed pt_layer_mask().  Raising here
            # rather than silently converting (was a docstring scolding, now
            # a runtime exception per STAGE 9 reviewer BLOCKING D fix).
            raise RuntimeError(
                f"_patched_flex_attention: Lesson E trap — block_mask dtype="
                f"{dense.dtype}, expected bool dense.  A caller bypassed "
                f"pt_layer_mask().  Find the caller and route via pt_layer_mask."
            )
        add = torch.zeros(dense.shape, dtype=q4.dtype, device=q4.device)
        add.masked_fill_(~dense, float("-inf"))
        attn_mask = add[None, None, :, :]
        return F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask)

    _fa.flex_attention = _patched_flex_attention


def pt_layer_mask(dense_bool: "torch.Tensor") -> "torch.Tensor":
    """Lesson E contract: PT Lance layer 의 attention_mask 인자는 *bool dense* 만.

    ★ 절대 bf16 additive (0/-inf) 를 layer 에 직접 전달하지 말 것 ★
       `_patched_flex_attention` 안의 `dense.to(torch.bool)` 가 -inf 를 True
       (truthy) 로 변환 → attention polarity 반전 → 모든 입력에 대해 같은 v_t
       (입력 무관 silent bug).  STAGE 7 §3 Lesson E 의 재발화 경로.

    이 함수는 명시적으로 `bool` dtype 만 받는다.  실수로 additive 를 전달하면
    즉시 *runtime exception* (TypeError) — silent path 차단.  `python -O` 에서
    strip 되는 `assert` 대신 raise 사용 (STAGE 9 reviewer BLOCKING D fix).

    Usage:
        for layer in pt.layers:
            h = layer(..., attention_mask=pt_layer_mask(attn_dense_bool), ...)
    """
    import torch
    if not isinstance(dense_bool, torch.Tensor):
        raise TypeError(
            f"pt_layer_mask expects torch.Tensor, got {type(dense_bool).__name__}"
        )
    if dense_bool.dtype != torch.bool:
        raise TypeError(
            f"pt_layer_mask: Lesson E contract violated — expected bool dense, "
            f"got dtype={dense_bool.dtype}.  Do NOT pass additive (bf16 0/-inf) "
            f"masks directly; flex_attention SDPA patch's dense.to(bool) inverts "
            f"polarity (-inf → True). Build bool dense from your predicate and "
            f"pass that.  Reference: STAGE 7 §3 Lesson E + STAGE 9 §0 재발화."
        )
    return dense_bool
