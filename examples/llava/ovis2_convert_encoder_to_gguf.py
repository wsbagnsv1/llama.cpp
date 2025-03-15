#!/usr/bin/env python3
import gguf
import argparse
import logging
import sys
import torch
import json
import os
import numpy as np
from typing import cast, ContextManager, Any, Iterator, Dict
from pathlib import Path
from torch import Tensor
from transformers import AutoProcessor

logger = logging.getLogger("ovis2-mmproj")


# (copied from convert_hf_to_gguf.py)
# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def to_gguf_name(name: str) -> str:
    og = name
    name = name.replace("text_model", "t").replace("vision_model", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_")
    name = name.replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("proj.", "out.")
    name = name.replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("multimodal_projector.", "mm.")
    logger.info(f"[to_gguf_name] {og} --> {name}")
    return name


class Ovis2VisionTower:
    hparams: dict
    gguf_writer: gguf.GGUFWriter
    fname_out: Path
    ftype: gguf.LlamaFileType

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)
        part_names.sort()
        return part_names

    def __init__(self,
                 dir_model: Path,
                 fname_out: Path,
                 ftype: gguf.LlamaFileType,
                 is_big_endian: bool,):
        hparams = Ovis2VisionTower.load_hparams(dir_model)
        self.hparams = hparams
        self.fname_out = fname_out
        self.ftype = ftype
        endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="clip", endianess=endianess)

        text_config = hparams["text_config"]
        vision_config = hparams["vision_config"]

        assert hparams["architectures"][0] == "Ovis2ForConditionalGeneration"
        assert text_config is not None
        assert vision_config is not None

        self.gguf_writer.add_string("clip.projector_type", "ovis2")
        self.gguf_writer.add_bool("clip.has_text_encoder", False)
        self.gguf_writer.add_bool("clip.has_vision_encoder", True)
        self.gguf_writer.add_bool("clip.has_llava_projector", False)  # legacy
        self.gguf_writer.add_uint32("clip.vision.image_size", vision_config["image_size"])
        self.gguf_writer.add_uint32("clip.vision.patch_size", vision_config["patch_size"])
        self.gguf_writer.add_uint32("clip.vision.embedding_length", vision_config["hidden_size"])
        self.gguf_writer.add_uint32("clip.vision.feed_forward_length", vision_config["intermediate_size"])
        self.gguf_writer.add_uint32("clip.vision.projection_dim", text_config["hidden_size"])
        self.gguf_writer.add_uint32("clip.vision.block_count", vision_config["num_hidden_layers"])
        self.gguf_writer.add_uint32("clip.vision.attention.head_count", vision_config["num_attention_heads"])
        self.gguf_writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_config.get("layer_norm_eps", 1e-6))
        # default values taken from HF transformers code
        self.gguf_writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
        self.gguf_writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
        self.gguf_writer.add_bool("clip.use_gelu", True)

        # load tensors
        for name, data_torch in self.get_tensors(dir_model):
            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            self.add_tensor(name, data_torch)

    def find_vision_tensors(self, model_dict) -> Dict[str, np.ndarray]:
        tensor_map = {}
        for name, ten in model_dict.items():
            # Skip non-vision tensors
            if not name.startswith("vision_model.") and not name.startswith("multimodal_projector."):
                continue
                
            # Handle QKV tensors if needed
            if "qkv" in name:
                if ten.ndim == 2:  # weight
                    c3, _ = ten.shape
                else:              # bias
                    c3 = ten.shape[0]
                assert c3 % 3 == 0
                c = c3 // 3
                wq = ten[:c]
                wk = ten[c: c * 2]
                wv = ten[c * 2:]
                tensor_map[to_gguf_name(name).replace("qkv", "q")] = wq
                tensor_map[to_gguf_name(name).replace("qkv", "k")] = wk
                tensor_map[to_gguf_name(name).replace("qkv", "v")] = wv
            else:
                tensor_map[to_gguf_name(name)] = ten
                
        return tensor_map

    def get_tensors(self, dir_model: Path) -> Iterator[tuple[str, Tensor]]:
        part_names = Ovis2VisionTower.get_model_part_names(dir_model, "model", ".safetensors")
        tensor_names_from_parts: set[str] = set()
        for part_name in part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            from safetensors import safe_open
            ctx = cast(ContextManager[Any], safe_open(dir_model / part_name, framework="pt", device="cpu"))
            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    # Skip non-vision tensors
                    if not name.startswith("vision_model.") and not name.startswith("multimodal_projector."):
                        continue
                        
                    data = model_part.get_slice(name)
                    data = LazyTorchTensor.from_safetensors_slice(data)
                    yield name, data

    def add_tensor(self, name: str, data_torch: Tensor):
        is_1d = len(data_torch.shape) == 1
        is_embd = ".embeddings." in name
        old_dtype = data_torch.dtype
        can_quantize = not is_1d and not is_embd
        data_qtype = gguf.GGMLQuantizationType.F32

        # filter only vision tensors
        if not name.startswith("vision_model.") and not name.startswith("multimodal_projector."):
            return
            
        # prefix
        name = name.replace("vision_model.encoder.layers.", "v.blk.")
        name = name.replace("vision_model.", "v.")
        # projector and input embd
        name = name.replace(".embeddings.patch_embedding.", ".patch_embd.")
        name = name.replace(".embeddings.position_embedding.", ".position_embd.")
        name = name.replace(
            "multimodal_projector.mm_input_projection_weight",
            "mm.input_projection.weight"
        )
        name = name.replace(
            "multimodal_projector.mm_soft_emb_norm.weight",
            "mm.soft_emb_norm.weight"
        )
        name = name.replace("post_layernorm.", "post_ln.")
        # each block
        name = name.replace(".self_attn.k_proj.", ".attn_k.")
        name = name.replace(".self_attn.v_proj.", ".attn_v.")
        name = name.replace(".self_attn.q_proj.", ".attn_q.")
        name = name.replace(".self_attn.out_proj.", ".attn_out.")
        name = name.replace(".layer_norm1.", ".ln1.")
        name = name.replace(".layer_norm2.", ".ln2.")
        name = name.replace(".mlp.fc1.", ".ffn_down.")
        name = name.replace(".mlp.fc2.", ".ffn_up.")

        # quantize
        if self.ftype != gguf.LlamaFileType.MOSTLY_F32 and can_quantize:
            if self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                data_torch = data_torch.to(torch.float16)
                data_qtype = gguf.GGMLQuantizationType.F16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                data_torch = gguf.quantize_q8_0(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q8_0
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_1:
                data_torch = gguf.quantize_q4_1(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q4_1
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_0:
                data_torch = gguf.quantize_q5_0(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q5_0
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_1:
                data_torch = gguf.quantize_q5_1(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q5_1
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q2_K:
                data_torch = gguf.quantize_q2_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q2_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_S:
                data_torch = gguf.quantize_q3_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q3_K_S
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_M:
                data_torch = gguf.quantize_q3_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q3_K_M
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L:
                data_torch = gguf.quantize_q3_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q3_K_L
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_S:
                data_torch = gguf.quantize_q4_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q4_K_S
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_M:
                data_torch = gguf.quantize_q4_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q4_K_M
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_S:
                data_torch = gguf.quantize_q5_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q5_K_S
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_M:
                data_torch = gguf.quantize_q5_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q5_K_M
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q6_K:
                data_torch = gguf.quantize_q6_k(data_torch)
                data_qtype = gguf.GGMLQuantizationType.Q6_K