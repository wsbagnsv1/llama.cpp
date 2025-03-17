# Ovis2 model implementation for convert_hf_to_gguf.py
# This file contains the implementation of the Ovis2 model class
# To use this, add the following code to convert_hf_to_gguf.py

"""
@Model.register("Ovis")
class OvisModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA
    has_vision: bool = True

    def __init__(self, *args, **kwargs):
        dir_model = kwargs["dir_model"]
        hparams = Model.load_hparams(dir_model)
        
        if "llm_config" in hparams:
            merged_hparams = {**hparams, **hparams["llm_config"]}
            # Add tokenizer path from merged config
            merged_hparams.setdefault("vocab_file", os.path.join(dir_model, "llm", "tokenizer.model"))
            kwargs["hparams"] = merged_hparams
        
        super().__init__(*args, **kwargs)
        # Initialize vocab sizes from config
        self.original_vocab_size = self.hparams['llm_config'].get('vocab_size', 151936)
        self.padded_vocab_size = self.original_vocab_size
        
        # Track all tensors to ensure they're properly processed
        self.all_tensor_names = set()

    def set_vocab(self):
        # Fix: Check for existing tokenizer model in a way compatible with gguf_writer structure
        tokenizer_already_set = False
        try:
            # Check if tokenizer model is already set by looking for the field
            for item in self.gguf_writer.kv_data:
                if hasattr(item, 'key') and item.key == 'tokenizer.ggml.model':
                    tokenizer_already_set = True
                    break
        except (AttributeError, TypeError):
            # Fallback if kv_data structure is different
            tokenizer_already_set = False
        
        if not tokenizer_already_set:
            try:
                # Handle nested tokenizer path
                if "llm/" in self.hparams.get("vocab_file", ""):
                    self.dir_model = os.path.dirname(os.path.dirname(self.hparams["vocab_file"]))
                super().set_vocab()
            except FileNotFoundError as fnfe:
                logger.warning(f"Falling back to GPT-2 tokenizer: {str(fnfe)}")
                try:
                    self._set_vocab_gpt2()
                except ValueError as ve:
                    if "Duplicated key name" in str(ve):
                        logger.info("Tokenizer already exists, skipping duplicate addition")
                    else:
                        raise
                
                # Try to add special tokens from available configs
                try:
                    special_tokens_path = os.path.join(self.dir_model, "special_tokens_map.json")
                    config_path = os.path.join(self.dir_model, "tokenizer_config.json")
                    
                    if os.path.exists(special_tokens_path) and os.path.exists(config_path):
                        import json
                        with open(special_tokens_path, "r", encoding="utf-8") as f:
                            special_tokens = json.load(f)
                            if "bos_token" in special_tokens:
                                self.gguf_writer.add_special_token("bos_token", special_tokens["bos_token"])
                            if "eos_token" in special_tokens:
                                self.gguf_writer.add_special_token("eos_token", special_tokens["eos_token"])
                        
                        with open(config_path, "r", encoding="utf-8") as f:
                            tokenizer_config = json.load(f)
                            if "add_bos_token" in tokenizer_config:
                                self.gguf_writer.add_add_bos_token(tokenizer_config["add_bos_token"])
                            if "add_eos_token" in tokenizer_config:
                                self.gguf_writer.add_add_eos_token(tokenizer_config["add_eos_token"])
                except Exception as e:
                    logger.warning(f"Failed to add special tokens: {str(e)}")
        else:
            logger.info("Tokenizer already exists, skipping initialization")
        
        # Enhanced vocab size fallback
        if not all([self.original_vocab_size, self.padded_vocab_size]):
            embd_weight = next((v for k,v in self.tensor_map if "embed" in k), None)
            if embd_weight is not None:
                self.original_vocab_size = embd_weight.shape[0]
                self.padded_vocab_size = embd_weight.shape[0]
                logger.info(f"Derived vocab sizes from embeddings: {self.original_vocab_size}")
            else:
                raise ValueError("Could not determine vocabulary sizes from model or embeddings")
        
        # Add defensive validation
        if not hasattr(self, 'original_vocab_size') or not hasattr(self, 'padded_vocab_size'):
            raise AttributeError("Vocabulary attributes missing after parent class initialization")
            
        logger.info(f"Vocab sizes - original: {self.original_vocab_size}, padded: {self.padded_vocab_size}")
        
        # Add validation after vocab setup
        if not all([self.original_vocab_size, self.padded_vocab_size]):
            raise ValueError("Vocabulary sizes not initialized after set_vocab()")
        logger.info(f"Vocab sizes initialized - original: {self.original_vocab_size}, padded: {self.padded_vocab_size}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        logger.info(f"Processing tensor: {name} (shape: {data_torch.shape})")
        
        # Handle embeddings
        if name == "model.embed_tokens.weight" or name == "llm.model.embed_tokens.weight":
            logger.info(f"Processing embeddings tensor: {name}")
            if not all([self.original_vocab_size, self.padded_vocab_size]):
                raise ValueError("Vocabulary sizes not initialized")
            
            if data_torch.shape[0] < self.padded_vocab_size:
                logger.info(f"Padding embeddings from {data_torch.shape[0]} to {self.padded_vocab_size}")
                padding = torch.zeros((self.padded_vocab_size - data_torch.shape[0], data_torch.shape[1]), 
                                    dtype=data_torch.dtype, device=data_torch.device)
                data_torch = torch.cat([data_torch, padding], dim=0)
            
            tensor_name = self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD)
            self.all_tensor_names.add(tensor_name)
            return [(tensor_name, data_torch)]
        
        # Output layers
        if name == "llm.lm_head.weight":
            logger.info(f"Processing output layer: {name}")
            tensor_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
            self.all_tensor_names.add(tensor_name)
            return [(tensor_name, data_torch)]
        
        if name == "vte.weight":
            logger.info(f"Processing vision-text embedding: {name}")
            tensor_name = "clip.vision.mm_proj.output.weight"
            self.all_tensor_names.add(tensor_name)
            return [(tensor_name, data_torch)]
        
        # Tensor processing
        processed_name = name
        # Enhanced vision component handling
        if name.startswith("visual_tokenizer.backbone"):
            processed_name = name.replace("visual_tokenizer", "clip.vision.model")
        elif name.startswith("llm."):
            processed_name = name[4:]
        elif name.startswith(("vision_tower.", "visual_tokenizer.")):
            processed_name = name.replace("visual_tokenizer", "clip.vision.model")
        elif name.startswith("multi_modal_projector."):
            processed_name = f"clip.vision.mm_proj.{name}"
        
        # Apply specific vision encoder transformations
        processed_name = processed_name\
            .replace("preprocessor.patchifier", "patch_embd")\
            .replace("backbone", "bb")\
            .replace("trunk", "t")\
            .replace("blocks", "b")\
            .replace("pre_patch.norm", "patch_embd.norm")\
            .replace("pre_patch.proj", "patch_embd.proj")\
            .replace("position_embedding", "pos_embd")\
            .replace("attn.proj", "attn_out")\
            .replace("attn.qkv", "attn_qkv")

        # Handle multi-block layers
        if "t.b." in processed_name:
            processed_name = processed_name.replace("t.b.", "blk.")
            
        # Handle specific unmapped tensors
        if processed_name in [
            "clip.vision.model.bb.patch_embd.norm.weight",
            "clip.vision.model.bb.patch_embd.proj.bias",
            "clip.vision.model.bb.patch_embd.proj.weight"
        ]:
            logger.info(f"Handling special vision tensor: {processed_name}")
            self.all_tensor_names.add(processed_name)
            return [(processed_name, data_torch)]
            
        # Handle position embedding tensor
        if processed_name == "clip.vision.model.bb.preprocessor.pos_embed":
            tensor_name = "clip.vision.model.bb.pos_embd"
            self.all_tensor_names.add(tensor_name)
            return [(tensor_name, data_torch)]
            
        # Handle post-transformer normalization
        if processed_name == "clip.vision.model.bb.t.post_t_norm.weight":
            self.all_tensor_names.add(processed_name)
            return [(processed_name, data_torch)]
            
        # Handle vision model head layers
        if processed_name.startswith("clip.vision.model.head."):
            self.all_tensor_names.add(processed_name)
            return [(processed_name, data_torch)]
            
        # Handle attention output weights in blocks
        if "clip.vision.model.bb.blk." in processed_name:
            # Handle MLP layers in vision blocks
            if processed_name.endswith((".mlp.fc1.weight", ".mlp.fc2.weight", ".mlp.fc3.weight")):
                self.all_tensor_names.add(processed_name)
                return [(processed_name, data_torch)]
                
            # Handle normalization layers in vision blocks
            if processed_name.endswith((".norm_1.weight", ".norm_2.weight")):
                self.all_tensor_names.add(processed_name)
                return [(processed_name, data_torch)]
                
            # Handle attention output weights
            if processed_name.endswith(".attn_out.weight"):
                self.all_tensor_names.add(processed_name)
                return [(processed_name, data_torch)]

        # Name collision prevention (updated reserved names)
        reserved_names = {
            self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT),
            "clip.vision.mm_proj.output.weight"
        }
        
        try:
            mapped_name = self.map_tensor_name(processed_name)
            if mapped_name in reserved_names:
                logger.info(f"Skipping reserved tensor: {processed_name}")
                return []
        except ValueError:
            pass

        # Truncation handling
        if len(processed_name) >= 64:
            logger.warning(f"Long tensor name detected: {processed_name}")
            # Instead of truncating, create a hash-based name that's unique but shorter
            import hashlib
            name_hash = hashlib.md5(processed_name.encode()).hexdigest()[:8]
            base_name = processed_name.split('.')[-1]  # Get the last part of the name
            processed_name = f"tensor.{name_hash}.{base_name}"
            logger.warning(f"Renamed to: {processed_name}")
            processed_name = processed_name[:63]
            logger.warning(f"Truncated tensor name: {processed_name}")

        # Final mapping with vision-specific format
        try:
            # Handle QKV split if needed
            if "attn_qkv" in processed_name:
                logger.info(f"Splitting QKV tensor: {processed_name}")
                c3, _ = data_torch.shape
                assert c3 % 3 == 0
                c = c3 // 3
                wq = data_torch[:c]
                wk = data_torch[c:2*c]
                wv = data_torch[2*c:]
                
                q_name = processed_name.replace("attn_qkv", "attn_q")
                k_name = processed_name.replace("attn_qkv", "attn_k")
                v_name = processed_name.replace("attn_qkv", "attn_v")
                
                self.all_tensor_names.add(q_name)
                self.all_tensor_names.add(k_name)
                self.all_tensor_names.add(v_name)
                
                return [
                    (q_name, wq),
                    (k_name, wk),
                    (v_name, wv)
                ]
            
            mapped_name = self.map_tensor_name(processed_name)
            logger.info(f"Mapped tensor: {processed_name} -> {mapped_name}")
            
            if mapped_name in self.gguf_writer.tensors:
                logger.error(f"Duplicate tensor detected: {mapped_name}")
                raise ValueError(f"Duplicate tensor {mapped_name}")
                
            self.all_tensor_names.add(mapped_name)
            return [(mapped_name, data_torch)]
        except ValueError as ve:
            logger.warning(f"Keeping unmapped tensor: {processed_name} - Reason: {str(ve)}")
            self.all_tensor_names.add(processed_name)
            return [(processed_name, data_torch)]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # Vision parameters
        if "vision_config" in self.hparams:
            vc = self.hparams["vision_config"]
            self.gguf_writer.add_uint32("clip.vision.image_size", vc.get("image_size", 336))
            self.gguf_writer.add_uint32("clip.vision.patch_size", vc.get("patch_size", 14))
            self.gguf_writer.add_uint32("clip.vision.hidden_size", vc.get("hidden_size", 1024))
            self.gguf_writer.add_uint32("clip.vision.projection_dim", self.hparams["llm_config"]["hidden_size"])

    def write(self):
        # Before writing, ensure all tensors are properly marked for CPU compatibility
        for tensor_name in self.all_tensor_names:
            if tensor_name in self.gguf_writer.tensors:
                # Ensure tensor is marked for CPU compatibility
                tensor_info = self.gguf_writer.tensors[tensor_name]
                if hasattr(tensor_info, 'buffer_type') and tensor_info.buffer_type == 'CUDA_Host':
                    logger.info(f"Fixing buffer type for tensor: {tensor_name}")
                    tensor_info.buffer_type = 'CPU'
        
        # Check tensor count before writing
        tensor_count = len(self.all_tensor_names)
        logger.info(f"Total tensors to write: {tensor_count}")
        
        # Force CPU compatibility for all tensors
        try:
            # Add explicit CPU compatibility flag
            self.gguf_writer.add_uint32("general.architecture", int(gguf.MODEL_ARCH.LLAMA))
            self.gguf_writer.add_string("general.quantization_version", "2")
            self.gguf_writer.add_string("general.file_type", "7")
            self.gguf_writer.add_bool("general.cpu_compatible", True)
        except Exception as e:
            logger.warning(f"Failed to set CPU compatibility flags: {str(e)}")
        
        # Call parent write method
        super().write()
        logger.info("Multimodal model converted successfully")
        logger.info(f"Embedding dimensions: {self.padded_vocab_size} × {self.hparams['hidden_size']}")

"""
