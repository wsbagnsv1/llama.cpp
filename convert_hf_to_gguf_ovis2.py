# Ovis2 model implementation for convert_hf_to_gguf.py
# This file contains the implementation of the Ovis2 model class
# To use this, add the following code to convert_hf_to_gguf.py

"""
@Model.register("Ovis2ForConditionalGeneration")
class Ovis2Model(Model):
    model_arch = gguf.MODEL_ARCH.OVIS2
    has_vision: bool = False

    # we need to merge the text_config into the root level of hparams
    def __init__(self, *args, **kwargs):
        hparams = Model.load_hparams(kwargs["dir_model"])
        if "text_config" in hparams:
            hparams = {**hparams, **hparams["text_config"]}
            kwargs["hparams"] = hparams
        super().__init__(*args, **kwargs)
        if "vision_config" in hparams:
            logger.info("Has vision encoder, but it will be ignored")
            self.has_vision = True

    def write(self):
        super().write()
        if self.has_vision:
            logger.info("NOTE: this script only converts the language model to GGUF")
            logger.info("      for the vision model, please use ovis2_convert_encoder_to_gguf.py")

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        # some default values are not specified in the hparams
        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 131072))
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams.get("num_attention_heads", 8))
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_key_length(hparams.get("head_dim", 256))
        self.gguf_writer.add_value_length(hparams.get("head_dim", 256))
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(hparams.get("rope_theta", 1_000_000.0))
        self.gguf_writer.add_sliding_window(hparams.get("sliding_window", 0))
        self.gguf_writer.add_head_count_kv(hparams.get("num_key_value_heads", 4))
        if hparams.get("rope_scaling") is not None:
            if hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        if name.startswith("language_model."):
            name = name.replace("language_model.", "")
        elif name.startswith("multi_modal_projector.") or name.startswith("vision_tower.") \
                or name.startswith("multimodal_projector.") or name.startswith("vision_model."):
            # ignore vision tensors
            return []

        # remove OOV (out-of-vocabulary) rows in token_embd
        if "embed_tokens.weight" in name:
            vocab = self._create_vocab_sentencepiece()
            tokens = vocab[0]
            data_torch = data_torch[:len(tokens)]

        # handle RMS norm weights
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        return [(self.map_tensor_name(name), data_torch)]
"""