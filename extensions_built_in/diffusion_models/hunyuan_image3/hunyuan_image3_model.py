import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager

from .src import (
    HunyuanImage3ForCausalMM,
    HunyuanImage3Text2ImagePipeline,
    FlowMatchDiscreteScheduler,
    build_batch_2d_rope,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "reverse": True,
    "solver": "euler",
}


class HunyuanImage3Model(BaseModel):
    arch = "hunyuan_image3"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.is_ara = True
        self.target_lora_modules = ["HunyuanImage3Model", "HunyuanImage3ForCausalMM"]
        self.hunyuan_model: Optional[HunyuanImage3ForCausalMM] = None

    @staticmethod
    def get_train_scheduler():
        return FlowMatchDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading HunyuanImage 3.0 model")
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading HunyuanImage3ForCausalMM")

        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        attn_impl = self.model_config.model_kwargs.get('attn_implementation', 'sdpa')
        moe_impl = self.model_config.model_kwargs.get('moe_impl', 'eager')

        if attn_impl:
            load_kwargs['attn_implementation'] = attn_impl
        if moe_impl:
            load_kwargs['moe_impl'] = moe_impl

        hunyuan_model = HunyuanImage3ForCausalMM.from_pretrained(
            model_path, **load_kwargs
        )
        hunyuan_model.load_tokenizer(model_path)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Model")
            quantize_model(self, hunyuan_model.model)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                hunyuan_model.model,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving model to CPU")
            hunyuan_model.to("cpu")
        else:
            hunyuan_model.to(self.device_torch)

        flush()

        hunyuan_model.requires_grad_(False)
        hunyuan_model.eval()

        self.noise_scheduler = HunyuanImage3Model.get_train_scheduler()

        self.print_and_status_update("Setting up pipeline")

        pipeline = HunyuanImage3Text2ImagePipeline(
            model=hunyuan_model,
            scheduler=self.noise_scheduler,
            vae=hunyuan_model.vae,
        )

        self.hunyuan_model = hunyuan_model
        self.vae = hunyuan_model.vae
        self.text_encoder = [hunyuan_model]
        self.tokenizer = [hunyuan_model._tkwrapper.tokenizer if hunyuan_model._tkwrapper else None]
        self.model = hunyuan_model.model
        self.pipeline = pipeline
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = HunyuanImage3Model.get_train_scheduler()

        pipeline = HunyuanImage3Text2ImagePipeline(
            model=unwrap_model(self.hunyuan_model),
            scheduler=scheduler,
            vae=unwrap_model(self.vae),
        )

        return pipeline

    def generate_single_image(
        self,
        pipeline: HunyuanImage3Text2ImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.hunyuan_model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        width = int(gen_config.width // sc * sc)
        height = int(gen_config.height // sc * sc)

        prompt = extra.get("prompt", "")
        seed = extra.get("seed", None)

        image = self.hunyuan_model.generate_image(
            prompt=prompt,
            seed=seed,
            image_size=f"{height}x{width}",
            stream=False,
        )
        return image

    def _prepare_model_inputs_for_training(
        self,
        prompt: str,
        height: int,
        width: int,
    ) -> dict:
        if self.hunyuan_model._tkwrapper is None:
            raise ValueError("Tokenizer not loaded. Ensure model is loaded with load_tokenizer().")

        image_size = (height, width)

        batch_prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(batch_prompt)

        batch_gen_image_info = [
            self.hunyuan_model.image_processor.build_image_info(image_size)
            for _ in range(batch_size)
        ]

        out = self.hunyuan_model._tkwrapper.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=None,
            mode="gen_image",
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_image_info=None,
            batch_system_prompt=None,
            batch_cot_text=None,
            max_length=None,
            bot_task="image",
            image_base_size=self.hunyuan_model.config.image_base_size,
            sequence_template=self.hunyuan_model.generation_config.sequence_template,
            cfg_factor=1,
            drop_think=True,
        )
        output, sections = out['output'], out['sections']

        rope_image_info = self.hunyuan_model.build_batch_rope_image_info(output, sections)
        seq_len = output.tokens.shape[1]

        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=self.hunyuan_model.config.attention_head_dim,
            device=self.device_torch,
            base=self.hunyuan_model.config.rope_theta,
        )

        batch_input_pos = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=self.device_torch
        )[None].expand(batch_size, -1)

        bsz, seq_len_attn = output.tokens.shape
        batch_image_slices = [
            output.joint_image_slices[i] + output.gen_image_slices[i]
            for i in range(bsz)
        ]
        attention_mask = torch.ones(seq_len_attn, seq_len_attn, dtype=torch.bool).tril(diagonal=0)
        attention_mask = attention_mask.repeat(bsz, 1, 1)
        for i in range(bsz):
            for image_slice in batch_image_slices[i]:
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1).to(self.device_torch)

        return {
            'input_ids': output.tokens.to(self.device_torch),
            'position_ids': batch_input_pos,
            'custom_pos_emb': (cos.to(self.device_torch), sin.to(self.device_torch)),
            'image_mask': output.gen_image_mask.to(self.device_torch) if output.gen_image_mask is not None else None,
            'gen_timestep_scatter_index': output.gen_timestep_scatter_index.to(self.device_torch) if output.gen_timestep_scatter_index is not None else None,
            'attention_mask': attention_mask,
            'batch_gen_image_info': batch_gen_image_info,
        }

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        self.hunyuan_model.to(self.device_torch)

        batch_size, num_channels, lat_h, lat_w = latent_model_input.shape

        vae_scale_h = self.hunyuan_model.config.vae_downsample_factor[0]
        vae_scale_w = self.hunyuan_model.config.vae_downsample_factor[1]
        pixel_height = lat_h * vae_scale_h
        pixel_width = lat_w * vae_scale_w

        if batch is not None:
            prompts = batch.get_caption_list()
        else:
            prompts = [""] * batch_size

        text_embeddings = self.get_prompt_embeds(
            prompt=prompts[0] if len(prompts) == 1 else prompts,
            height=pixel_height,
            width=pixel_width,
        )

        input_ids = text_embeddings.text_embeds
        attention_mask = text_embeddings.attention_mask
        image_mask = getattr(text_embeddings, 'image_mask', None)
        gen_timestep_scatter_index = getattr(text_embeddings, 'gen_timestep_scatter_index', None)
        custom_pos_emb = getattr(text_embeddings, 'custom_pos_emb', None)
        position_ids = getattr(text_embeddings, 'position_ids', None)

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        timestep_scaled = timestep / 1000.0 if timestep.max() > 1.0 else timestep
        timestep_scaled = timestep_scaled.to(self.device_torch, dtype=self.torch_dtype)

        forward_kwargs = {
            'input_ids': input_ids.to(self.device_torch),
            'mode': "gen_image",
            'first_step': True,
            'images': latent_model_input.to(self.device_torch, dtype=self.torch_dtype),
            'timestep': timestep_scaled,
            'return_dict': True,
        }

        if attention_mask is not None:
            forward_kwargs['attention_mask'] = attention_mask.to(self.device_torch)
        if image_mask is not None:
            forward_kwargs['image_mask'] = image_mask.to(self.device_torch)
        if gen_timestep_scatter_index is not None:
            forward_kwargs['gen_timestep_scatter_index'] = gen_timestep_scatter_index.to(self.device_torch)
        if custom_pos_emb is not None:
            if isinstance(custom_pos_emb, tuple):
                forward_kwargs['custom_pos_emb'] = (
                    custom_pos_emb[0].to(self.device_torch),
                    custom_pos_emb[1].to(self.device_torch)
                )
            else:
                forward_kwargs['custom_pos_emb'] = custom_pos_emb.to(self.device_torch)
        if position_ids is not None:
            forward_kwargs['position_ids'] = position_ids.to(self.device_torch)

        outputs = self.hunyuan_model(**forward_kwargs)

        noise_pred = outputs.diffusion_prediction

        return noise_pred

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
    ) -> PromptEmbeds:
        model_inputs = self._prepare_model_inputs_for_training(
            prompt=prompt,
            height=height,
            width=width,
        )

        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        pe = PromptEmbeds(input_ids, attention_mask=attention_mask)
        pe.text_embeds = input_ids
        pe.image_mask = model_inputs.get('image_mask')
        pe.gen_timestep_scatter_index = model_inputs.get('gen_timestep_scatter_index')
        pe.custom_pos_emb = model_inputs.get('custom_pos_emb')
        pe.position_ids = model_inputs.get('position_ids')

        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        hunyuan_model = unwrap_model(self.hunyuan_model)
        hunyuan_model.save_pretrained(
            save_directory=output_path,
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "hunyuan_image3"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_key = new_key.replace("hunyuan_model.", "")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            vae_result = self.vae.encode(images)
            if isinstance(vae_result, torch.Tensor):
                latents = vae_result
            else:
                latents = vae_result.latent_dist.sample()

            config = self.vae.config
            if hasattr(config, 'shift_factor') and config.shift_factor:
                latents = latents - config.shift_factor
            if hasattr(config, 'scaling_factor') and config.scaling_factor:
                latents = latents * config.scaling_factor

        if hasattr(self.vae, "ffactor_temporal"):
            if latents.dim() == 5 and latents.shape[2] == 1:
                latents = latents.squeeze(2)

        return latents
