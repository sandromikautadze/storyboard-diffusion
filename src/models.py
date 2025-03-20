import torch
from diffusers import StableDiffusionPipeline

class MultiPromptPipelineApproach1(StableDiffusionPipeline):
    """
    Multi-Prompt CFG with a SINGLE unconditional pass:
      - At each diffusion step:
        1. uncond_out = UNet(latent, uncond_embeds)
        2. cond_out_i = UNet(latent, cond_embeds_i) for each subprompt i
        3. cond_combined = weighted average of all cond_out_i
        4. final_out = uncond_out + guidance_scale*(cond_combined - uncond_out)
    """

    @torch.no_grad()
    def __call__(
        self,
        subprompt_embeds: list[torch.Tensor],
        subprompt_weights: list[float],
        uncond_embeds: torch.Tensor,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs
    ):
        device = self._execution_device
        batch_size = uncond_embeds.shape[0]
        num_subprompts = len(subprompt_embeds)

        if num_subprompts != len(subprompt_weights):
            raise ValueError("subprompt_embeds and subprompt_weights must have the same length.")

        # 1. Validate or fallback to default height/width
        if not height or not width:
            height, width = self._default_height_width()

        # 2. Set timesteps on the scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latents
        if latents is None:
            shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
            latents = torch.randn(shape, generator=generator, device=device, dtype=uncond_embeds.dtype)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device)

        # 4. Diffusion loop
        for i, t in enumerate(timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # (A) Unconditional pass
            uncond_out = self.unet(latent_model_input, t, encoder_hidden_states=uncond_embeds, **kwargs).sample

            # (B) Conditional passes (one per subprompt)
            cond_outs = []
            for cond_embed in subprompt_embeds:
                out = self.unet(latent_model_input, t, encoder_hidden_states=cond_embed, **kwargs).sample
                cond_outs.append(out)

            # (C) Weighted average of conditional outputs
            total_w = sum(subprompt_weights)
            cond_combined = sum(w * o for w, o in zip(subprompt_weights, cond_outs)) / total_w

            # (D) Classifier-Free Guidance
            guided_out = uncond_out + guidance_scale * (cond_combined - uncond_out)

            # (E) Step
            latents = self.scheduler.step(guided_out, t, latents, **kwargs).prev_sample

        # 5. Decode latents
        if output_type == "latent":
            if return_dict:
                from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
                return StableDiffusionPipelineOutput(images=latents, nsfw_content_detected=None)
            return latents

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        if return_dict:
            from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
        return image
    
class MultiPromptPipelineApproach2(StableDiffusionPipeline):
    """
    Multi-Prompt CFG with MULTIPLE unconditional passes:
      - 1 global unconditional pass per step: e_uncond
      - For each subprompt i:
          e_uncond_i (subprompt-specific unconditional)
          e_cond_i    (subprompt conditional)
      - Combine: e = e_uncond + g * sum_i[ w_i * ( e_cond_i - e_uncond_i ) ]
    """

    @torch.no_grad()
    def __call__(
        self,
        global_uncond_embeds: torch.Tensor,
        subprompt_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        subprompt_weights: list[float],
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs
    ):
        """
        Args:
            global_uncond_embeds (Tensor): [batch, seq_len, hidden_dim] for the entire prompt's unconditional pass.
            subprompt_pairs (list of (uncond_i, cond_i)):
                Each element is a tuple: (uncond_embeds_i, cond_embeds_i).
            subprompt_weights (list[float]): Weights w_i for each subprompt i.
        """
        device = self._execution_device
        batch_size = global_uncond_embeds.shape[0]
        num_subprompts = len(subprompt_pairs)

        if num_subprompts != len(subprompt_weights):
            raise ValueError("subprompt_pairs and subprompt_weights must have the same length.")

        # 1. Validate or fallback to default
        if not height or not width:
            height, width = self._default_height_width()

        # 2. Scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latents
        if latents is None:
            shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
            latents = torch.randn(shape, generator=generator, device=device, dtype=global_uncond_embeds.dtype)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device)

        # 4. Diffusion loop
        for i, t in enumerate(timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # (A) Single global unconditional pass
            e_uncond_global = self.unet(
                latent_model_input, t, encoder_hidden_states=global_uncond_embeds, **kwargs
            ).sample

            # (B) For each subprompt: unconditional + conditional
            sub_deltas = []
            for (uncond_i, cond_i), w in zip(subprompt_pairs, subprompt_weights):
                e_uncond_i = self.unet(latent_model_input, t, encoder_hidden_states=uncond_i, **kwargs).sample
                e_cond_i = self.unet(latent_model_input, t, encoder_hidden_states=cond_i, **kwargs).sample

                # Delta for subprompt i
                delta_i = w * (e_cond_i - e_uncond_i)
                sub_deltas.append(delta_i)

            # (C) Combine sub-deltas
            sum_deltas = sum(sub_deltas)  # sum_i w_i ( e_cond_i - e_uncond_i )

            # (D) Final output
            guided_out = e_uncond_global + guidance_scale * sum_deltas

            # (E) Scheduler step
            latents = self.scheduler.step(guided_out, t, latents, **kwargs).prev_sample

        # 5. Decode
        if output_type == "latent":
            if return_dict:
                from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
                return StableDiffusionPipelineOutput(images=latents, nsfw_content_detected=None)
            return latents

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        if return_dict:
            from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
        return image
