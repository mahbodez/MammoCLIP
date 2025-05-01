import os
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderConfig
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Callable
from transformers.models.clip.modeling_clip import CLIPOutput, CLIPVisionModel
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.models.clip.modeling_clip import clip_loss
from .blocks import AttentionFusion, ViewEmbedding
import logging
from typing import Literal
from safetensors.torch import load_file, save_file
import json

logger = logging.getLogger(__name__)


class MammoCLIPConfig(VisionTextDualEncoderConfig):
    model_type = "mammo_clip"

    def __init__(
        self,
        num_views=4,
        fusion_type: Literal["linear", "attention"] = "linear",
        **kwargs,
    ):
        self.num_views = num_views
        self.fusion_type = fusion_type
        super().__init__(**kwargs)

    @classmethod
    def from_vision_text_configs(cls, vision_config, text_config, **kwargs):
        config = super().from_vision_text_configs(vision_config, text_config, **kwargs)
        config.num_views = kwargs.pop("num_views", 4)
        config.fusion_type = kwargs.pop("fusion_type", "linear")
        return config


class MammoCLIP(VisionTextDualEncoderModel):
    def __init__(
        self,
        config: MammoCLIPConfig,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        verbose: bool = False,
        printing_func: Optional[Callable] = None,
    ):
        super().__init__(config, vision_model, text_model)
        if config.fusion_type == "attention":
            self.vision_fusion = AttentionFusion(
                embedding_dim=config.vision_config.hidden_size
            )
        elif config.fusion_type == "linear":
            self.vision_fusion = nn.Linear(
                config.vision_config.hidden_size * config.num_views,
                config.vision_config.hidden_size,
                bias=False,
            )
        else:
            raise ValueError(
                f"Unknown fusion type {config.fusion_type}. Supported types are 'linear' and 'attention'."
            )
        self.view_embedding = ViewEmbedding(
            num_views=config.num_views, embedding_dim=config.vision_config.hidden_size
        )
        self.verbose = verbose
        self.print = printing_func or print

        self._print(f"Model initialized with {config.num_views} views.")

    def _print(self, *args, **kwargs):
        if self.verbose:
            self.print(*args, **kwargs)

    def enable_verbose(self, printing_func: Optional[Callable] = None):
        self.verbose = True
        self.print = printing_func or print
        self.print("Verbose mode enabled. Use 'disable_verbose' to turn it off.")

    def disable_verbose(self):
        self.verbose = False
        self.print = print
        self.print("Verbose mode disabled. Use 'enable_verbose' to turn it on.")

    def _stack_images(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Processes multi-view image tensors by passing each view through the vision model and stacking the resulting feature representations.

        Args:
            pixel_values (torch.FloatTensor): Input tensor of shape (batch_size, num_views, height, width) representing multi-view images.
            output_attentions (Optional[bool], optional): Whether to return attention weights from the vision model. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to return hidden states from the vision model. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return outputs as a dictionary. Defaults to None.

        Returns:
            torch.FloatTensor: Tensor of shape (batch_size, num_views, hidden_size) containing the stacked feature representations for each view.
        """
        bs, c, h, w = pixel_values.shape
        self._print(f"Processing images with shape: {pixel_values.shape} (bs, c, h, w)")

        stacked_image_features = []
        for i in range(c):
            # Process each view separately
            view = pixel_values[:, i, :, :].view(bs, 1, h, w)
            # since the vision model expects n channels we need to repeat the view n times
            view = view.repeat(1, self.vision_model.config.num_channels, 1, 1)
            # now the shape is (bs, n, h, w)
            output = self.vision_model(
                pixel_values=view,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[1]
            # pooler_output
            # output is of shape (bs, hidden_size)
            stacked_image_features.append(output)
        # now we have a list of outputs for each view
        # we need to stack them to get a tensor of shape (bs, n_views, hidden_size)
        stacked_image_features = torch.stack(
            stacked_image_features, dim=1
        )  # (bs, n_views, hidden_size)
        return stacked_image_features

    def _embed_images(
        self,
        stacked_images: torch.FloatTensor,
    ):
        """
        Embeds a batch of stacked images using a learnable view embedding.

        Args:
            stacked_images (torch.FloatTensor): A tensor of shape (batch_size, n_views, hidden_size)
                representing a batch of images, where each image may have multiple views.

        Returns:
            torch.FloatTensor: The embedded images tensor after applying the view embedding,
                typically of the same shape as the input but with embedded representations.

        Logs:
            Prints the shapes of the input stacked images and the resulting embedded images for debugging.
        """
        # get stacked images of shape (bs, n_views, hidden_size)
        # and apply the view embedding
        # Now we add positional (per-view) learnable embeddings
        embedded_images = self.view_embedding(stacked_images)
        self._print(
            f"Stacked images shape: {stacked_images.shape}, Embedded images shape: {embedded_images.shape}"
        )
        return embedded_images

    def _fuse_images(
        self,
        embedded_images: torch.FloatTensor,
    ):
        """
        Fuses multiple embedded image representations into a single embedding using attention.

        Args:
            embedded_images (torch.FloatTensor): A tensor containing embedded representations of images,
                typically of shape (batch_size, n_views, hidden_size).

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - fused_embedding: The fused image embedding tensor of shape (batch_size, hidden_size).
                - attn_weights: The attention weights tensor of shape (batch_size, n_views), indicating
                  the contribution of each view to the fused embedding.

        Logs:
            Prints the shapes of the fused embedding and attention weights for debugging purposes.
        """
        B, N, D = embedded_images.shape
        # now we can use the attention fusion to get a single embedding
        if isinstance(self.vision_fusion, AttentionFusion):
            fused_embedding, attn_weights = self.vision_fusion(embedded_images)
        elif isinstance(self.vision_fusion, nn.Linear):
            embedded_images = embedded_images.reshape(B, N * D)
            fused_embedding = self.vision_fusion(embedded_images)
            attn_weights = None
        else:
            raise ValueError(
                "Currently only AttentionFusion and Linear fusion are supported."
                f" Got {type(self.vision_fusion)}"
            )
        # fused_embedding is of shape (bs, hidden_size)
        # attn_weights is of shape (bs, n_views)
        self._print(f"Fused embedding shape: {fused_embedding.shape}")
        if attn_weights is not None:
            self._print(f"Attention weights shape: {attn_weights.shape}")
        return fused_embedding, attn_weights

    def _process_images(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Processes input images through stacking, embedding, and fusion steps.

        Args:
            pixel_values (torch.FloatTensor): Input tensor containing image pixel values.
            output_attentions (Optional[bool], optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to return hidden states. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return outputs as a dictionary. Defaults to None.

        Returns:
            dict: A dictionary containing:
                - 'stacked_images': The stacked image tensor.
                - 'embedded_images': The embedded image representations.
                - 'fused_embedding': The fused image embedding.
                - 'attn_weights': The attention weights from the fusion step.
        """
        # get the stacked images
        stacked_images = self._stack_images(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # embed the images
        embedded_images = self._embed_images(stacked_images)
        # fuse the images
        fused_embedding, attn_weights = self._fuse_images(embedded_images)
        return dict(
            stacked_images=stacked_images,
            embedded_images=embedded_images,
            fused_embedding=fused_embedding,
            attn_weights=attn_weights,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        """
        Performs a forward pass through the model, encoding both text and image inputs, projecting them into a shared embedding space, and computing similarity logits.

        Args:
            input_ids (Optional[torch.LongTensor]): Input token IDs for the text encoder.
            pixel_values (Optional[torch.FloatTensor]): Input image tensor for the vision encoder.
            attention_mask (Optional[torch.Tensor]): Attention mask for the text encoder.
            position_ids (Optional[torch.LongTensor]): Position IDs for the text encoder.
            return_loss (Optional[bool]): Whether to compute and return the contrastive loss.
            token_type_ids (Optional[torch.LongTensor]): Token type IDs for the text encoder.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary or a tuple.

        Returns:
            Union[Tuple[torch.Tensor], CLIPOutput]:
                If `return_dict` is True, returns a `CLIPOutput` containing loss (if computed), logits, embeddings, and model outputs. Otherwise, returns a tuple of these values.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # fused_embedding
        image_embeds = self._process_images(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )["fused_embedding"]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                # vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            # vision_model_output=vision_outputs,
        )

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path=None,
        text_model_name_or_path=None,
        *model_args,
        **kwargs,
    ):

        kwargs_vision = {
            argument[len("vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(
                    vision_model_name_or_path, *model_args, **kwargs_vision
                )
                # TODO: Should we use the pre-trained projection as well ?
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(
                    vision_model_name_or_path, *model_args, **kwargs_vision
                )

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(
                text_model_name_or_path, *model_args, **kwargs_text
            )
        num_views = kwargs.pop("num_views", 4)
        fusion_type = kwargs.pop("fusion_type", "linear")
        # instantiate config with corresponding kwargs
        config = MammoCLIPConfig.from_vision_text_configs(
            vision_config=vision_model.config,
            text_config=text_model.config,
            num_views=num_views,
            fusion_type=fusion_type,
            **kwargs,
        )

        # init model
        model = cls(
            config=config,
            vision_model=vision_model,
            text_model=text_model,
            **kwargs,
        )

        # the projection layers are always newly initialized when loading the model
        # using pre-trained vision and text model.
        logger.warning(
            "The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight',"
            " 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
    ):
        """
        Loads a pretrained model from a directory containing a safetensors file.

        Args:
            pretrained_model_name_or_path (str): Path to the directory containing the pretrained model's safetensors file.

        Returns:
            An instance of the model with weights loaded from the safetensors file.

        Raises:
            ValueError: If no safetensors file is found in the specified directory.

        Notes:
            - The method first creates an untrained model instance using the provided path.
            - It then loads the weights from the safetensors file and applies them to the model.
            - Warnings are logged if there are missing or unexpected keys in the state dictionary.
        """
        # find the safetensors file in the directory
        safetensors_file = None
        for file in os.listdir(pretrained_model_name_or_path):
            if file.endswith(".safetensors"):
                safetensors_file = os.path.join(pretrained_model_name_or_path, file)
                break
        if safetensors_file is None:
            raise ValueError(
                f"No safetensors file found in {pretrained_model_name_or_path}"
            )
        # load the config file
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            # create the config object
            config = MammoCLIPConfig.from_dict(config_dict)
        else:
            raise ValueError(f"No config file found in {pretrained_model_name_or_path}")
        # create an untrained instance of the model,
        # including the vision/text backbones from the saved config
        vision_model = AutoModel.from_config(config.vision_config)
        text_model = AutoModel.from_config(config.text_config)
        instance = cls(config, vision_model=vision_model, text_model=text_model)
        # now we need to load the weights from the safetensors file
        state_dict = load_file(safetensors_file)
        # apply the weights non‐strictly so we can log missing/unexpected
        missing, unexpected = instance.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            logger.warning(
                f"Missing keys in state_dict: {missing}. This may be due to the model being untrained."
            )
        if len(unexpected) > 0:
            logger.warning(
                f"Unexpected keys in state_dict: {unexpected}. This may be due to the model being untrained."
            )
        # now we can return the model
        return instance

    def save_pretrained(self, save_directory: str):
        """
        Save the model's state_dict and configuration to a directory.
        named `model.safetensors` and `config.json` respectively.

        Args:
            save_directory (str): The directory where the model and config will be saved, if it doesn't exist, it will be created.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f"save_directory should be a directory, not a file: {save_directory}"
            )
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "model.safetensors")
        config_path = os.path.join(save_directory, "config.json")
        # save weights and config, raise on error
        try:
            save_file(self.state_dict(), model_path)
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f)
        except Exception as e:
            raise ValueError(f"Error saving model: {e}")

        self._print(f"Model saved to {model_path} and config saved to {config_path}")
