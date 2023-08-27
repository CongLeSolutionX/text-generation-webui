from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image


class AbstractMultimodalPipeline(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        'name of the pipeline, should be same as in --multimodal-pipeline'

    @staticmethod
    @abstractmethod
    def image_start() -> Optional[str]:
        'return image start string, string representation of image start token, or None if not applicable'

    @staticmethod
    @abstractmethod
    def image_end() -> Optional[str]:
        'return image end string, string representation of image end token, or None if not applicable'

    @staticmethod
    @abstractmethod
    def placeholder_token_id() -> int:
        'return placeholder token id'

    @staticmethod
    @abstractmethod
    def num_image_embeds() -> int:
        'return the number of embeds used by a single image (for example: 256 for LLaVA)'

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        'forward the images through vision pipeline, and return their embeddings'

    @staticmethod
    @abstractmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        'embed tokens, the exact function varies by LLM, for LLaMA it is `shared.model.model.embed_tokens`'

    @staticmethod
    @abstractmethod
    def placeholder_embeddings() -> torch.Tensor:
        'get placeholder embeddings if there are multiple images, and `add_all_images_to_prompt` is False'

    def _get_device(self, setting_name: str, params: dict):
        if params[setting_name] is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(params[setting_name])

    def _get_dtype(self, setting_name: str, params: dict):
        return torch.float32 if int(params[setting_name]) == 32 else torch.float16
