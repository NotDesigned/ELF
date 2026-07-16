"""Sentence-level plan encoders used by ELF fusion experiments."""

from typing import Iterable, List

import torch


class SentenceT5PlanEncoder:
    """Frozen Sentence-T5 encoder with ELF-style latent normalization."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence_encoder_type='sentence_t5' requires sentence-transformers. "
                "Install requirements.txt or set use_sentence_plan=False."
            ) from exc

        self.model = SentenceTransformer(model_name, device=str(device))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.latent_mean = float(latent_mean)
        self.latent_std = float(latent_std)
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

    @torch.no_grad()
    def encode(self, texts: Iterable[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        text_list: List[str] = [text if isinstance(text, str) else "" for text in texts]
        embeddings = self.model.encode(
            text_list,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        embeddings = embeddings.to(device=device, dtype=dtype)
        return (embeddings - self.latent_mean) / max(self.latent_std, 1e-8)


def build_sentence_plan_encoder(config, device: torch.device):
    """Build an external sentence encoder when the fusion config asks for one."""
    if not bool(getattr(config, "use_sentence_plan", False)):
        return None
    encoder_type = getattr(config, "sentence_encoder_type", "sentence_t5")
    if encoder_type == "sentence_t5":
        return SentenceT5PlanEncoder(
            model_name=getattr(config, "sentence_t5_model_name", "sentence-transformers/sentence-t5-xl"),
            device=device,
            latent_mean=getattr(config, "sentence_latent_mean", 0.0),
            latent_std=getattr(config, "sentence_latent_std", 1.0),
        )
    if encoder_type == "learned":
        return None
    raise ValueError(f"Unknown sentence_encoder_type: {encoder_type}")
