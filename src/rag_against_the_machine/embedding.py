from transformers import AutoModel, AutoTokenizer
import torch


class Embedder:
    """Lightweight CPU sentence embedding using all-MiniLM-L6-v2."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        """Initialize the sentence embedder model and tokenizer.

        Args:
            model_name: Hugging Face model repository or local path.
            device: Target hardware device ('cpu' or 'cuda'). Both model
                weights and tokenized tensors must reside on the same
                hardware device to execute tensor operations without errors.
                If None, defaults to 'cuda' when available, otherwise 'cpu'.
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.auto_model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        # Switches the neural network into inference mode.
        # During evaluation, dropout layers are disabled for determinism.
        self.auto_model.eval()

    def _mean_pooling(
        self,
        model_output: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sentence-level vector representation via mean pooling.

        Each chunk is mapped to a point with 384 coordinates in geometric
        space. This function computes the average of valid token vectors,
        ignoring padding tokens, producing a single 384-dimensional vector
        per sentence for geometric distance and similarity calculations.

        Args:
            model_output: Model predictions with token embeddings at index 0.
            attention_mask: Binary mask (1 for valid tokens, 0 for padding).
                Preserves batch matrix consistency while allowing pooling to
                ignore padding.

        Returns:
            torch.Tensor: Pooled sentence embeddings of shape
                [batch_size, 384].
        """

        # Apply the mask to each valid token:
        # Valid words are multiplied by 1 (their values stay exactly the same)
        # Padding tokens are multiplied by 0 (their values become all zeros)
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Compress token into one single vector:
        # add all the token vectors together into one single vector (dim=1)
        # Because padding tokens are now zeros, they add nothing to the total,
        # ensuring the mathematical sum is not corrupted.
        sum_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embedding_encoder(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode a list of texts into L2-normalized 384-d vectors.

        Args:
            texts: List of text strings to embed (chunks or query).
            batch_size: Number of texts per inference batch (default: 32).

        Returns:
            torch.Tensor: Normalized embeddings of shape [len(texts), 384].
        """
        if not texts:
            return torch.empty((0, 384), dtype=torch.float32)

        # Each tensor generated inside the for loop is a 2D matrix (a grid)
        # containing exactly 32 separate rows.
        # Each of those 32 rows is the dedicated 384-dimensional vector for
        # one specific text chunk.
        # all_embeddings data structure example:
        #       [tensor_1 (32 rows), tensor_2 (32 rows), ...].
        all_embeddings: list[torch.Tensor] = []

        # take 32(batch_size) chunks a time
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch:
            # - padding=True: equalizes lengths using PAD tokens
            # - truncation=True: clips sequences exceeding max_length
            #       Chunks have a dimension lower then 512 words
            #       this limit is necessary bc the model all-MiniLM-L6-v2
            #       takes a maximum of 512 position in the matrix
            # - return_tensors="pt": outputs PyTorch tensors
            # - .to(self.device):
            #       moves tensors to the same hardware device as the model
            #       to prevent PyTorch device mismatch RuntimeErrors.
            encoded_inputs = self.auto_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # inference_mode: use the neural network only in reading mode
            # encoded_input contains input_ids and attention_mask
            #   input_ids = numeric ID for each word in chunks
            #   attention_mask: binary mask (1[for each numeric ID] and 0)
            with torch.inference_mode():

                # The model processes the input_ids through its neural layers
                # and expands every single integer ID
                # into a continuous vector of 384 decimal coordinates.
                #       each ID = 1 row with 384 dimensions
                model_output = self.auto_model(**encoded_inputs)

                # _mean_pooling takes the 384 coordinates of each TOKEN from
                # model_output and averages them into a SINGLE 384-dim vector
                # per chunk, ignoring padding from encoded_inputs.
                embeddings = self._mean_pooling(
                    model_output,
                    encoded_inputs["attention_mask"]
                )

                # 1) turns CPU calculations during the search phase into a
                #    simple multiplication
                # 2) The raw vectors generated by the model have a fixed
                #    dimension (384) but arbitrary magnitudes (vector lengths)
                #    Magnitude: geometric distance of the vector from ZERO
                # 3) L2 normalization scales the length of every vector to
                #    exactly 1.0 while preserving its semantic dir (angle).
                # 4) By pre-normalizing here, we can compute cosine similarity
                #    later using a blazing-fast dot product, completely skip
                #    the heavy division operations during the search phase.
                # 5) normalize function calculates the true geometric length
                #    (the magnitude) of the vector and divides every single
                #    one of the 384 coordinates by that exact number.
                embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                )

            # ADD this batch (32, 384) to all_embeddings var
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
