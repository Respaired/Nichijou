# Training

Start with [configs/datasets.example.json](../configs/datasets.example.json). Each entry points to metadata and latent datasets joined by a shared `key`:

- **Metadata:** `key` and a text column containing phonemized dialogue with language and speaker tags.
- **Latents:** the matching `key` and a tensor column containing PCA-projected Fish codec latents, shaped `[frames, 80]`.

Set `text_column` and `latents_columns` to your column names. Dataset paths can point to Parquet files or supported dataset directories. For length training, use individual neighboring clips with numbered keys and `must_merge: false`.
