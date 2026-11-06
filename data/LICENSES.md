# Data provenance and terms

TinyStories is by Ronen Eldan and Yuanzhi Li:
https://huggingface.co/datasets/roneneldan/TinyStories
Paper: https://arxiv.org/abs/2305.07759

Pinned source revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
Source files: `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`.
The upstream dataset card declares **Community Data License Agreement – Sharing 1.0**:
https://cdla.dev/sharing-1-0/

This course subset, split arrangement, and tokenized data are derived from that
dataset and are shared under CDLA-Sharing-1.0, not the code's MIT license.
Retain these notices and the accompanying full license when redistributing.
The data are synthetic, GPT-4-generated children's stories and can contain
errors, bias, or inappropriate content. Generation quality is not factual accuracy.

The tokenizer is trained from scratch on the first 20,000 selected training
stories only. Normalized exact duplicates are removed within/across the three
selected splits; this does not establish the absence of near duplicates.
See `manifest.json` for sizes and SHA-256 checksums. Teacher-only source hashes
and hidden data are deliberately not part of the public data directory.
