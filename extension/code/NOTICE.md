# Source attribution and adaptation

This standalone package copies and adapts the completed course MiniGPT code in
the parent repository, based on nanoGPT (MIT). The original MIT notice is in
`LICENSE` beside this file and must be retained when redistributing the code.
The parent repository snapshot is commit
`e80d3bad38762cceaa64d009ca5d5c0ec95f68ae`.

The attention, pre-norm block, embedding/LM-head tying, generation behavior,
fixed-data pipeline, AdamW setup, schedule, token accounting, evaluation windows,
and sampling prompts follow that implementation. The public contract tests and
plotter are adapted from its tests/utilities.

New extension work includes a configurable GELU/SiLU/SwiGLU FFN, budget-matched
hidden dimensions, exact parameter accounting, independent model/configuration
and checkpoint loading, standalone CLIs, explicit effective precision, source
hash validation on resume, extra tests, formal/smoke configs and an end-to-end
smoke/resume runner. The core package is not imported at runtime; one optional
test imports it only to verify GELU equivalence when the parent repo exists.

SwiGLU follows Noam Shazeer, *GLU Variants Improve Transformer* (2020),
https://arxiv.org/abs/2002.05202. This is a small decoder-only, biased FFN
adaptation with the course dropout and initialization rules, not a complete
reproduction of the paper's T5 experiments. Additional references:

- Hendrycks and Gimpel, *Gaussian Error Linear Units (GELUs)* (2016),
  https://arxiv.org/abs/1606.08415.
- Ramachandran, Zoph and Le, *Searching for Activation Functions* (2017),
  https://arxiv.org/abs/1710.05941.
