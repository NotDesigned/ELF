# Frozen experiment contract

Before any submission, campaign `nanogpt-dropout-20260711T182322Z` freezes one factor: dropout. Control is `0.0`; treatment is `0.2`. Everything else is identical: nanoGPT upstream commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, Shakespeare-char data checksums recorded in each manifest, seed 1337, 4 layers, 4 heads, embedding width 128, block size 128, batch 64, one accumulation step, 300 optimizer steps, eval at steps 0/100/200/300 with 50 batches, and the same AdamW/cosine schedule.

Hypothesis: at this short budget, dropout 0.2 increases step-300 validation loss by at least 0.020 relative to dropout 0.0. Primary metric is finite validation loss at matched step 300. Supported if treatment minus control is at least +0.020; falsified if it is at most 0; inconclusive otherwise or if either run lacks scheduler COMPLETED+0:0, the matched finite metric, or a readable checkpoint.
