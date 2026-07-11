# nanoGPT dropout short-budget experiment

Decision: **SUPPORTED**.

The frozen hypothesis was that changing only dropout from 0.0 to 0.2 would raise validation loss by at least 0.020 after 300 optimizer steps. With the same hardcoded seed 1337, 0.80M-parameter architecture, data, optimizer schedule, 50-batch evaluations, immutable source, and SIF, control reached 2.2797 validation loss and treatment reached 2.4025. The treatment-minus-control delta is +0.1228, above the +0.020 support threshold.

Both initial step-0 evaluations were identical (train 4.1831, validation 4.1799), providing a strong pairing check before dropout affected training. Control job 1789 completed in 42 seconds and treatment job 1790 in 46 seconds; both were `COMPLETED` with exit `0:0` on one L40S. Each run has four finite evaluation records at steps 0/100/200/300 and a readable step-300 checkpoint with a verified SHA-256 completion marker.

This conclusion is scoped to one seed and this short training budget. It establishes the preregistered within-run comparison, not a general estimate across seeds or longer training.
