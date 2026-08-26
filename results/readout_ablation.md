| Readout Method | Description / Mechanics | Trainable Params | Macro mAP | Micro mAP | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `MEAN` | Flat average over all $T \times D$ tokens | 15,420 | 0.289 | 0.501 | 0.131 |
| `MEAN-MAX` | Concatenates mean + coordinate-wise max | 30,780 | 0.260 | 0.458 | 0.057 |
| **`ATTENTIVE`** | **Softmax-weighted 2-layer scoring net** | **48,444** | **0.341** | **0.529** | **0.196** |
| `FACTORIZED` | Separate spatial & temporal scorers | 81,468 | 0.338 | **0.529** | **0.204** |