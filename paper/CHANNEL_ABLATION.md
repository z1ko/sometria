# Reference Guide: Reconstruction Targets & Linear Readouts

**Repository Reference Document**  
*Topic:* Masked-Reconstruction Pretext Objectives & Probe Readout Architecture Evaluation  
*Context:* Kinematic vs. Dynamic Reconstruction Targets & Readout Aggregation Benchmarks

---

## Key Quantitative Summary

| Experiment Axis | Key Metric Baseline | Standard/Kinematic Approach | Proposed Dynamic / Attentive | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Reconstruction Target** | Macro mAP (10ep) | 0.216 *(Full Kinematics, $d_{\text{head}}=32$)* | **0.297** *(All 5 Channels, $d_{\text{head}}=40$)* | Torque ($	au$) provides non-redundant $+0.08$ mAP gain |
| **Proxy Control ($\ddot\theta$ vs $	au$)** | Macro mAP (10ep) | 0.238 *($\{\sin\theta, \ddot\theta\}$, $d_{\text{head}}=16$)* | **0.259** *($\{\sin\theta, \tau\}$, $d_{\text{head}}=16$)* | Acceleration proxies ~70% of dynamics; inertia/gravity fill the rest |
| **Probe Readout Method** | Macro mAP (40ep) | 0.289 *(`MEAN` Pooling)* | **0.341** *(`ATTENTIVE` Pooling)* | Attentive readout yields +49% Macro F1 & $12\times$ lower seed spread |

---

## 1. Input Representation & Physics Formulation

Each degree of freedom is encoded with **5 channels per frame**:
* **Joint Angle:** $(\sin\theta, \cos\theta)$
* **First Derivative (Velocity):** $\dot\theta$
* **Second Derivative (Acceleration):** $\ddot\theta$
* **Generalized Joint Torque:** $\tau$ (recovered via inverse dynamics)

### Governing Equation of Motion
$$\tau = M(\theta)\,\ddot\theta + C(\theta,\dot\theta) + G(\theta)$$

* **Kinematic Proxy Role:** Acceleration $\ddot\theta$ is the kinematic derivative directly tied to force via mass matrix $M(\theta)$.
* **Dynamics Gap:** Full torque $	au$ encodes configuration-dependent inertia $M(\theta)$ and gravitational loads $G(\theta)$, which cannot be deduced from kinematics alone without explicit structural terms.

---

## 2. Pretext Loss Target Ablations (Table 1 Data)

* **Encoder Setup:** Encoder always reads all 5 input channels; only decoder prediction head and loss function are ablated.
* **Pretraining:** 10 epochs per arm (represents strict lower bounds; loss actively decreasing).
* **Readout Head:** Attentive pooling linear probe.

### Table 1: Loss Channel Transfer Performance

| Loss Channels | Content Type | $d_{\text{head}}$ | Macro mAP | Micro mAP | Macro F1 | Gap Recovered (Macro mAP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| *Chance (label prevalence)* | --- | --- | 0.042 | --- | --- | --- |
| *Random frozen encoder* | --- | --- | 0.128 | 0.334 | 0.021 | 0.0% |
| $\tau$ | Dynamic | 8 | 0.153 | 0.367 | 0.041 | 14.8% |
| $\sin\theta, \cos\theta$ | Kinematic | 16 | 0.180 | 0.392 | 0.036 | 30.8% |
| $\sin\theta, \ddot\theta$ | Kinematic + Proxy | 16 | 0.238 | 0.438 | 0.085 | 65.1% |
| $\sin\theta, \tau$ | Kinematic + Dynamic | 16 | 0.259 | 0.458 | 0.100 | 77.5% |
| $\sin\theta, \cos\theta, \tau$ | Kinematic + Dynamic | 24 | 0.261 | 0.460 | 0.116 | 78.7% |
| $\sin\theta, \cos\theta, \dot\theta, \ddot\theta$ | Kinematic Only | 32 | 0.216 | 0.426 | 0.074 | 52.1% |
| **All Five Channels** | **Kinematic + Dynamic** | **40** | **0.297** | **0.493** | **0.143** | **100.0%** |

*Noise floor: $\pm 0.008$ macro mAP (estimated from two seeds of the all-five baseline).*

---

## 3. Torque Interaction & Additivity Analysis

Adding generalized torque $	au$ produces an almost perfectly additive improvement across kinematic baselines.

### Table 2: Incremental Gain of Torque ($	au$)

| Kinematic Base | Without $\tau$ | With $\tau$ | Absolute $\Delta$ Macro mAP |
| :--- | :---: | :---: | :---: |
| $\sin\theta$ | 0.180 | 0.259 | **+0.079** |
| $\sin\theta, \cos\theta, \dot\theta, \ddot\theta$ | 0.216 | 0.297 | **+0.080** |

### Key Theoretical & Empirical Insights:
1. **Superadditive Combination:** Angle alone recovers 31% of the macro mAP gap and torque recovers 15%. Combined ($\{\sin\theta, \tau\}$), they recover **78%** of the gap (and **65%** of macro F1 gap). Kinematics and dynamics are complementary, not redundant.
2. **Kinematic Saturation:** Scoring all kinematic channels ($\{\sin\theta, \cos\theta, \dot\theta, \ddot\theta\}$) caps out at $0.216$ mAP. A minimalist dynamic arm ($\{\sin\theta, \tau\}$) beats it easily at $0.259$ with half the prediction head width.
3. **Acceleration as Incomplete Proxy:** $\{\sin\theta, \ddot\theta\}$ achieves $0.238$, recovering ~70% of torque's gain at matched $d_{\text{head}}=16$. The $0.021$ mAP deficit ($2.7\times$ noise floor) represents state-dependent mass and gravity matrices $M(\theta)$ and $G(\theta)$.
4. **Refuting Supervision Breadth Confounder:** Wide kinematic heads ($d_{\text{head}}=32$, $0.216$) underperform narrow dynamic heads ($d_{\text{head}}=16$, $0.259$). Physical content drives transfer, not head capacity.

---

## 4. Probe Readout Architecture Benchmark (Table 2 Data)

Benchmarking linear probe token aggregation methods over identical frozen backbones (all-channel objective, pretrained for 40 epochs).

### Table 3: Linear Probe Readout Comparison

| Readout Method | Description / Mechanics | Trainable Params | Macro mAP | Micro mAP | Macro F1 | Seed Spread (mAP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `MEAN` | Flat average over all $T \times D$ tokens | 15,420 | 0.289 | 0.501 | 0.131 | 0.024 |
| `MEAN-MAX` | Concatenates mean + coordinate-wise max | 30,780 | 0.260 | 0.458 | 0.057 | 0.011 |
| **`ATTENTIVE`** | **Softmax-weighted 2-layer scoring net** | **48,444** | **0.341** | **0.529** | **0.196** | **0.002** |
| `FACTORIZED` | Separate spatial & temporal scorers | 81,468 | 0.338 | **0.529** | **0.204** | **0.001** |

### Key Readout Takeaways:
1. **Unstructured Capacity Penalizes:** `MEAN-MAX` doubles parameters over `MEAN` but suffers severe drops (Macro F1 collapses from $0.131 \to 0.057$). Capacity along uninformative axes actively dilutes signal.
2. **`ATTENTIVE` Efficiency:** Gains $+0.052$ macro mAP (+18%) and $+0.065$ macro F1 (+49%) over standard mean pooling.
3. **`FACTORIZED` Redundancy:** Spatial/temporal factorization costs 68% more parameters than `ATTENTIVE` with negligible mAP change ($0.338$ vs $0.341$). Flat attention naturally learns coordinate structure.
4. **Readout Instability Warning:** Standard `MEAN` pooling exhibits $12\times$ higher run-to-run seed spread ($0.024$ vs $0.002$). Attentive pooling is critical for reliable ablation studies.

---

## 5. Summary Findings for Repository References

* **Main Finding:** Masked reconstruction of torque ($	au$) provides a clean, additive $+0.08$ mAP gain over pure kinematics by forcing the encoder to represent configuration-dependent dynamic loads ($M, G$).
* **Methodological Guidance:** Always use `ATTENTIVE` linear readouts for representation evaluation. Standard mean pooling introduces severe noise that masks subtle algorithmic gains.