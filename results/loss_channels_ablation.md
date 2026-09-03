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

Note: Tested on mae_40ep with a attentive probe for 20ep()