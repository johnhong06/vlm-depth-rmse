| Domain | Dataset | Model | GT intrinsics | Domain info | RMSE↓ (m) | AbsRel↓ | δ1↑ | per-image RMSE / AbsRel / δ1 | px (excl.) | img |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| Indoor | ibims1 | DepthLM-12B | yes | no | 1.352 [1.039, 1.666] | 0.167 [0.148, 0.188] | 0.755 [0.703, 0.799] | 0.886 / 0.167 / 0.755 | 9750 (250) | 100 |
| Indoor | ibims1 | DAv2-metric-L<sup>1</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.588 [0.460, 0.712] | 0.125 [0.110, 0.142] | 0.888 [0.844, 0.927] | 0.459 / 0.126 / 0.888 | 9750 (250) | 100 |
| Indoor | ibims1 | UniDepthV2-L | no | no | 0.498 [0.401, 0.589] | 0.090 [0.076, 0.107] | 0.943 [0.907, 0.971] | 0.368 / 0.090 / 0.942 | 9750 (250) | 100 |
| Indoor | ibims1 | Metric3Dv2-L | yes | no | 0.778 [0.637, 0.912] | 0.194 [0.165, 0.224] | 0.727 [0.651, 0.801] | 0.624 / 0.194 / 0.727 | 9750 (250) | 100 |
| Indoor | nyuv2 | DepthLM-12B | yes | no | 0.623 [0.593, 0.655] | 0.172 [0.167, 0.176] | 0.682 [0.660, 0.703] | 0.563 / 0.173 / 0.677 | 8483 (201) | 652 |
| Indoor | nyuv2 | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.462 [0.429, 0.497] | 0.120 [0.114, 0.127] | 0.891 [0.875, 0.906] | 0.374 / 0.121 / 0.889 | 8483 (201) | 652 |
| Indoor | nyuv2 | UniDepthV2-L | no | no | 0.444 [0.415, 0.474] | 0.123 [0.117, 0.130] | 0.902 [0.887, 0.916] | 0.364 / 0.124 / 0.899 | 8483 (201) | 652 |
| Indoor | nyuv2 | Metric3Dv2-L | yes | no | 0.304 [0.283, 0.326] | 0.068 [0.065, 0.071] | 0.970 [0.962, 0.975] | 0.240 / 0.068 / 0.969 | 8483 (201) | 652 |
| Indoor | mean | DepthLM-12B | yes | no | 0.988 [0.832, 1.145] | 0.169 [0.159, 0.180] | 0.718 [0.688, 0.744] | 0.725 / 0.170 / 0.716 | 18233 (451) | 752 |
| Indoor | mean | DAv2-metric-L<sup>1</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.525 [0.458, 0.590] | 0.123 [0.114, 0.132] | 0.890 [0.867, 0.910] | 0.417 / 0.123 / 0.888 | 18233 (451) | 752 |
| Indoor | mean | UniDepthV2-L | no | no | 0.471 [0.418, 0.520] | 0.106 [0.099, 0.116] | 0.922 [0.905, 0.938] | 0.366 / 0.107 / 0.921 | 18233 (451) | 752 |
| Indoor | mean | Metric3Dv2-L | yes | no | 0.541 [0.471, 0.608] | 0.131 [0.116, 0.146] | 0.848 [0.811, 0.885] | 0.432 / 0.131 / 0.848 | 18233 (451) | 752 |
| Outdoor | diode_outdoor | DepthLM-12B | yes | no | 10.444 [9.863, 11.036] | 0.518 [0.485, 0.554] | 0.255 [0.231, 0.279] | 8.745 / 0.518 / 0.255 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 8.186 [7.837, 8.602] | 0.742 [0.677, 0.818] | 0.336 [0.313, 0.361] | 7.307 / 0.741 / 0.336 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | UniDepthV2-L | no | no | 21.822 [19.631, 24.021] | 1.246 [1.039, 1.478] | 0.442 [0.408, 0.478] | 14.221 / 1.251 / 0.441 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | Metric3Dv2-L | yes | no | 4.122 [3.739, 4.485] | 0.205 [0.177, 0.236] | 0.849 [0.835, 0.862] | 3.060 / 0.206 / 0.848 | 9756 (244) | 446 |
| Outdoor | mean | DepthLM-12B | yes | no | 10.444 [9.863, 11.036] | 0.518 [0.485, 0.554] | 0.255 [0.231, 0.279] | 8.745 / 0.518 / 0.255 | 9756 (244) | 446 |
| Outdoor | mean | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 8.186 [7.837, 8.602] | 0.742 [0.677, 0.818] | 0.336 [0.313, 0.361] | 7.307 / 0.741 / 0.336 | 9756 (244) | 446 |
| Outdoor | mean | UniDepthV2-L | no | no | 21.822 [19.631, 24.021] | 1.246 [1.039, 1.478] | 0.442 [0.408, 0.478] | 14.221 / 1.251 / 0.441 | 9756 (244) | 446 |
| Outdoor | mean | Metric3Dv2-L | yes | no | 4.122 [3.739, 4.485] | 0.205 [0.177, 0.236] | 0.849 [0.835, 0.862] | 3.060 / 0.206 / 0.848 | 9756 (244) | 446 |

DepthLM answers: converted. Main metrics: z-depth, pooled over all common pixels of a dataset; brackets = 95 % CI from an image-level bootstrap (B = 1000, seed 0). Per-image = metric per image, then averaged. Domain rows ('mean') average their datasets; indoor and outdoor are never averaged together. px (excl.) = common pixels (pixels dropped because some model gave no prediction).

1. model max 20 m < cap 25 m

### Appendix — RMSE in Euclidean distance

| Domain | Dataset | Model | RMSE z-depth (m) | RMSE Euclidean (m) | per-image z / Euclidean |
|:--|:--|:--|:--|:--|:--|
| Indoor | ibims1 | DepthLM-12B | 1.352 [1.039, 1.666] | 1.447 [1.118, 1.776] | 0.886 / 0.955 |
| Indoor | ibims1 | DAv2-metric-L | 0.588 [0.460, 0.712] | 0.631 [0.494, 0.762] | 0.459 / 0.493 |
| Indoor | ibims1 | UniDepthV2-L | 0.498 [0.401, 0.589] | 0.535 [0.429, 0.633] | 0.368 / 0.395 |
| Indoor | ibims1 | Metric3Dv2-L | 0.778 [0.637, 0.912] | 0.830 [0.680, 0.975] | 0.624 / 0.668 |
| Indoor | nyuv2 | DepthLM-12B | 0.623 [0.593, 0.655] | 0.667 [0.636, 0.702] | 0.563 / 0.602 |
| Indoor | nyuv2 | DAv2-metric-L | 0.462 [0.429, 0.497] | 0.495 [0.458, 0.534] | 0.374 / 0.400 |
| Indoor | nyuv2 | UniDepthV2-L | 0.444 [0.415, 0.474] | 0.474 [0.444, 0.506] | 0.364 / 0.390 |
| Indoor | nyuv2 | Metric3Dv2-L | 0.304 [0.283, 0.326] | 0.325 [0.302, 0.349] | 0.240 / 0.257 |
| Indoor | mean | DepthLM-12B | 0.988 [0.832, 1.145] | 1.057 [0.891, 1.220] | 0.725 / 0.779 |
| Indoor | mean | DAv2-metric-L | 0.525 [0.458, 0.590] | 0.563 [0.491, 0.632] | 0.417 / 0.447 |
| Indoor | mean | UniDepthV2-L | 0.471 [0.418, 0.520] | 0.505 [0.447, 0.555] | 0.366 / 0.392 |
| Indoor | mean | Metric3Dv2-L | 0.541 [0.471, 0.608] | 0.577 [0.503, 0.649] | 0.432 / 0.462 |
| Outdoor | diode_outdoor | DepthLM-12B | 10.444 [9.863, 11.036] | 11.261 [10.637, 11.923] | 8.745 / 9.427 |
| Outdoor | diode_outdoor | DAv2-metric-L | 8.186 [7.837, 8.602] | 8.812 [8.432, 9.258] | 7.307 / 7.859 |
| Outdoor | diode_outdoor | UniDepthV2-L | 21.822 [19.631, 24.021] | 23.559 [21.212, 25.919] | 14.221 / 15.341 |
| Outdoor | diode_outdoor | Metric3Dv2-L | 4.122 [3.739, 4.485] | 4.418 [4.008, 4.798] | 3.060 / 3.287 |
| Outdoor | mean | DepthLM-12B | 10.444 [9.863, 11.036] | 11.261 [10.637, 11.923] | 8.745 / 9.427 |
| Outdoor | mean | DAv2-metric-L | 8.186 [7.837, 8.602] | 8.812 [8.432, 9.258] | 7.307 / 7.859 |
| Outdoor | mean | UniDepthV2-L | 21.822 [19.631, 24.021] | 23.559 [21.212, 25.919] | 14.221 / 15.341 |
| Outdoor | mean | Metric3Dv2-L | 4.122 [3.739, 4.485] | 4.418 [4.008, 4.798] | 3.060 / 3.287 |
