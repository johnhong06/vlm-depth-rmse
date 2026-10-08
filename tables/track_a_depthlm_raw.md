| Domain | Dataset | Model | GT intrinsics | Domain info | RMSE↓ (m) | AbsRel↓ | δ1↑ | per-image RMSE / AbsRel / δ1 | px (excl.) | img |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| Indoor | ibims1 | DepthLM-12B | yes | no | 1.261 [0.956, 1.566] | 0.141 [0.121, 0.163] | 0.810 [0.763, 0.853] | 0.787 / 0.141 / 0.810 | 9750 (250) | 100 |
| Indoor | ibims1 | DAv2-metric-L<sup>1</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.588 [0.460, 0.712] | 0.125 [0.110, 0.142] | 0.888 [0.844, 0.927] | 0.459 / 0.126 / 0.888 | 9750 (250) | 100 |
| Indoor | ibims1 | UniDepthV2-L | no | no | 0.498 [0.401, 0.589] | 0.090 [0.076, 0.107] | 0.943 [0.907, 0.971] | 0.368 / 0.090 / 0.942 | 9750 (250) | 100 |
| Indoor | ibims1 | Metric3Dv2-L | yes | no | 0.778 [0.637, 0.912] | 0.194 [0.165, 0.224] | 0.727 [0.651, 0.801] | 0.624 / 0.194 / 0.727 | 9750 (250) | 100 |
| Indoor | nyuv2 | DepthLM-12B | yes | no | 0.528 [0.498, 0.559] | 0.133 [0.130, 0.138] | 0.839 [0.824, 0.851] | 0.465 / 0.135 / 0.836 | 8483 (201) | 652 |
| Indoor | nyuv2 | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.462 [0.429, 0.497] | 0.120 [0.114, 0.127] | 0.891 [0.875, 0.906] | 0.374 / 0.121 / 0.889 | 8483 (201) | 652 |
| Indoor | nyuv2 | UniDepthV2-L | no | no | 0.444 [0.415, 0.474] | 0.123 [0.117, 0.130] | 0.902 [0.887, 0.916] | 0.364 / 0.124 / 0.899 | 8483 (201) | 652 |
| Indoor | nyuv2 | Metric3Dv2-L | yes | no | 0.304 [0.283, 0.326] | 0.068 [0.065, 0.071] | 0.970 [0.962, 0.975] | 0.240 / 0.068 / 0.969 | 8483 (201) | 652 |
| Indoor | mean | DepthLM-12B | yes | no | 0.894 [0.741, 1.047] | 0.137 [0.127, 0.148] | 0.824 [0.799, 0.846] | 0.626 / 0.138 / 0.823 | 18233 (451) | 752 |
| Indoor | mean | DAv2-metric-L<sup>1</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.525 [0.458, 0.590] | 0.123 [0.114, 0.132] | 0.890 [0.867, 0.910] | 0.417 / 0.123 / 0.888 | 18233 (451) | 752 |
| Indoor | mean | UniDepthV2-L | no | no | 0.471 [0.418, 0.520] | 0.106 [0.099, 0.116] | 0.922 [0.905, 0.938] | 0.366 / 0.107 / 0.921 | 18233 (451) | 752 |
| Indoor | mean | Metric3Dv2-L | yes | no | 0.541 [0.471, 0.608] | 0.131 [0.116, 0.146] | 0.848 [0.811, 0.885] | 0.432 / 0.131 / 0.848 | 18233 (451) | 752 |
| Outdoor | diode_outdoor | DepthLM-12B | yes | no | 10.138 [9.549, 10.721] | 0.516 [0.479, 0.556] | 0.306 [0.279, 0.330] | 8.420 / 0.516 / 0.304 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 8.186 [7.837, 8.602] | 0.742 [0.677, 0.818] | 0.336 [0.313, 0.361] | 7.307 / 0.741 / 0.336 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | UniDepthV2-L | no | no | 21.822 [19.631, 24.021] | 1.246 [1.039, 1.478] | 0.442 [0.408, 0.478] | 14.221 / 1.251 / 0.441 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | Metric3Dv2-L | yes | no | 4.122 [3.739, 4.485] | 0.205 [0.177, 0.236] | 0.849 [0.835, 0.862] | 3.060 / 0.206 / 0.848 | 9756 (244) | 446 |
| Outdoor | mean | DepthLM-12B | yes | no | 10.138 [9.549, 10.721] | 0.516 [0.479, 0.556] | 0.306 [0.279, 0.330] | 8.420 / 0.516 / 0.304 | 9756 (244) | 446 |
| Outdoor | mean | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 8.186 [7.837, 8.602] | 0.742 [0.677, 0.818] | 0.336 [0.313, 0.361] | 7.307 / 0.741 / 0.336 | 9756 (244) | 446 |
| Outdoor | mean | UniDepthV2-L | no | no | 21.822 [19.631, 24.021] | 1.246 [1.039, 1.478] | 0.442 [0.408, 0.478] | 14.221 / 1.251 / 0.441 | 9756 (244) | 446 |
| Outdoor | mean | Metric3Dv2-L | yes | no | 4.122 [3.739, 4.485] | 0.205 [0.177, 0.236] | 0.849 [0.835, 0.862] | 3.060 / 0.206 / 0.848 | 9756 (244) | 446 |

DepthLM answers: raw. Main metrics: z-depth, pooled over all common pixels of a dataset; brackets = 95 % CI from an image-level bootstrap (B = 1000, seed 0). Per-image = metric per image, then averaged. Domain rows ('mean') average their datasets; indoor and outdoor are never averaged together. px (excl.) = common pixels (pixels dropped because some model gave no prediction).

1. model max 20 m < cap 25 m

### Appendix — RMSE in Euclidean distance

| Domain | Dataset | Model | RMSE z-depth (m) | RMSE Euclidean (m) | per-image z / Euclidean |
|:--|:--|:--|:--|:--|:--|
| Indoor | ibims1 | DepthLM-12B | 1.261 [0.956, 1.566] | 1.342 [1.017, 1.669] | 0.787 / 0.841 |
| Indoor | ibims1 | DAv2-metric-L | 0.588 [0.460, 0.712] | 0.631 [0.494, 0.762] | 0.459 / 0.493 |
| Indoor | ibims1 | UniDepthV2-L | 0.498 [0.401, 0.589] | 0.535 [0.429, 0.633] | 0.368 / 0.395 |
| Indoor | ibims1 | Metric3Dv2-L | 0.778 [0.637, 0.912] | 0.830 [0.680, 0.975] | 0.624 / 0.668 |
| Indoor | nyuv2 | DepthLM-12B | 0.528 [0.498, 0.559] | 0.561 [0.530, 0.595] | 0.465 / 0.493 |
| Indoor | nyuv2 | DAv2-metric-L | 0.462 [0.429, 0.497] | 0.495 [0.458, 0.534] | 0.374 / 0.400 |
| Indoor | nyuv2 | UniDepthV2-L | 0.444 [0.415, 0.474] | 0.474 [0.444, 0.506] | 0.364 / 0.390 |
| Indoor | nyuv2 | Metric3Dv2-L | 0.304 [0.283, 0.326] | 0.325 [0.302, 0.349] | 0.240 / 0.257 |
| Indoor | mean | DepthLM-12B | 0.894 [0.741, 1.047] | 0.951 [0.787, 1.113] | 0.626 / 0.667 |
| Indoor | mean | DAv2-metric-L | 0.525 [0.458, 0.590] | 0.563 [0.491, 0.632] | 0.417 / 0.447 |
| Indoor | mean | UniDepthV2-L | 0.471 [0.418, 0.520] | 0.505 [0.447, 0.555] | 0.366 / 0.392 |
| Indoor | mean | Metric3Dv2-L | 0.541 [0.471, 0.608] | 0.577 [0.503, 0.649] | 0.432 / 0.462 |
| Outdoor | diode_outdoor | DepthLM-12B | 10.138 [9.549, 10.721] | 10.915 [10.279, 11.556] | 8.420 / 9.061 |
| Outdoor | diode_outdoor | DAv2-metric-L | 8.186 [7.837, 8.602] | 8.812 [8.432, 9.258] | 7.307 / 7.859 |
| Outdoor | diode_outdoor | UniDepthV2-L | 21.822 [19.631, 24.021] | 23.559 [21.212, 25.919] | 14.221 / 15.341 |
| Outdoor | diode_outdoor | Metric3Dv2-L | 4.122 [3.739, 4.485] | 4.418 [4.008, 4.798] | 3.060 / 3.287 |
| Outdoor | mean | DepthLM-12B | 10.138 [9.549, 10.721] | 10.915 [10.279, 11.556] | 8.420 / 9.061 |
| Outdoor | mean | DAv2-metric-L | 8.186 [7.837, 8.602] | 8.812 [8.432, 9.258] | 7.307 / 7.859 |
| Outdoor | mean | UniDepthV2-L | 21.822 [19.631, 24.021] | 23.559 [21.212, 25.919] | 14.221 / 15.341 |
| Outdoor | mean | Metric3Dv2-L | 4.122 [3.739, 4.485] | 4.418 [4.008, 4.798] | 3.060 / 3.287 |
