| Domain | Dataset | Model | GT intrinsics | Domain info | RMSE↓ (m) | AbsRel↓ | δ1↑ | per-image RMSE / AbsRel / δ1 | px (excl.) | img |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| Indoor | ibims1 | DepthLM-12B<sup>1</sup> | yes | no | 1.261 [0.956, 1.566] | 0.141 [0.121, 0.163] | 0.810 [0.763, 0.853] | 0.787 / 0.141 / 0.810 | 9750 (250) | 100 |
| Indoor | ibims1 | DAv2-metric-L<sup>2</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.588 [0.460, 0.712] | 0.125 [0.110, 0.142] | 0.888 [0.844, 0.927] | 0.459 / 0.126 / 0.888 | 9750 (250) | 100 |
| Indoor | ibims1 | UniDepthV2-L | no | no | 0.498 [0.401, 0.589] | 0.090 [0.076, 0.107] | 0.943 [0.907, 0.971] | 0.368 / 0.090 / 0.942 | 9750 (250) | 100 |
| Indoor | ibims1 | Metric3Dv2-L | yes | no | 0.778 [0.637, 0.912] | 0.194 [0.165, 0.224] | 0.727 [0.651, 0.801] | 0.624 / 0.194 / 0.727 | 9750 (250) | 100 |
| Indoor | nyuv2 | DepthLM-12B | yes | no | 0.623 [0.593, 0.655] | 0.172 [0.167, 0.176] | 0.682 [0.660, 0.703] | 0.563 / 0.173 / 0.677 | 8483 (201) | 652 |
| Indoor | nyuv2 | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.462 [0.429, 0.497] | 0.120 [0.114, 0.127] | 0.891 [0.875, 0.906] | 0.374 / 0.121 / 0.889 | 8483 (201) | 652 |
| Indoor | nyuv2 | UniDepthV2-L | no | no | 0.444 [0.415, 0.474] | 0.123 [0.117, 0.130] | 0.902 [0.887, 0.916] | 0.364 / 0.124 / 0.899 | 8483 (201) | 652 |
| Indoor | nyuv2 | Metric3Dv2-L | yes | no | 0.304 [0.283, 0.326] | 0.068 [0.065, 0.071] | 0.970 [0.962, 0.975] | 0.240 / 0.068 / 0.969 | 8483 (201) | 652 |
| Indoor | mean | DepthLM-12B<sup>1</sup> | yes | no | 0.942 [0.791, 1.095] | 0.156 [0.146, 0.167] | 0.746 [0.720, 0.770] | 0.675 / 0.157 / 0.744 | 18233 (451) | 752 |
| Indoor | mean | DAv2-metric-L<sup>2</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 0.525 [0.458, 0.590] | 0.123 [0.114, 0.132] | 0.890 [0.867, 0.910] | 0.417 / 0.123 / 0.888 | 18233 (451) | 752 |
| Indoor | mean | UniDepthV2-L | no | no | 0.471 [0.418, 0.520] | 0.106 [0.099, 0.116] | 0.922 [0.905, 0.938] | 0.366 / 0.107 / 0.921 | 18233 (451) | 752 |
| Indoor | mean | Metric3Dv2-L | yes | no | 0.541 [0.471, 0.608] | 0.131 [0.116, 0.146] | 0.848 [0.811, 0.885] | 0.432 / 0.131 / 0.848 | 18233 (451) | 752 |
| Outdoor | ddad | DepthLM-12B<sup>1</sup> | yes | no | 13.693 [12.967, 14.407] | 0.253 [0.240, 0.266] | 0.680 [0.668, 0.693] | 10.616 / 0.253 / 0.680 | 9920 (80) | 1000 |
| Outdoor | ddad | DAv2-metric-L<sup>3</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 18.398 [18.018, 18.790] | 0.932 [0.905, 0.958] | 0.199 [0.183, 0.217] | 17.073 / 0.930 / 0.201 | 9920 (80) | 1000 |
| Outdoor | ddad | UniDepthV2-L | no | no | 9.629 [8.683, 10.656] | 0.133 [0.126, 0.141] | 0.866 [0.857, 0.875] | 6.670 / 0.133 / 0.867 | 9920 (80) | 1000 |
| Outdoor | ddad | Metric3Dv2-L<sup>4</sup> | yes | no | 8.862 [8.238, 9.470] | 0.118 [0.113, 0.122] | 0.868 [0.859, 0.877] | 6.396 / 0.118 / 0.868 | 9920 (80) | 1000 |
| Outdoor | nuscenes | DepthLM-12B<sup>5</sup><sup>1</sup> | yes | no | 6.540 [6.119, 6.990] | 0.283 [0.164, 0.501] | 0.823 [0.812, 0.836] | 4.573 / 0.281 / 0.824 | 9816 (184) | 1000 |
| Outdoor | nuscenes | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 9.472 [9.130, 9.786] | 0.821 [0.629, 1.078] | 0.170 [0.158, 0.183] | 8.285 / 0.817 / 0.171 | 9816 (184) | 1000 |
| Outdoor | nuscenes | UniDepthV2-L | no | no | 9.026 [7.465, 10.504] | 0.527 [0.194, 1.077] | 0.871 [0.860, 0.881] | 4.915 / 0.521 / 0.871 | 9816 (184) | 1000 |
| Outdoor | nuscenes | Metric3Dv2-L | yes | no | 13.678 [12.109, 15.221] | 0.514 [0.231, 0.882] | 0.845 [0.834, 0.856] | 6.941 / 0.509 / 0.845 | 9816 (184) | 1000 |
| Outdoor | diode_outdoor | DepthLM-12B | yes | no | 10.444 [9.836, 11.091] | 0.518 [0.482, 0.555] | 0.255 [0.232, 0.280] | 8.745 / 0.518 / 0.255 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | DAv2-metric-L | no | yes (indoor Hypersim / outdoor VKITTI model) | 8.186 [7.801, 8.597] | 0.742 [0.669, 0.817] | 0.336 [0.310, 0.361] | 7.307 / 0.741 / 0.336 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | UniDepthV2-L | no | no | 21.822 [19.728, 23.991] | 1.246 [1.041, 1.489] | 0.442 [0.406, 0.480] | 14.221 / 1.251 / 0.441 | 9756 (244) | 446 |
| Outdoor | diode_outdoor | Metric3Dv2-L | yes | no | 4.122 [3.743, 4.539] | 0.205 [0.176, 0.241] | 0.849 [0.835, 0.862] | 3.060 / 0.206 / 0.848 | 9756 (244) | 446 |
| Outdoor | mean | DepthLM-12B<sup>1</sup><sup>5</sup> | yes | no | 10.226 [9.873, 10.604] | 0.351 [0.309, 0.423] | 0.586 [0.576, 0.597] | 7.978 / 0.351 / 0.586 | 29492 (508) | 2446 |
| Outdoor | mean | DAv2-metric-L<sup>3</sup> | no | yes (indoor Hypersim / outdoor VKITTI model) | 12.019 [11.814, 12.247] | 0.832 [0.762, 0.920] | 0.235 [0.225, 0.246] | 10.888 / 0.829 / 0.236 | 29492 (508) | 2446 |
| Outdoor | mean | UniDepthV2-L | no | no | 13.492 [12.574, 14.348] | 0.635 [0.496, 0.825] | 0.726 [0.714, 0.739] | 8.602 / 0.635 / 0.726 | 29492 (508) | 2446 |
| Outdoor | mean | Metric3Dv2-L<sup>4</sup> | yes | no | 8.888 [8.295, 9.439] | 0.279 [0.183, 0.404] | 0.854 [0.847, 0.861] | 5.466 / 0.278 / 0.854 | 29492 (508) | 2446 |

DepthLM answers: official. Main metrics: z-depth, pooled over all common pixels of a dataset; brackets = 95 % CI from an image-level bootstrap (B = 1000, seed 0). Per-image = metric per image, then averaged. Domain rows ('mean') average their datasets; indoor and outdoor are never averaged together. px (excl.) = common pixels (pixels dropped because some model gave no prediction).

1. answer used as z as is (DepthLM official z labels for this dataset)
2. model max 20 m < cap 25 m
3. model max 80 m < cap 120 m
4. trained on DDAD
5. trained on nuScenes (other scenes)

### Appendix — RMSE in Euclidean distance

| Domain | Dataset | Model | RMSE z-depth (m) | RMSE Euclidean (m) | per-image z / Euclidean |
|:--|:--|:--|:--|:--|:--|
| Indoor | ibims1 | DepthLM-12B | 1.261 [0.956, 1.566] | 1.342 [1.017, 1.669] | 0.787 / 0.841 |
| Indoor | ibims1 | DAv2-metric-L | 0.588 [0.460, 0.712] | 0.631 [0.494, 0.762] | 0.459 / 0.493 |
| Indoor | ibims1 | UniDepthV2-L | 0.498 [0.401, 0.589] | 0.535 [0.429, 0.633] | 0.368 / 0.395 |
| Indoor | ibims1 | Metric3Dv2-L | 0.778 [0.637, 0.912] | 0.830 [0.680, 0.975] | 0.624 / 0.668 |
| Indoor | nyuv2 | DepthLM-12B | 0.623 [0.593, 0.655] | 0.667 [0.636, 0.702] | 0.563 / 0.602 |
| Indoor | nyuv2 | DAv2-metric-L | 0.462 [0.429, 0.497] | 0.495 [0.458, 0.534] | 0.374 / 0.400 |
| Indoor | nyuv2 | UniDepthV2-L | 0.444 [0.415, 0.474] | 0.474 [0.444, 0.506] | 0.364 / 0.390 |
| Indoor | nyuv2 | Metric3Dv2-L | 0.304 [0.283, 0.326] | 0.325 [0.302, 0.349] | 0.240 / 0.257 |
| Indoor | mean | DepthLM-12B | 0.942 [0.791, 1.095] | 1.005 [0.841, 1.164] | 0.675 / 0.721 |
| Indoor | mean | DAv2-metric-L | 0.525 [0.458, 0.590] | 0.563 [0.491, 0.632] | 0.417 / 0.447 |
| Indoor | mean | UniDepthV2-L | 0.471 [0.418, 0.520] | 0.505 [0.447, 0.555] | 0.366 / 0.392 |
| Indoor | mean | Metric3Dv2-L | 0.541 [0.471, 0.608] | 0.577 [0.503, 0.649] | 0.432 / 0.462 |
| Outdoor | ddad | DepthLM-12B | 13.693 [12.967, 14.407] | 14.860 [14.074, 15.640] | 10.616 / 11.537 |
| Outdoor | ddad | DAv2-metric-L | 18.398 [18.018, 18.790] | 20.153 [19.725, 20.571] | 17.073 / 18.665 |
| Outdoor | ddad | UniDepthV2-L | 9.629 [8.683, 10.656] | 10.406 [9.455, 11.446] | 6.670 / 7.273 |
| Outdoor | ddad | Metric3Dv2-L | 8.862 [8.238, 9.470] | 9.661 [9.015, 10.287] | 6.396 / 6.988 |
| Outdoor | nuscenes | DepthLM-12B | 6.540 [6.119, 6.990] | 7.019 [6.571, 7.497] | 4.573 / 4.912 |
| Outdoor | nuscenes | DAv2-metric-L | 9.472 [9.130, 9.786] | 10.306 [9.918, 10.643] | 8.285 / 8.958 |
| Outdoor | nuscenes | UniDepthV2-L | 9.026 [7.465, 10.504] | 9.808 [8.107, 11.462] | 4.915 / 5.300 |
| Outdoor | nuscenes | Metric3Dv2-L | 13.678 [12.109, 15.221] | 15.056 [13.318, 16.740] | 6.941 / 7.562 |
| Outdoor | diode_outdoor | DepthLM-12B | 10.444 [9.836, 11.091] | 11.261 [10.604, 11.972] | 8.745 / 9.427 |
| Outdoor | diode_outdoor | DAv2-metric-L | 8.186 [7.801, 8.597] | 8.812 [8.402, 9.268] | 7.307 / 7.859 |
| Outdoor | diode_outdoor | UniDepthV2-L | 21.822 [19.728, 23.991] | 23.559 [21.319, 25.859] | 14.221 / 15.341 |
| Outdoor | diode_outdoor | Metric3Dv2-L | 4.122 [3.743, 4.539] | 4.418 [4.009, 4.863] | 3.060 / 3.287 |
| Outdoor | mean | DepthLM-12B | 10.226 [9.873, 10.604] | 11.047 [10.681, 11.454] | 7.978 / 8.625 |
| Outdoor | mean | DAv2-metric-L | 12.019 [11.814, 12.247] | 13.090 [12.865, 13.345] | 10.888 / 11.828 |
| Outdoor | mean | UniDepthV2-L | 13.492 [12.574, 14.348] | 14.591 [13.607, 15.556] | 8.602 / 9.305 |
| Outdoor | mean | Metric3Dv2-L | 8.888 [8.295, 9.439] | 9.712 [9.055, 10.306] | 5.466 / 5.945 |
