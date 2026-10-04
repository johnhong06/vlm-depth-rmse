### ① δ1 reproduction

| Model | Dataset | pred | δ1 pooled | δ1 per-image | n | paper δ1 |
|:--|:--|:--|--:|--:|--:|:--|
| DAv2-metric-L | ibims1 | z | 0.887 | 0.887 | 10000 | DepthVLM T2 0.887 |
| DAv2-metric-L | nuscenes | z | 0.168 | 0.168 | 10000 | DepthVLM T2 0.168 |
| DepthPro | ibims1 | z | 0.829 | 0.829 | 10000 | DepthVLM T2 0.880 |
| DepthPro | nuscenes | z | 0.482 | 0.482 | 10000 | DepthVLM T2 0.389 |
| Metric3Dv2-L | ibims1 | z | 0.726 | 0.726 | 10000 | DepthVLM T2 0.726 |
| Metric3Dv2-L | nuscenes | z | 0.847 | 0.847 | 10000 | DepthVLM T2 0.747 |
| UniDepthV2-L | ibims1 | z | 0.941 | 0.941 | 10000 | DepthVLM T2 0.941 |
| UniDepthV2-L | nuscenes | z | 0.872 | 0.872 | 10000 | DepthVLM T2 0.872 |
| DepthLM-12B | ibims1 | raw (answer as is) | 0.810 | 0.810 | 9750 | DepthLM T1 0.870, DepthVLM T1 0.754 |
| DepthLM-12B | ibims1 | z-converted | 0.755 | 0.755 | 9750 | DepthLM T1 0.870, DepthVLM T1 0.754 |
| DepthLM-12B | nuscenes | raw (answer as is) | 0.823 | 0.824 | 9816 | DepthLM T1 0.819, DepthVLM T1 0.736 |
| DepthLM-12B | nuscenes | z-converted | 0.735 | 0.736 | 9816 | DepthLM T1 0.819, DepthVLM T1 0.736 |
| DAv2-metric-L | ddad | z | 0.201 | 0.201 | 10000 |  |
| DAv2-metric-L | diode_outdoor | z | 0.337 | 0.337 | 10000 |  |
| DAv2-metric-L | nyuv2 | z | 0.860 | 0.860 | 10000 |  |
| DepthPro | ddad | z | 0.136 | 0.136 | 10000 |  |
| DepthPro | diode_outdoor | z | 0.205 | 0.206 | 10000 |  |
| DepthPro | nyuv2 | z | 0.855 | 0.855 | 10000 |  |
| Metric3Dv2-L | ddad | z | 0.868 | 0.868 | 10000 |  |
| Metric3Dv2-L | diode_outdoor | z | 0.848 | 0.848 | 10000 |  |
| Metric3Dv2-L | nyuv2 | z | 0.938 | 0.938 | 10000 |  |
| UniDepthV2-L | ddad | z | 0.866 | 0.866 | 10000 |  |
| UniDepthV2-L | diode_outdoor | z | 0.442 | 0.442 | 10000 |  |
| UniDepthV2-L | nyuv2 | z | 0.871 | 0.871 | 10000 |  |
| DepthLM-12B | ddad | raw (answer as is) | 0.680 | 0.680 | 9920 | DepthLM T1 0.670, DepthVLM T1 0.654 |
| DepthLM-12B | ddad | z-converted | 0.651 | 0.651 | 9920 | DepthLM T1 0.670, DepthVLM T1 0.654 |
| DepthLM-12B | diode_outdoor | raw (answer as is) | 0.306 | 0.304 | 9756 |  |
| DepthLM-12B | diode_outdoor | z-converted | 0.255 | 0.255 | 9756 |  |
| DepthLM-12B | nyuv2 | raw (answer as is) | 0.816 | 0.816 | 9713 | DepthLM T1 0.799, DepthVLM T1 0.866 |
| DepthLM-12B | nyuv2 | z-converted | 0.673 | 0.674 | 9713 | DepthLM T1 0.799, DepthVLM T1 0.866 |

### ② sparse (common pixels) vs dense (all valid GT), same predictions — sparse / dense pooled / dense image-balanced

| Model | Dataset | RMSE | AbsRel | δ1 | pixels sparse / dense |
|:--|:--|:--|:--|:--|--:|
| DAv2-metric-L | ibims1 | 0.586 / 0.579 / 0.578 | 0.126 / 0.126 / 0.125 | 0.887 / 0.887 / 0.889 | 10000 / 29050557 |
| DAv2-metric-L | nuscenes | 9.431 / 9.728 / 9.438 | 0.817 / 0.737 / 0.687 | 0.168 / 0.165 / 0.167 | 10000 / 3489108 |
| DepthPro | ibims1 | 0.865 / 0.868 / 0.859 | 0.166 / 0.168 / 0.165 | 0.829 / 0.825 / 0.829 | 10000 / 29050557 |
| DepthPro | nuscenes | 9.624 / 28.423 / 33.837 | 0.484 / 0.403 / 0.382 | 0.482 / 0.480 / 0.480 | 10000 / 3489108 |
| Metric3Dv2-L | ibims1 | 0.776 / 0.774 / 0.769 | 0.195 / 0.195 / 0.193 | 0.726 / 0.723 / 0.729 | 10000 / 29050557 |
| Metric3Dv2-L | nuscenes | 13.599 / 12.969 / 13.031 | 0.507 / 0.328 / 0.300 | 0.847 / 0.843 / 0.845 | 10000 / 3489108 |
| UniDepthV2-L | ibims1 | 0.497 / 0.487 / 0.487 | 0.090 / 0.090 / 0.090 | 0.941 / 0.941 / 0.942 | 10000 / 29050557 |
| UniDepthV2-L | nuscenes | 8.971 / 8.344 / 8.747 | 0.520 / 0.312 / 0.291 | 0.872 / 0.868 / 0.870 | 10000 / 3489108 |
| DAv2-metric-L | ddad | 18.356 / 18.608 / 18.312 | 0.929 / 1.015 / 0.942 | 0.201 / 0.141 / 0.194 | 10000 / 44048702 |
| DAv2-metric-L | diode_outdoor | 8.171 / 7.684 / 8.253 | 0.738 / 0.718 / 0.740 | 0.337 / 0.334 / 0.339 | 10000 / 274076454 |
| DAv2-metric-L | nyuv2 | 0.910 / 0.903 / 0.903 | 0.185 / 0.180 / 0.180 | 0.860 / 0.855 / 0.855 | 10000 / 156663738 |
| DepthPro | ddad | 19.450 / 32.898 / 41.172 | 0.481 / 0.503 / 0.492 | 0.136 / 0.119 / 0.141 | 10000 / 44048702 |
| DepthPro | diode_outdoor | 13.607 / 88.806 / 101.880 | 0.579 / 0.667 / 0.759 | 0.205 / 0.213 / 0.208 | 10000 / 274076454 |
| DepthPro | nyuv2 | 0.956 / 0.943 / 0.943 | 0.188 / 0.180 / 0.180 | 0.855 / 0.856 / 0.856 | 10000 / 156663738 |
| Metric3Dv2-L | ddad | 8.876 / 8.705 / 8.892 | 0.118 / 0.122 / 0.119 | 0.868 / 0.865 / 0.870 | 10000 / 44048702 |
| Metric3Dv2-L | diode_outdoor | 4.162 / 3.840 / 4.333 | 0.204 / 0.193 / 0.211 | 0.848 / 0.859 / 0.849 | 10000 / 274076454 |
| Metric3Dv2-L | nyuv2 | 0.811 / 0.877 / 0.877 | 0.129 / 0.123 / 0.123 | 0.938 / 0.937 / 0.937 | 10000 / 156663738 |
| UniDepthV2-L | ddad | 9.600 / 9.788 / 10.436 | 0.133 / 0.137 / 0.139 | 0.866 / 0.873 / 0.867 | 10000 / 44048702 |
| UniDepthV2-L | diode_outdoor | 22.036 / 19.897 / 22.258 | 1.249 / 1.042 / 1.233 | 0.442 / 0.476 / 0.441 | 10000 / 274076454 |
| UniDepthV2-L | nyuv2 | 0.977 / 0.970 / 0.970 | 0.193 / 0.186 / 0.186 | 0.871 / 0.868 / 0.868 | 10000 / 156663738 |

### ③ z conversion (DepthLM-12B), medians per ray-factor bin

| Dataset | ray factor | n | z / d | raw / GT | z-converted / GT |
|:--|:--|--:|--:|--:|--:|
| ddad | (0.999, 1.02] | 2756 | 0.993 | 0.973 | 0.965 |
| ddad | (1.02, 1.05] | 1853 | 0.967 | 0.993 | 0.961 |
| ddad | (1.05, 1.1] | 1922 | 0.934 | 1.006 | 0.939 |
| ddad | (1.1, 1.2] | 1579 | 0.874 | 1.004 | 0.879 |
| ddad | (1.2, 1.4] | 1805 | 0.788 | 1.056 | 0.827 |
| ddad | (1.4, 3.0] | 5 | 0.707 | 1.313 | 0.937 |
| diode_outdoor | (0.999, 1.02] | 1401 | 0.990 | 0.714 | 0.706 |
| diode_outdoor | (1.02, 1.05] | 2089 | 0.965 | 0.737 | 0.711 |
| diode_outdoor | (1.05, 1.1] | 3114 | 0.932 | 0.750 | 0.699 |
| diode_outdoor | (1.1, 1.2] | 3085 | 0.882 | 0.796 | 0.699 |
| diode_outdoor | (1.2, 1.4] | 67 | 0.828 | 0.815 | 0.676 |
| ibims1 | (0.999, 1.02] | 1254 | 0.990 | 0.938 | 0.930 |
| ibims1 | (1.02, 1.05] | 2049 | 0.966 | 0.952 | 0.919 |
| ibims1 | (1.05, 1.1] | 3331 | 0.932 | 0.968 | 0.903 |
| ibims1 | (1.1, 1.2] | 2877 | 0.883 | 0.986 | 0.865 |
| ibims1 | (1.2, 1.4] | 239 | 0.823 | 0.984 | 0.806 |
| nuscenes | (0.999, 1.02] | 1790 | 0.990 | 0.983 | 0.973 |
| nuscenes | (1.02, 1.05] | 2297 | 0.967 | 0.991 | 0.957 |
| nuscenes | (1.05, 1.1] | 2447 | 0.934 | 0.994 | 0.926 |
| nuscenes | (1.1, 1.2] | 2641 | 0.877 | 0.990 | 0.865 |
| nuscenes | (1.2, 1.4] | 550 | 0.789 | 0.918 | 0.713 |
| nuscenes | (1.4, 3.0] | 91 | 0.698 | 0.920 | 0.646 |
| nyuv2 | (0.999, 1.02] | 1447 | 0.990 | 0.845 | 0.837 |
| nyuv2 | (1.02, 1.05] | 2186 | 0.966 | 0.874 | 0.845 |
| nyuv2 | (1.05, 1.1] | 3429 | 0.932 | 0.905 | 0.844 |
| nyuv2 | (1.1, 1.2] | 2646 | 0.888 | 0.948 | 0.839 |
| nyuv2 | (1.2, 1.4] | 5 | 0.831 | 1.088 | 0.904 |
