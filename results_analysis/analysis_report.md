# Analysis of Aggregated Results

## Overview
Total records analyzed: 211

## Base Model Breakdown
Average p-value by Base Model:

| Base Model | Mean P_500 |
| --- | --- |
| Pythia-12B | 1.2437e-01 |

## Dataset Breakdown
Average p-value by Dataset (sorted from lowest to highest, lowest indicates highest privacy leakage/inference signal):

| Dataset | Mean P_500 |
| --- | --- |
| ubuntu | 1.3024e-75 |
| philpapers | 2.5043e-57 |
| youtubesubtitles | 6.1062e-36 |
| stackexchange | 1.5146e-33 |
| bookcorpus2 | 2.1473e-18 |
| wikipedia | 5.2401e-11 |
| europarl | 5.7874e-09 |
| books3 | 3.5422e-08 |
| hackernews | 5.6531e-06 |
| gutenberg | 1.3564e-05 |
| uspto | 1.5238e-05 |
| math | 2.4937e-02 |
| openwebtext2 | 4.7353e-02 |
| github | 6.1053e-02 |
| freelaw | 6.2635e-02 |
| cc | 1.1082e-01 |
| arxiv | 3.3492e-01 |
| opensubtitles | 4.4566e-01 |

## Quantization Config Breakdown
Average p-value by Quantization Configuration:

| Config | Mean P_500 |
| --- | --- |
| awq-b4-gs32-zp1 | 3.3946e-02 |
| EleutherAI_pythia-12b-deduped | 3.5661e-02 |
| gptq-b4-gs128-da0 | 4.2166e-02 |
| static-4bit-nf4-dq-float32 | 4.4950e-02 |
| awq-b4-gs128-zp1 | 4.7693e-02 |
| gptq-b4-gs32-da0 | 4.7936e-02 |
| static-4bit-fp4-dq-float16 | 4.9247e-02 |
| gptq-b8-gs128-da1 | 4.9308e-02 |
| static-4bit-nf4-dq-float16 | 5.3381e-02 |
| gptq-b8-gs-1-da0 | 5.5256e-02 |
| awq-b4-gs64-zp1 | 5.6709e-02 |
| static-4bit-nf4-bfloat16 | 6.1857e-02 |
| gptq-b8-gs-1-da1 | 6.5710e-02 |
| static-4bit-fp4-float16 | 7.2136e-02 |
| gptq-b4-gs64-da0 | 7.8660e-02 |
| gptq-b8-gs32-da1 | 8.2079e-02 |
| gptq-b4-gs-1-da0 | 8.2322e-02 |
| static-4bit-nf4-dq-bfloat16 | 8.4613e-02 |
| gptq-b4-gs64-da1 | 8.5025e-02 |
| static-4bit-fp4-bfloat16 | 8.5520e-02 |
| static-4bit-nf4-float16 | 8.6471e-02 |
| gptq-b8-gs64-da0 | 8.7038e-02 |
| gptq-b4-gs32-da1 | 1.0230e-01 |
| static-4bit-fp4-dq-float32 | 1.0764e-01 |
| gptq-b8-gs128-da0 | 1.0981e-01 |
| gptq-b4-gs128-da1 | 1.1147e-01 |
| static-4bit-nf4-float32 | 1.1612e-01 |
| gptq-b8-gs64-da1 | 1.1885e-01 |
| gptq-b8-gs32-da0 | 1.3119e-01 |
| static-4bit-fp4-float32 | 1.3419e-01 |
| gptq-b4-gs-1-da1 | 1.4737e-01 |
| gptq-b2-gs64-da1 | 2.1432e-01 |
| gptq-b3-gs-1-da0 | 2.2697e-01 |
| gptq-b2-gs32-da0 | 2.6849e-01 |
| gptq-b3-gs128-da1 | 2.7354e-01 |
| gptq-b3-gs128-da0 | 2.7482e-01 |
| gptq-b3-gs64-da0 | 2.7680e-01 |
| gptq-b2-gs64-da0 | 2.7871e-01 |
| gptq-b2-gs32-da1 | 2.8082e-01 |
| gptq-b3-gs64-da1 | 3.1247e-01 |
| gptq-b3-gs32-da0 | 3.1440e-01 |
| gptq-b2-gs128-da1 | 3.1729e-01 |
| gptq-b3-gs-1-da1 | 3.4352e-01 |
| gptq-b3-gs32-da1 | 3.6068e-01 |
| gptq-b2-gs128-da0 | 4.2601e-01 |
| gptq-b2-gs-1-da0 | 4.5650e-01 |
| gptq-b2-gs-1-da1 | 4.5790e-01 |

## Top 15 Highest Privacy Leakage Configurations (Lowest P-Values)

| Base Model | Dataset | Config | Mean P_500 |
| --- | --- | --- | --- |
| Pythia-12B | ubuntu | awq-b4-gs128-zp1 | 2.7551e-82 |
| Pythia-12B | ubuntu | awq-b4-gs32-zp1 | 2.9593e-82 |
| Pythia-12B | ubuntu | EleutherAI_pythia-12b-deduped | 8.1013e-78 |
| Pythia-12B | ubuntu | awq-b4-gs64-zp1 | 5.2013e-75 |
| Pythia-12B | philpapers | awq-b4-gs32-zp1 | 2.8553e-62 |
| Pythia-12B | philpapers | EleutherAI_pythia-12b-deduped | 2.9989e-59 |
| Pythia-12B | philpapers | awq-b4-gs64-zp1 | 7.4829e-57 |
| Pythia-12B | youtubesubtitles | awq-b4-gs64-zp1 | 9.1441e-41 |
| Pythia-12B | youtubesubtitles | awq-b4-gs128-zp1 | 1.2438e-37 |
| Pythia-12B | stackexchange | awq-b4-gs32-zp1 | 2.5480e-37 |
| Pythia-12B | youtubesubtitles | awq-b4-gs32-zp1 | 1.0271e-36 |
| Pythia-12B | youtubesubtitles | EleutherAI_pythia-12b-deduped | 2.3273e-35 |
| Pythia-12B | stackexchange | awq-b4-gs128-zp1 | 3.9272e-35 |
| Pythia-12B | stackexchange | awq-b4-gs64-zp1 | 2.1802e-33 |
| Pythia-12B | stackexchange | EleutherAI_pythia-12b-deduped | 3.8386e-33 |

