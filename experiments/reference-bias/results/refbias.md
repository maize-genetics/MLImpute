# Reference-bias eval: refmap vs chain (PAV), per-index-sample read attribution

120 (individual, arm) results, 0.1x coverage rung. See /home/zrm22/.claude/plans/dreamy-booping-sutton.md for full methodology and caveats (esp. the differing carrier/occurrence caps between the refmap and chain binaries, and that B73's count is partly definitional in both arms).

## De-confounded B73 bias estimate (mash-relatedness-corrected)

For each read set's true source assembly, fit hit_ratio ~ mash-similarity across the 24 non-B73 founders, then evaluate the residual (actual - predicted) at B73's own similarity. Near-zero means B73's elevation is fully explained by real sequence relatedness; a positive residual is what remains after relatedness is accounted for -- the bias estimate, in hit_ratio units. See refbias_mash.py.

| arm | n | mean_residual | min | max |
|---|---:|---:|---:|---:|
| chain_ordinary_256 | 40 | 0.2258 | 0.1871 | 0.2835 |
| refmap | 40 | 0.0046 | -0.0222 | 0.2242 |

(IDX-INBRED/B73 rows compare B73 against itself -- mash similarity saturates at 1.0, outside the fitted range of the other 24 points, so that one row's residual is a regression-extrapolation artifact, not bias; it is included above for completeness but excluded below.)

Excluding B73's own read set:

| arm | n | mean_residual | min | max |
|---|---:|---:|---:|---:|
| chain_ordinary_256 | 39 | 0.2258 | 0.1871 | 0.2835 |
| refmap | 39 | -0.0010 | -0.0222 | 0.1468 |

## Placement rate (fraction of input reads producing output)

| dataset | individual | arm | placement_rate |
|---|---|---|---:|
| IDX-HYB | B73xCML103 | chain_ordinary_256 | 0.6404 |
| IDX-HYB | B73xCML103 | chain_ordinary_5000 | 0.6592 |
| IDX-HYB | B73xCML103 | refmap | 0.8047 |
| IDX-HYB | B73xOh43 | chain_ordinary_256 | 0.6502 |
| IDX-HYB | B73xOh43 | chain_ordinary_5000 | 0.6674 |
| IDX-HYB | B73xOh43 | refmap | 0.8049 |
| IDX-HYB | B97xCML103 | chain_ordinary_256 | 0.4954 |
| IDX-HYB | B97xCML103 | chain_ordinary_5000 | 0.5305 |
| IDX-HYB | B97xCML103 | refmap | 0.7942 |
| IDX-HYB | Il14HxB97 | chain_ordinary_256 | 0.4952 |
| IDX-HYB | Il14HxB97 | chain_ordinary_5000 | 0.5290 |
| IDX-HYB | Il14HxB97 | refmap | 0.8008 |
| IDX-HYB | Oh43xIl14H | chain_ordinary_256 | 0.4957 |
| IDX-HYB | Oh43xIl14H | chain_ordinary_5000 | 0.5296 |
| IDX-HYB | Oh43xIl14H | refmap | 0.8030 |
| IDX-INBRED | B73 | chain_ordinary_256 | 0.7964 |
| IDX-INBRED | B73 | chain_ordinary_5000 | 0.7966 |
| IDX-INBRED | B73 | refmap | 0.8114 |
| IDX-INBRED | B97 | chain_ordinary_256 | 0.5054 |
| IDX-INBRED | B97 | chain_ordinary_5000 | 0.5385 |
| IDX-INBRED | B97 | refmap | 0.7915 |
| IDX-INBRED | CML103 | chain_ordinary_256 | 0.4871 |
| IDX-INBRED | CML103 | chain_ordinary_5000 | 0.5244 |
| IDX-INBRED | CML103 | refmap | 0.7976 |
| IDX-INBRED | Il14H | chain_ordinary_256 | 0.4871 |
| IDX-INBRED | Il14H | chain_ordinary_5000 | 0.5213 |
| IDX-INBRED | Il14H | refmap | 0.8119 |
| IDX-INBRED | Oh43 | chain_ordinary_256 | 0.5058 |
| IDX-INBRED | Oh43 | chain_ordinary_5000 | 0.5396 |
| IDX-INBRED | Oh43 | refmap | 0.7967 |
| IDX-RIL | B73xCML103 | chain_ordinary_256 | 0.6346 |
| IDX-RIL | B73xCML103 | chain_ordinary_5000 | 0.6544 |
| IDX-RIL | B73xCML103 | refmap | 0.8051 |
| IDX-RIL | B73xOh43 | chain_ordinary_256 | 0.6524 |
| IDX-RIL | B73xOh43 | chain_ordinary_5000 | 0.6694 |
| IDX-RIL | B73xOh43 | refmap | 0.8042 |
| IDX-RIL | B97xCML103 | chain_ordinary_256 | 0.4933 |
| IDX-RIL | B97xCML103 | chain_ordinary_5000 | 0.5288 |
| IDX-RIL | B97xCML103 | refmap | 0.7967 |
| IDX-RIL | Il14HxB97 | chain_ordinary_256 | 0.4952 |
| IDX-RIL | Il14HxB97 | chain_ordinary_5000 | 0.5292 |
| IDX-RIL | Il14HxB97 | refmap | 0.8037 |
| IDX-RIL | Oh43xIl14H | chain_ordinary_256 | 0.5024 |
| IDX-RIL | Oh43xIl14H | chain_ordinary_5000 | 0.5360 |
| IDX-RIL | Oh43xIl14H | refmap | 0.8053 |
| MIX-HYB | B73xTx303 | chain_ordinary_256 | 0.6380 |
| MIX-HYB | B73xTx303 | chain_ordinary_5000 | 0.6556 |
| MIX-HYB | B73xTx303 | refmap | 0.7936 |
| MIX-HYB | B97xCML459 | chain_ordinary_256 | 0.4949 |
| MIX-HYB | B97xCML459 | chain_ordinary_5000 | 0.5273 |
| MIX-HYB | B97xCML459 | refmap | 0.7862 |
| MIX-HYB | CML103xIa453 | chain_ordinary_256 | 0.4886 |
| MIX-HYB | CML103xIa453 | chain_ordinary_5000 | 0.5228 |
| MIX-HYB | CML103xIa453 | refmap | 0.7955 |
| MIX-HYB | Il14HxEP1 | chain_ordinary_256 | 0.4770 |
| MIX-HYB | Il14HxEP1 | chain_ordinary_5000 | 0.5089 |
| MIX-HYB | Il14HxEP1 | refmap | 0.7869 |
| MIX-HYB | Oh43xA188 | chain_ordinary_256 | 0.5085 |
| MIX-HYB | Oh43xA188 | chain_ordinary_5000 | 0.5405 |
| MIX-HYB | Oh43xA188 | refmap | 0.7975 |
| MIX-RIL | B73xTx303 | chain_ordinary_256 | 0.6297 |
| MIX-RIL | B73xTx303 | chain_ordinary_5000 | 0.6495 |
| MIX-RIL | B73xTx303 | refmap | 0.7940 |
| MIX-RIL | B97xCML459 | chain_ordinary_256 | 0.4958 |
| MIX-RIL | B97xCML459 | chain_ordinary_5000 | 0.5281 |
| MIX-RIL | B97xCML459 | refmap | 0.7866 |
| MIX-RIL | CML103xIa453 | chain_ordinary_256 | 0.4902 |
| MIX-RIL | CML103xIa453 | chain_ordinary_5000 | 0.5250 |
| MIX-RIL | CML103xIa453 | refmap | 0.7974 |
| MIX-RIL | Il14HxEP1 | chain_ordinary_256 | 0.4792 |
| MIX-RIL | Il14HxEP1 | chain_ordinary_5000 | 0.5110 |
| MIX-RIL | Il14HxEP1 | refmap | 0.7889 |
| MIX-RIL | Oh43xA188 | chain_ordinary_256 | 0.5048 |
| MIX-RIL | Oh43xA188 | chain_ordinary_5000 | 0.5372 |
| MIX-RIL | Oh43xA188 | refmap | 0.8004 |
| OUT-HYB | A188xEP1 | chain_ordinary_256 | 0.4898 |
| OUT-HYB | A188xEP1 | chain_ordinary_5000 | 0.5202 |
| OUT-HYB | A188xEP1 | refmap | 0.7811 |
| OUT-HYB | CML459xIa453 | chain_ordinary_256 | 0.4868 |
| OUT-HYB | CML459xIa453 | chain_ordinary_5000 | 0.5182 |
| OUT-HYB | CML459xIa453 | refmap | 0.7877 |
| OUT-HYB | EP1xIa453 | chain_ordinary_256 | 0.4783 |
| OUT-HYB | EP1xIa453 | chain_ordinary_5000 | 0.5092 |
| OUT-HYB | EP1xIa453 | refmap | 0.7788 |
| OUT-HYB | Tx303xA188 | chain_ordinary_256 | 0.4992 |
| OUT-HYB | Tx303xA188 | chain_ordinary_5000 | 0.5315 |
| OUT-HYB | Tx303xA188 | refmap | 0.7877 |
| OUT-HYB | Tx303xCML459 | chain_ordinary_256 | 0.4868 |
| OUT-HYB | Tx303xCML459 | chain_ordinary_5000 | 0.5199 |
| OUT-HYB | Tx303xCML459 | refmap | 0.7788 |
| OUT-INBRED | A188 | chain_ordinary_256 | 0.5106 |
| OUT-INBRED | A188 | chain_ordinary_5000 | 0.5412 |
| OUT-INBRED | A188 | refmap | 0.7982 |
| OUT-INBRED | CML459 | chain_ordinary_256 | 0.4868 |
| OUT-INBRED | CML459 | chain_ordinary_5000 | 0.5189 |
| OUT-INBRED | CML459 | refmap | 0.7819 |
| OUT-INBRED | EP1 | chain_ordinary_256 | 0.4673 |
| OUT-INBRED | EP1 | chain_ordinary_5000 | 0.4973 |
| OUT-INBRED | EP1 | refmap | 0.7633 |
| OUT-INBRED | Ia453 | chain_ordinary_256 | 0.4887 |
| OUT-INBRED | Ia453 | chain_ordinary_5000 | 0.5199 |
| OUT-INBRED | Ia453 | refmap | 0.7942 |
| OUT-INBRED | Tx303 | chain_ordinary_256 | 0.4869 |
| OUT-INBRED | Tx303 | chain_ordinary_5000 | 0.5212 |
| OUT-INBRED | Tx303 | refmap | 0.7771 |
| OUT-RIL | A188xEP1 | chain_ordinary_256 | 0.4953 |
| OUT-RIL | A188xEP1 | chain_ordinary_5000 | 0.5255 |
| OUT-RIL | A188xEP1 | refmap | 0.7859 |
| OUT-RIL | CML459xIa453 | chain_ordinary_256 | 0.4875 |
| OUT-RIL | CML459xIa453 | chain_ordinary_5000 | 0.5191 |
| OUT-RIL | CML459xIa453 | refmap | 0.7928 |
| OUT-RIL | EP1xIa453 | chain_ordinary_256 | 0.4813 |
| OUT-RIL | EP1xIa453 | chain_ordinary_5000 | 0.5125 |
| OUT-RIL | EP1xIa453 | refmap | 0.7846 |
| OUT-RIL | Tx303xA188 | chain_ordinary_256 | 0.5023 |
| OUT-RIL | Tx303xA188 | chain_ordinary_5000 | 0.5349 |
| OUT-RIL | Tx303xA188 | refmap | 0.7907 |
| OUT-RIL | Tx303xCML459 | chain_ordinary_256 | 0.4858 |
| OUT-RIL | Tx303xCML459 | chain_ordinary_5000 | 0.5195 |
| OUT-RIL | Tx303xCML459 | refmap | 0.7799 |

## B73 excess (hit_ratio[B73] / median(hit_ratio) over the 20 founders that are never a true parent in this corpus)

A value near 1 means B73 sits inside the background spread; a large value means B73 is elevated beyond every founder that provably contributes no reads. NOT yet relatedness-corrected (see refbias_mash.py for that).

| dataset | individual | arm | B73_excess | B73_hit_ratio | background_median | background_spread(min-max) |
|---|---|---|---:|---:|---:|---|
| IDX-HYB | B73xCML103 | chain_ordinary_256 | 2.171 | 0.8692 | 0.4003 | 0.3550-0.4576 |
| IDX-HYB | B73xCML103 | chain_ordinary_5000 | 2.116 | 0.8444 | 0.3991 | 0.3520-0.4536 |
| IDX-HYB | B73xCML103 | refmap | 1.856 | 0.6791 | 0.3660 | 0.3131-0.4015 |
| IDX-HYB | B73xOh43 | chain_ordinary_256 | 2.260 | 0.8813 | 0.3900 | 0.3678-0.4650 |
| IDX-HYB | B73xOh43 | chain_ordinary_5000 | 2.208 | 0.8586 | 0.3889 | 0.3671-0.4624 |
| IDX-HYB | B73xOh43 | refmap | 1.984 | 0.7009 | 0.3533 | 0.3347-0.4201 |
| IDX-HYB | B97xCML103 | chain_ordinary_256 | 1.552 | 0.6871 | 0.4428 | 0.3873-0.5006 |
| IDX-HYB | B97xCML103 | chain_ordinary_5000 | 1.467 | 0.6416 | 0.4373 | 0.3806-0.4905 |
| IDX-HYB | B97xCML103 | refmap | 1.083 | 0.3942 | 0.3639 | 0.3114-0.3984 |
| IDX-HYB | Il14HxB97 | chain_ordinary_256 | 1.661 | 0.6808 | 0.4098 | 0.3927-0.4933 |
| IDX-HYB | Il14HxB97 | chain_ordinary_5000 | 1.576 | 0.6373 | 0.4043 | 0.3879-0.4876 |
| IDX-HYB | Il14HxB97 | refmap | 1.195 | 0.3877 | 0.3245 | 0.3154-0.4373 |
| IDX-HYB | Oh43xIl14H | chain_ordinary_256 | 1.658 | 0.6761 | 0.4077 | 0.3875-0.5052 |
| IDX-HYB | Oh43xIl14H | chain_ordinary_5000 | 1.572 | 0.6328 | 0.4024 | 0.3830-0.5013 |
| IDX-HYB | Oh43xIl14H | refmap | 1.189 | 0.3836 | 0.3226 | 0.3091-0.4519 |
| IDX-INBRED | B73 | chain_ordinary_256 | 2.705 | 0.9967 | 0.3685 | 0.3454-0.4439 |
| IDX-INBRED | B73 | chain_ordinary_5000 | 2.705 | 0.9965 | 0.3684 | 0.3454-0.4439 |
| IDX-INBRED | B73 | refmap | 2.749 | 0.9885 | 0.3596 | 0.3365-0.4356 |
| IDX-INBRED | B97 | chain_ordinary_256 | 1.647 | 0.7111 | 0.4318 | 0.4024-0.5184 |
| IDX-INBRED | B97 | chain_ordinary_5000 | 1.565 | 0.6674 | 0.4263 | 0.3973-0.5113 |
| IDX-INBRED | B97 | refmap | 1.194 | 0.4214 | 0.3530 | 0.3361-0.4429 |
| IDX-INBRED | CML103 | chain_ordinary_256 | 1.455 | 0.6605 | 0.4540 | 0.3684-0.4808 |
| IDX-INBRED | CML103 | chain_ordinary_5000 | 1.371 | 0.6135 | 0.4473 | 0.3597-0.4703 |
| IDX-INBRED | CML103 | refmap | 0.987 | 0.3668 | 0.3716 | 0.2855-0.3955 |
| IDX-INBRED | Il14H | chain_ordinary_256 | 1.676 | 0.6469 | 0.3859 | 0.3706-0.5886 |
| IDX-INBRED | Il14H | chain_ordinary_5000 | 1.589 | 0.6045 | 0.3803 | 0.3663-0.5865 |
| IDX-INBRED | Il14H | refmap | 1.206 | 0.3542 | 0.2938 | 0.2846-0.5415 |
| IDX-INBRED | Oh43 | chain_ordinary_256 | 1.652 | 0.7046 | 0.4265 | 0.4037-0.4990 |
| IDX-INBRED | Oh43 | chain_ordinary_5000 | 1.566 | 0.6605 | 0.4217 | 0.3998-0.4902 |
| IDX-INBRED | Oh43 | refmap | 1.184 | 0.4145 | 0.3502 | 0.3334-0.4057 |
| IDX-RIL | B73xCML103 | chain_ordinary_256 | 2.135 | 0.8625 | 0.4039 | 0.3603-0.4581 |
| IDX-RIL | B73xCML103 | chain_ordinary_5000 | 2.081 | 0.8364 | 0.4020 | 0.3572-0.4537 |
| IDX-RIL | B73xCML103 | refmap | 1.816 | 0.6662 | 0.3669 | 0.3158-0.3999 |
| IDX-RIL | B73xOh43 | chain_ordinary_256 | 2.252 | 0.8840 | 0.3925 | 0.3693-0.4725 |
| IDX-RIL | B73xOh43 | chain_ordinary_5000 | 2.203 | 0.8615 | 0.3911 | 0.3681-0.4698 |
| IDX-RIL | B73xOh43 | refmap | 1.987 | 0.7062 | 0.3554 | 0.3365-0.4276 |
| IDX-RIL | B97xCML103 | chain_ordinary_256 | 1.530 | 0.6825 | 0.4460 | 0.3872-0.5042 |
| IDX-RIL | B97xCML103 | chain_ordinary_5000 | 1.445 | 0.6367 | 0.4405 | 0.3806-0.4938 |
| IDX-RIL | B97xCML103 | refmap | 1.062 | 0.3883 | 0.3656 | 0.3102-0.4007 |
| IDX-RIL | Il14HxB97 | chain_ordinary_256 | 1.638 | 0.6743 | 0.4117 | 0.3928-0.4977 |
| IDX-RIL | Il14HxB97 | chain_ordinary_5000 | 1.553 | 0.6310 | 0.4062 | 0.3883-0.4890 |
| IDX-RIL | Il14HxB97 | refmap | 1.173 | 0.3818 | 0.3255 | 0.3138-0.4353 |
| IDX-RIL | Oh43xIl14H | chain_ordinary_256 | 1.675 | 0.6851 | 0.4090 | 0.3887-0.5042 |
| IDX-RIL | Oh43xIl14H | chain_ordinary_5000 | 1.590 | 0.6422 | 0.4038 | 0.3846-0.5003 |
| IDX-RIL | Oh43xIl14H | refmap | 1.216 | 0.3949 | 0.3247 | 0.3120-0.4534 |
| MIX-HYB | B73xTx303 | chain_ordinary_256 | 2.144 | 0.8919 | 0.4160 | 0.3708-0.4778 |
| MIX-HYB | B73xTx303 | chain_ordinary_5000 | 2.091 | 0.8680 | 0.4150 | 0.3688-0.4748 |
| MIX-HYB | B73xTx303 | refmap | 1.829 | 0.6950 | 0.3800 | 0.3337-0.4250 |
| MIX-HYB | B97xCML459 | chain_ordinary_256 | 1.583 | 0.7216 | 0.4557 | 0.4169-0.5148 |
| MIX-HYB | B97xCML459 | chain_ordinary_5000 | 1.503 | 0.6772 | 0.4505 | 0.4107-0.5069 |
| MIX-HYB | B97xCML459 | refmap | 1.076 | 0.4022 | 0.3737 | 0.3365-0.4234 |
| MIX-HYB | CML103xIa453 | chain_ordinary_256 | 1.552 | 0.6771 | 0.4361 | 0.4173-0.5213 |
| MIX-HYB | CML103xIa453 | chain_ordinary_5000 | 1.471 | 0.6328 | 0.4301 | 0.4130-0.5160 |
| MIX-HYB | CML103xIa453 | refmap | 1.068 | 0.3713 | 0.3477 | 0.3362-0.4636 |
| MIX-HYB | Il14HxEP1 | chain_ordinary_256 | 1.632 | 0.6832 | 0.4186 | 0.4032-0.5506 |
| MIX-HYB | Il14HxEP1 | chain_ordinary_5000 | 1.549 | 0.6404 | 0.4133 | 0.3986-0.5465 |
| MIX-HYB | Il14HxEP1 | refmap | 1.115 | 0.3605 | 0.3233 | 0.3148-0.4841 |
| MIX-HYB | Oh43xA188 | chain_ordinary_256 | 1.637 | 0.7297 | 0.4457 | 0.4264-0.5089 |
| MIX-HYB | Oh43xA188 | chain_ordinary_5000 | 1.558 | 0.6865 | 0.4406 | 0.4220-0.5005 |
| MIX-HYB | Oh43xA188 | refmap | 1.119 | 0.4027 | 0.3597 | 0.3441-0.4037 |
| MIX-RIL | B73xTx303 | chain_ordinary_256 | 2.103 | 0.8809 | 0.4189 | 0.3693-0.4820 |
| MIX-RIL | B73xTx303 | chain_ordinary_5000 | 2.048 | 0.8542 | 0.4171 | 0.3667-0.4786 |
| MIX-RIL | B73xTx303 | refmap | 1.752 | 0.6739 | 0.3846 | 0.3294-0.4261 |
| MIX-RIL | B97xCML459 | chain_ordinary_256 | 1.581 | 0.7211 | 0.4560 | 0.4134-0.5232 |
| MIX-RIL | B97xCML459 | chain_ordinary_5000 | 1.501 | 0.6769 | 0.4509 | 0.4075-0.5157 |
| MIX-RIL | B97xCML459 | refmap | 1.076 | 0.4033 | 0.3749 | 0.3332-0.4326 |
| MIX-RIL | CML103xIa453 | chain_ordinary_256 | 1.548 | 0.6779 | 0.4378 | 0.4204-0.5081 |
| MIX-RIL | CML103xIa453 | chain_ordinary_5000 | 1.467 | 0.6329 | 0.4315 | 0.4160-0.5027 |
| MIX-RIL | CML103xIa453 | refmap | 1.067 | 0.3727 | 0.3493 | 0.3389-0.4482 |
| MIX-RIL | Il14HxEP1 | chain_ordinary_256 | 1.630 | 0.6851 | 0.4203 | 0.4069-0.5527 |
| MIX-RIL | Il14HxEP1 | chain_ordinary_5000 | 1.551 | 0.6424 | 0.4141 | 0.4023-0.5486 |
| MIX-RIL | Il14HxEP1 | refmap | 1.115 | 0.3626 | 0.3251 | 0.3169-0.4873 |
| MIX-RIL | Oh43xA188 | chain_ordinary_256 | 1.617 | 0.7247 | 0.4481 | 0.4308-0.5076 |
| MIX-RIL | Oh43xA188 | chain_ordinary_5000 | 1.540 | 0.6809 | 0.4421 | 0.4264-0.4999 |
| MIX-RIL | Oh43xA188 | refmap | 1.101 | 0.3939 | 0.3578 | 0.3451-0.4057 |
| OUT-HYB | A188xEP1 | chain_ordinary_256 | 1.602 | 0.7364 | 0.4597 | 0.4419-0.5063 |
| OUT-HYB | A188xEP1 | chain_ordinary_5000 | 1.528 | 0.6933 | 0.4538 | 0.4374-0.4971 |
| OUT-HYB | A188xEP1 | refmap | 1.057 | 0.3790 | 0.3585 | 0.3468-0.4006 |
| OUT-HYB | CML459xIa453 | chain_ordinary_256 | 1.586 | 0.7126 | 0.4493 | 0.4330-0.5524 |
| OUT-HYB | CML459xIa453 | chain_ordinary_5000 | 1.505 | 0.6693 | 0.4446 | 0.4291-0.5480 |
| OUT-HYB | CML459xIa453 | refmap | 1.067 | 0.3793 | 0.3554 | 0.3457-0.4908 |
| OUT-HYB | EP1xIa453 | chain_ordinary_256 | 1.637 | 0.7045 | 0.4304 | 0.4137-0.5892 |
| OUT-HYB | EP1xIa453 | chain_ordinary_5000 | 1.557 | 0.6616 | 0.4249 | 0.4095-0.5859 |
| OUT-HYB | EP1xIa453 | refmap | 1.106 | 0.3701 | 0.3348 | 0.3224-0.5312 |
| OUT-HYB | Tx303xA188 | chain_ordinary_256 | 1.555 | 0.7412 | 0.4768 | 0.4342-0.5240 |
| OUT-HYB | Tx303xA188 | chain_ordinary_5000 | 1.478 | 0.6962 | 0.4711 | 0.4270-0.5143 |
| OUT-HYB | Tx303xA188 | refmap | 1.030 | 0.3953 | 0.3836 | 0.3493-0.4095 |
| OUT-HYB | Tx303xCML459 | chain_ordinary_256 | 1.506 | 0.7290 | 0.4841 | 0.4194-0.5186 |
| OUT-HYB | Tx303xCML459 | chain_ordinary_5000 | 1.423 | 0.6826 | 0.4797 | 0.4119-0.5078 |
| OUT-HYB | Tx303xCML459 | refmap | 0.983 | 0.3921 | 0.3988 | 0.3320-0.4082 |
| OUT-INBRED | A188 | chain_ordinary_256 | 1.627 | 0.7540 | 0.4635 | 0.4466-0.5206 |
| OUT-INBRED | A188 | chain_ordinary_5000 | 1.552 | 0.7114 | 0.4585 | 0.4422-0.5131 |
| OUT-INBRED | A188 | refmap | 1.067 | 0.3904 | 0.3658 | 0.3519-0.4084 |
| OUT-INBRED | CML459 | chain_ordinary_256 | 1.517 | 0.7321 | 0.4826 | 0.4323-0.5135 |
| OUT-INBRED | CML459 | chain_ordinary_5000 | 1.439 | 0.6869 | 0.4775 | 0.4246-0.5037 |
| OUT-INBRED | CML459 | refmap | 0.982 | 0.3845 | 0.3914 | 0.3391-0.4046 |
| OUT-INBRED | EP1 | chain_ordinary_256 | 1.587 | 0.7163 | 0.4513 | 0.4368-0.5135 |
| OUT-INBRED | EP1 | chain_ordinary_5000 | 1.509 | 0.6730 | 0.4460 | 0.4324-0.5077 |
| OUT-INBRED | EP1 | refmap | 1.033 | 0.3653 | 0.3535 | 0.3410-0.4283 |
| OUT-INBRED | Ia453 | chain_ordinary_256 | 1.674 | 0.6926 | 0.4138 | 0.3909-0.6641 |
| OUT-INBRED | Ia453 | chain_ordinary_5000 | 1.594 | 0.6510 | 0.4084 | 0.3869-0.6630 |
| OUT-INBRED | Ia453 | refmap | 1.175 | 0.3739 | 0.3181 | 0.3034-0.6327 |
| OUT-INBRED | Tx303 | chain_ordinary_256 | 1.491 | 0.7278 | 0.4881 | 0.4094-0.5311 |
| OUT-INBRED | Tx303 | chain_ordinary_5000 | 1.409 | 0.6799 | 0.4824 | 0.4018-0.5200 |
| OUT-INBRED | Tx303 | refmap | 0.992 | 0.4009 | 0.4041 | 0.3274-0.4183 |
| OUT-RIL | A188xEP1 | chain_ordinary_256 | 1.607 | 0.7417 | 0.4616 | 0.4397-0.5063 |
| OUT-RIL | A188xEP1 | chain_ordinary_5000 | 1.533 | 0.6991 | 0.4561 | 0.4354-0.4968 |
| OUT-RIL | A188xEP1 | refmap | 1.064 | 0.3860 | 0.3626 | 0.3496-0.3985 |
| OUT-RIL | CML459xIa453 | chain_ordinary_256 | 1.593 | 0.7054 | 0.4428 | 0.4253-0.5864 |
| OUT-RIL | CML459xIa453 | chain_ordinary_5000 | 1.514 | 0.6625 | 0.4377 | 0.4217-0.5827 |
| OUT-RIL | CML459xIa453 | refmap | 1.074 | 0.3753 | 0.3493 | 0.3358-0.5327 |
| OUT-RIL | EP1xIa453 | chain_ordinary_256 | 1.640 | 0.6986 | 0.4259 | 0.4062-0.6031 |
| OUT-RIL | EP1xIa453 | chain_ordinary_5000 | 1.560 | 0.6561 | 0.4206 | 0.4018-0.6000 |
| OUT-RIL | EP1xIa453 | refmap | 1.119 | 0.3675 | 0.3285 | 0.3160-0.5482 |
| OUT-RIL | Tx303xA188 | chain_ordinary_256 | 1.550 | 0.7421 | 0.4789 | 0.4334-0.5269 |
| OUT-RIL | Tx303xA188 | chain_ordinary_5000 | 1.472 | 0.6968 | 0.4735 | 0.4268-0.5171 |
| OUT-RIL | Tx303xA188 | refmap | 1.033 | 0.3968 | 0.3842 | 0.3493-0.4102 |
| OUT-RIL | Tx303xCML459 | chain_ordinary_256 | 1.497 | 0.7262 | 0.4851 | 0.4216-0.5130 |
| OUT-RIL | Tx303xCML459 | chain_ordinary_5000 | 1.415 | 0.6791 | 0.4801 | 0.4140-0.5029 |
| OUT-RIL | Tx303xCML459 | refmap | 0.983 | 0.3893 | 0.3960 | 0.3357-0.4075 |

## In-index inbred lines: does the true founder dominate?

| dataset | individual(=true founder) | arm | true_founder_hit_ratio | rank_among_25 | runner_up | runner_up_hit_ratio | margin |
|---|---|---|---:|---:|---|---:|---:|
| IDX-INBRED | B73 | chain_ordinary_256 | 0.9967 | 1 | Oh7B | 0.4439 | 0.5527 |
| IDX-INBRED | B73 | chain_ordinary_5000 | 0.9965 | 1 | Oh7B | 0.4439 | 0.5526 |
| IDX-INBRED | B73 | refmap | 0.9885 | 1 | Oh7B | 0.4356 | 0.5529 |
| IDX-INBRED | B97 | chain_ordinary_256 | 0.9821 | 1 | B73 | 0.7111 | 0.2710 |
| IDX-INBRED | B97 | chain_ordinary_5000 | 0.9829 | 1 | B73 | 0.6674 | 0.3156 |
| IDX-INBRED | B97 | refmap | 0.9883 | 1 | Ms71 | 0.4429 | 0.5454 |
| IDX-INBRED | CML103 | chain_ordinary_256 | 0.9797 | 1 | B73 | 0.6605 | 0.3192 |
| IDX-INBRED | CML103 | chain_ordinary_5000 | 0.9809 | 1 | B73 | 0.6135 | 0.3674 |
| IDX-INBRED | CML103 | refmap | 0.9883 | 1 | Tzi8 | 0.3955 | 0.5928 |
| IDX-INBRED | Il14H | chain_ordinary_256 | 0.9791 | 1 | B73 | 0.6469 | 0.3322 |
| IDX-INBRED | Il14H | chain_ordinary_5000 | 0.9803 | 1 | B73 | 0.6045 | 0.3758 |
| IDX-INBRED | Il14H | refmap | 0.9887 | 1 | P39 | 0.5415 | 0.4473 |
| IDX-INBRED | Oh43 | chain_ordinary_256 | 0.9816 | 1 | B73 | 0.7046 | 0.2770 |
| IDX-INBRED | Oh43 | chain_ordinary_5000 | 0.9825 | 1 | B73 | 0.6605 | 0.3220 |
| IDX-INBRED | Oh43 | refmap | 0.9884 | 1 | B97 | 0.4153 | 0.5730 |

## Held-out inbred lines: negative control (no true founder in index -- profile should be flat, no spike)

| dataset | individual | arm | top_founder | top_hit_ratio | spread(max/min) |
|---|---|---|---|---:|---:|
| OUT-INBRED | A188 | chain_ordinary_256 | B73 | 0.7540 | 1.688x |
| OUT-INBRED | A188 | chain_ordinary_5000 | B73 | 0.7114 | 1.609x |
| OUT-INBRED | A188 | refmap | B97 | 0.4298 | 1.221x |
| OUT-INBRED | CML459 | chain_ordinary_256 | B73 | 0.7321 | 1.694x |
| OUT-INBRED | CML459 | chain_ordinary_5000 | B73 | 0.6869 | 1.618x |
| OUT-INBRED | CML459 | refmap | Ms71 | 0.4046 | 1.193x |
| OUT-INBRED | EP1 | chain_ordinary_256 | B73 | 0.7163 | 1.640x |
| OUT-INBRED | EP1 | chain_ordinary_5000 | B73 | 0.6730 | 1.557x |
| OUT-INBRED | EP1 | refmap | P39 | 0.4283 | 1.259x |
| OUT-INBRED | Ia453 | chain_ordinary_256 | B73 | 0.6926 | 1.772x |
| OUT-INBRED | Ia453 | chain_ordinary_5000 | P39 | 0.6630 | 1.714x |
| OUT-INBRED | Ia453 | refmap | P39 | 0.6327 | 2.085x |
| OUT-INBRED | Tx303 | chain_ordinary_256 | B73 | 0.7278 | 1.778x |
| OUT-INBRED | Tx303 | chain_ordinary_5000 | B73 | 0.6799 | 1.692x |
| OUT-INBRED | Tx303 | refmap | Mo18W | 0.4183 | 1.278x |

## Hybrid / RIL: parent-ratio check

Hybrid expectation: ~50/50 between the two true parents (0 breakpoints, whole-genome 1:1 mix). RIL expectation is dataset-specific (~40 breakpoints per haplotype) -- exact per-individual ratio needs the segment-length derivation (not computed by this script; see the plan's Phase 3). Only reported here for individuals where BOTH parents are indexed founders (IDX-HYB/IDX-RIL) or the MIX case below -- OUT-HYB/OUT-RIL have no true-parent signal to check by construction.

| dataset | individual | arm | parentA | parentB | A_hit_ratio | B_hit_ratio | A/(A+B) |
|---|---|---|---|---|---:|---:|---:|
| IDX-HYB | B73xCML103 | chain_ordinary_256 | B73 | CML103 | 0.8692 | 0.6044 | 0.590 |
| IDX-HYB | B73xCML103 | chain_ordinary_5000 | B73 | CML103 | 0.8444 | 0.6155 | 0.578 |
| IDX-HYB | B73xCML103 | refmap | B73 | CML103 | 0.6791 | 0.6751 | 0.501 |
| IDX-HYB | B73xOh43 | chain_ordinary_256 | B73 | Oh43 | 0.8813 | 0.6413 | 0.579 |
| IDX-HYB | B73xOh43 | chain_ordinary_5000 | B73 | Oh43 | 0.8586 | 0.6503 | 0.569 |
| IDX-HYB | B73xOh43 | refmap | B73 | Oh43 | 0.7009 | 0.7011 | 0.500 |
| IDX-HYB | B97xCML103 | chain_ordinary_256 | B97 | CML103 | 0.7327 | 0.7054 | 0.510 |
| IDX-HYB | B97xCML103 | chain_ordinary_5000 | B97 | CML103 | 0.7257 | 0.7036 | 0.508 |
| IDX-HYB | B97xCML103 | refmap | B97 | CML103 | 0.6775 | 0.6746 | 0.501 |
| IDX-HYB | Il14HxB97 | chain_ordinary_256 | Il14H | B97 | 0.6910 | 0.7279 | 0.487 |
| IDX-HYB | Il14HxB97 | chain_ordinary_5000 | Il14H | B97 | 0.6886 | 0.7239 | 0.487 |
| IDX-HYB | Il14HxB97 | refmap | Il14H | B97 | 0.6681 | 0.6724 | 0.498 |
| IDX-HYB | Oh43xIl14H | chain_ordinary_256 | Oh43 | Il14H | 0.7291 | 0.6935 | 0.513 |
| IDX-HYB | Oh43xIl14H | chain_ordinary_5000 | Oh43 | Il14H | 0.7253 | 0.6915 | 0.512 |
| IDX-HYB | Oh43xIl14H | refmap | Oh43 | Il14H | 0.6759 | 0.6725 | 0.501 |
| IDX-RIL | B73xCML103 | chain_ordinary_256 | B73 | CML103 | 0.8625 | 0.6176 | 0.583 |
| IDX-RIL | B73xCML103 | chain_ordinary_5000 | B73 | CML103 | 0.8364 | 0.6290 | 0.571 |
| IDX-RIL | B73xCML103 | refmap | B73 | CML103 | 0.6662 | 0.6890 | 0.492 |
| IDX-RIL | B73xOh43 | chain_ordinary_256 | B73 | Oh43 | 0.8840 | 0.6411 | 0.580 |
| IDX-RIL | B73xOh43 | chain_ordinary_5000 | B73 | Oh43 | 0.8615 | 0.6501 | 0.570 |
| IDX-RIL | B73xOh43 | refmap | B73 | Oh43 | 0.7062 | 0.6998 | 0.502 |
| IDX-RIL | B97xCML103 | chain_ordinary_256 | B97 | CML103 | 0.7248 | 0.7195 | 0.502 |
| IDX-RIL | B97xCML103 | chain_ordinary_5000 | B97 | CML103 | 0.7175 | 0.7174 | 0.500 |
| IDX-RIL | B97xCML103 | refmap | B97 | CML103 | 0.6639 | 0.6903 | 0.490 |
| IDX-RIL | Il14HxB97 | chain_ordinary_256 | Il14H | B97 | 0.6919 | 0.7279 | 0.487 |
| IDX-RIL | Il14HxB97 | chain_ordinary_5000 | Il14H | B97 | 0.6889 | 0.7243 | 0.487 |
| IDX-RIL | Il14HxB97 | refmap | Il14H | B97 | 0.6623 | 0.6779 | 0.494 |
| IDX-RIL | Oh43xIl14H | chain_ordinary_256 | Oh43 | Il14H | 0.7568 | 0.6621 | 0.533 |
| IDX-RIL | Oh43xIl14H | chain_ordinary_5000 | Oh43 | Il14H | 0.7540 | 0.6600 | 0.533 |
| IDX-RIL | Oh43xIl14H | refmap | Oh43 | Il14H | 0.7088 | 0.6404 | 0.525 |

## MIX (one indexed parent x one held-out parent): where does the held-out half's signal go?

The indexed parent has a true home in the index; the held-out parent does not. This isolates reference-bias fallback behavior with a same-run internal control -- half the reads in this exact file have nowhere true to go.

| dataset | individual | arm | indexed_parent | indexed_hit_ratio | held_out_parent | B73_hit_ratio | B73_excess |
|---|---|---|---|---:|---|---:|---:|
| MIX-HYB | B73xTx303 | chain_ordinary_256 | B73 | 0.8919 | Tx303 | 0.8919 | 2.144 |
| MIX-HYB | B73xTx303 | chain_ordinary_5000 | B73 | 0.8680 | Tx303 | 0.8680 | 2.091 |
| MIX-HYB | B73xTx303 | refmap | B73 | 0.6950 | Tx303 | 0.6950 | 1.829 |
| MIX-HYB | B97xCML459 | chain_ordinary_256 | B97 | 0.7446 | CML459 | 0.7216 | 1.583 |
| MIX-HYB | B97xCML459 | chain_ordinary_5000 | B97 | 0.7404 | CML459 | 0.6772 | 1.503 |
| MIX-HYB | B97xCML459 | refmap | B97 | 0.6874 | CML459 | 0.4022 | 1.076 |
| MIX-HYB | CML103xIa453 | chain_ordinary_256 | CML103 | 0.6897 | Ia453 | 0.6771 | 1.552 |
| MIX-HYB | CML103xIa453 | chain_ordinary_5000 | CML103 | 0.6886 | Ia453 | 0.6328 | 1.471 |
| MIX-HYB | CML103xIa453 | refmap | CML103 | 0.6439 | Ia453 | 0.3713 | 1.068 |
| MIX-HYB | Il14HxEP1 | chain_ordinary_256 | Il14H | 0.7321 | EP1 | 0.6832 | 1.632 |
| MIX-HYB | Il14HxEP1 | chain_ordinary_5000 | Il14H | 0.7303 | EP1 | 0.6404 | 1.549 |
| MIX-HYB | Il14HxEP1 | refmap | Il14H | 0.6928 | EP1 | 0.3605 | 1.115 |
| MIX-HYB | Oh43xA188 | chain_ordinary_256 | Oh43 | 0.7492 | A188 | 0.7297 | 1.637 |
| MIX-HYB | Oh43xA188 | chain_ordinary_5000 | Oh43 | 0.7466 | A188 | 0.6865 | 1.558 |
| MIX-HYB | Oh43xA188 | refmap | Oh43 | 0.6926 | A188 | 0.4027 | 1.119 |
| MIX-RIL | B73xTx303 | chain_ordinary_256 | B73 | 0.8809 | Tx303 | 0.8809 | 2.103 |
| MIX-RIL | B73xTx303 | chain_ordinary_5000 | B73 | 0.8542 | Tx303 | 0.8542 | 2.048 |
| MIX-RIL | B73xTx303 | refmap | B73 | 0.6739 | Tx303 | 0.6739 | 1.752 |
| MIX-RIL | B97xCML459 | chain_ordinary_256 | B97 | 0.7499 | CML459 | 0.7211 | 1.581 |
| MIX-RIL | B97xCML459 | chain_ordinary_5000 | B97 | 0.7454 | CML459 | 0.6769 | 1.501 |
| MIX-RIL | B97xCML459 | refmap | B97 | 0.6924 | CML459 | 0.4033 | 1.076 |
| MIX-RIL | CML103xIa453 | chain_ordinary_256 | CML103 | 0.7062 | Ia453 | 0.6779 | 1.548 |
| MIX-RIL | CML103xIa453 | chain_ordinary_5000 | CML103 | 0.7055 | Ia453 | 0.6329 | 1.467 |
| MIX-RIL | CML103xIa453 | refmap | CML103 | 0.6668 | Ia453 | 0.3727 | 1.067 |
| MIX-RIL | Il14HxEP1 | chain_ordinary_256 | Il14H | 0.7289 | EP1 | 0.6851 | 1.630 |
| MIX-RIL | Il14HxEP1 | chain_ordinary_5000 | Il14H | 0.7263 | EP1 | 0.6424 | 1.551 |
| MIX-RIL | Il14HxEP1 | refmap | Il14H | 0.6892 | EP1 | 0.3626 | 1.115 |
| MIX-RIL | Oh43xA188 | chain_ordinary_256 | Oh43 | 0.7424 | A188 | 0.7247 | 1.617 |
| MIX-RIL | Oh43xA188 | chain_ordinary_5000 | Oh43 | 0.7391 | A188 | 0.6809 | 1.540 |
| MIX-RIL | Oh43xA188 | refmap | Oh43 | 0.6828 | A188 | 0.3939 | 1.101 |

## Reciprocity: 5x5 matrix over in-index inbreds (sequence sharing is symmetric; algorithmic bias need not be)

### arm=chain_ordinary_256

row = true source, column = hit_ratio credited to that founder

| source (row) / credited (col) | B73 | B97 | CML103 | Il14H | Oh43 |
|---|---|---|---|---|---|
| **B73** | 0.9967 | 0.4295 | 0.3742 | 0.3582 | 0.4225 |
| **B97** | 0.7111 | 0.9821 | 0.4443 | 0.4219 | 0.4995 |
| **CML103** | 0.6605 | 0.4713 | 0.9797 | 0.3915 | 0.4557 |
| **Il14H** | 0.6469 | 0.4563 | 0.3993 | 0.9791 | 0.4566 |
| **Oh43** | 0.7046 | 0.5027 | 0.4329 | 0.4249 | 0.9816 |

### arm=chain_ordinary_5000

row = true source, column = hit_ratio credited to that founder

| source (row) / credited (col) | B73 | B97 | CML103 | Il14H | Oh43 |
|---|---|---|---|---|---|
| **B73** | 0.9965 | 0.4295 | 0.3742 | 0.3582 | 0.4225 |
| **B97** | 0.6674 | 0.9829 | 0.4369 | 0.4159 | 0.4928 |
| **CML103** | 0.6135 | 0.4585 | 0.9809 | 0.3831 | 0.4453 |
| **Il14H** | 0.6045 | 0.4480 | 0.3936 | 0.9803 | 0.4492 |
| **Oh43** | 0.6605 | 0.4951 | 0.4270 | 0.4191 | 0.9825 |

### arm=refmap

row = true source, column = hit_ratio credited to that founder

| source (row) / credited (col) | B73 | B97 | CML103 | Il14H | Oh43 |
|---|---|---|---|---|---|
| **B73** | 0.9885 | 0.4214 | 0.3655 | 0.3516 | 0.4150 |
| **B97** | 0.4214 | 0.9883 | 0.3622 | 0.3514 | 0.4158 |
| **CML103** | 0.3668 | 0.3655 | 0.9883 | 0.3044 | 0.3531 |
| **Il14H** | 0.3542 | 0.3539 | 0.3039 | 0.9887 | 0.3604 |
| **Oh43** | 0.4145 | 0.4153 | 0.3505 | 0.3575 | 0.9884 |

