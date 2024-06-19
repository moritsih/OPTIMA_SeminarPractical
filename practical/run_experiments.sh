#!/bin/bash

# ceiling
python main.py --source_domains Spectralis Topcon Cirrus --experiment_name Supervised-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2

# should be as good as ceiling if SVDNA works well
python main.py --source_domains Spectralis Topcon  --experiment_name Spectralis-Topcon-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Topcon Cirrus --experiment_name Topcon-Cirrus-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2

# should be significantly lower than ceiling
python main.py --source_domains Spectralis Topcon  --experiment_name Spectralis-Topcon-noSVDNA-unoff --exp_with_svdna "False" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-noSVDNA-unoff --exp_with_svdna "False" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Topcon Cirrus --experiment_name Topcon-Cirrus-noSVDNA-unoff --exp_with_svdna "False" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2

# more extreme cases
python main.py --source_domains Spectralis --experiment_name Spectralis-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Topcon --experiment_name Topcon-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Cirrus --experiment_name Cirrus-unoff --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2

# Ablation
python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-NoiseAdaptOnly-unoff --exp_with_svdna "True" --with_histogram "False" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-HistmatchingOnly-unoff --exp_with_svdna "True" --histogram_matching_only "True" --batch_size 8 --epochs 130 --use_official_testset "False" --loss_smoothing 1e-5 --train_split 0.7 --val_split 0.1 --test_split 0.2
