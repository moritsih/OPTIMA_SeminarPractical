#!/bin/bash

# ceiling
#python main.py --source_domains Spectralis Topcon Cirrus --experiment_name Supervised --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2

# should be as good as ceiling if SVDNA works well
#python main.py --source_domains Spectralis Topcon  --experiment_name Spectralis-Topcon --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
#python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
#python main.py --source_domains Topcon Cirrus --experiment_name Topcon-Cirrus --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2

# should be significantly lower than ceiling
#python main.py --source_domains Spectralis Topcon  --experiment_name Spectralis-Topcon-noSVDNA --exp_with_svdna "False" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
#python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-noSVDNA --exp_with_svdna "False" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
#python main.py --source_domains Topcon Cirrus --experiment_name Topcon-Cirrus-noSVDNA --exp_with_svdna "False" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2

# more extreme cases
#python main.py --source_domains Spectralis --experiment_name Spectralis --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
#python main.py --source_domains Topcon --experiment_name Topcon --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
#python main.py --source_domains Cirrus --experiment_name Cirrus --exp_with_svdna "True" --with_histogram "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2

# Ablation
python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-NoiseAdaptOnly --exp_with_svdna "True" --with_histogram "False" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
python main.py --source_domains Spectralis Cirrus --experiment_name Spectralis-Cirrus-HistmatchingOnly --exp_with_svdna "True" --histogram_matching_only "True" --batch_size 8 --epochs 130 --use_official_testset "True" --loss_smoothing 1e-5 --train_split 0.8 --val_split 0.2
