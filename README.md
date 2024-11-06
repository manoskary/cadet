[![Python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2208.14819-B31B1B.svg)](https://arxiv.org/abs/2208.14819)
[![Conference](http://img.shields.io/badge/ISMIR-2022-4b44ce.svg)](https://ismir2022.ismir.net/program/papers)


# Cadet
Cadence Detection in Symbolic Classical Music using Graph Neural Networks (ISMIR2022).

## Introduction

This repository contains the training and the models from the paper **Cadence Detection in Symbolic Classical Music using Graph Neural Networks** submitted at ISMIR 2022.
[pdf](https://arxiv.org/pdf/2208.14819.pdf)

## Requirements and Installation

#### Installation

If you are using conda please install the environment:

```shell
conda env create -f environment.yml
conda activate cadet
cd path/to/cadet
pip install .
```

 
If this fails, follow the steps bellow.

#### Pre-installation requirements:

```shell
conda create -n cadet python=3.8 pip
conda activate cadet
```
Proceed by visiting the following websites and installing the appropriate version of the following packages.

- Pytorch >=1.8.1 [link](https://pytorch.org/get-started/locally/);
- DGL >= 0.7 [link](https://www.dgl.ai/pages/start.html).

After the installation of Pytorch and DGL with the platform of your choice (pip and conda supported on this repo), 
you can install the rest of the requirements using the following commands:

- To install other requirements using pip:
```shell
pip install -r requirements.txt
```


- To install other requirement using conda:
```shell
conda env update -f environment.yml
```

You might also need to install the repo as a package if you are running from the terminal.
It's suggested to use the experimental pip install to keep up with new versions or edits you might want to make:
```shell
pip install . -e
```


## Getting Started

To run the pre-trained models you will need a [wandb account](https://wandb.ai). 
You can find the project results and download the models at [https://wandb.ai/melkisedeath/Cadence Detection](https://wandb.ai/melkisedeath/Cadence%20Detection).


#### Train a model

```shell
cd cadet/train
python train_lightning.py --dataset wtc --cad-type pac 
```
Using the above command you will train PAC detection on the Bach fugues of the 1st Welle tempered clavier book.

If you wish not to log your run with WANBD on the cloud then run:
```shell
WANDB_MODE=offline python script.py --args
```

#### Load and train pre-trained model

```shell
cd cadet/train
python train_lightning --dataset wtc --cad-type pac --wandb_id --load_from_checkpoints 
```
Using the above command you will load a pretrained model for PAC detection, pretrained on String Quartets and fine tune it on the Bach fugues of the 1st Well tempered clavier book.

#### Reproduce results from pre-trained model

Load pretrained model from Bach Fugues PAC prediction and skip training only to reproduce results on the test set.

```shell
cd cadet/train
python train_lightning --dataset wtc --cad-type pac --wandb_id  --skip-training --load_from_checkpoints 
```


## Create Graph from Score

To create the graphs from the score you will need to provide a directory of scores.
```shell
cd cadet/utils
python create_homo_graph_dataset.py --data_dir 
```

#### External Repositories

An external repository is provided with the dataset which can also be found as a git sub-module

You can browse the latest version of the repository [here](https://github.com/melkisedeath/tonnetzcad).

## Feature Extraction

The feature extraction includes three categories of featrures:
- The General Note features
- The graph topology features
- The Cadence Relevant features

The topology features are produced by taking the $N$ first eigenvectors of the Laplacian of the Adjacency matrix.
The script that produces this values can be find in `cad/utils/pos_enc.py`. The complete list of the other features is presented below.

#### List of Features

The features are computed per note/rest in the score. Chord refers to the set of notes that have the same onset value as the current note.

| Function Name                                   | Type   | Description                                                                                        | 
|-------------------------------------------------|--------|----------------------------------------------------------------------------------------------------| 
| **General Note Features**                       |        |                                                                                                    |
| onset_feature.score_position                    | float  | normalized onset between 0 and 1                                                                   | 
| duration_feature.duration                       | float  | duration of notes in formalized value                                                              |
| fermata_feature.fermata                         | binary | If note has fermata                                                                                |
| grace_feature.n_grace                           | float  | How many grace notes on this onset position                                                        |
| grace_feature.grace_pos                         | float  | Which grace note in the sequence of grace notes                                                    |
| onset_feature.onset                             | float  | normalized onset feature                                                                           |
| polynomial_pitch_feature.pitch                  | float  | normalized midi pitch between 0 and 1                                                              |
| grace_feature.grace_note                        | binary | is grace note                                                                                      |
| relative_score_position_feature.score_position  | float  |                                                                                                    |
| slur_feature.slur_incr                          | float  |                                                                                                    |
| slur_feature.slur_decr                          | float  |                                                                                                    |
| time_signature_feature.time_signature_num_1     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_2     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_3     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_4     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_5     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_6     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_7     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_8     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_9     | float  |                                                                                                    |
| time_signature_feature.time_signature_num_10    | float  |                                                                                                    |
| time_signature_feature.time_signature_num_11    | float  |                                                                                                    |
| time_signature_feature.time_signature_num_12    | float  |                                                                                                    |
| time_signature_feature.time_signature_num_other | float  |                                                                                                    |
| time_signature_feature.time_signature_den_1     | float  |                                                                                                    |
| time_signature_feature.time_signature_den_2     | float  |                                                                                                    |
| time_signature_feature.time_signature_den_4     | float  |                                                                                                    |
| time_signature_feature.time_signature_den_8     | float  |                                                                                                    |
| time_signature_feature.time_signature_den_16    | float  |                                                                                                    |
| time_signature_feature.time_signature_den_other | float  |                                                                                                    |
| vertical_neighbor_feature.n_total               | float  |                                                                                                    |
| vertical_neighbor_feature.n_above               | float  |                                                                                                    |
| vertical_neighbor_feature.n_below               | float  |                                                                                                    |
| vertical_neighbor_feature.highest_pitch         | float  |                                                                                                    |
| vertical_neighbor_feature.lowest_pitch          | float  |                                                                                                    |
| vertical_neighbor_feature.pitch_range           | float  |                                                                                                    |
| int_vec1                                        | int    | first value of interval vector computed for all notes on the onset of the current one.             |
| int_vec2                                        | int    | second value of interval vector computed for all notes on the onset of the current one.            |
| int_vec3                                        | int    | third value of interval vector  computed for all notes on the onset of the current one.            |
| int_vec4                                        | int    | fourth value of interval vector  computed for all notes on the onset of the current one.           |
| int_vec5                                        | int    | fifth value of interval vector  computed for all notes on the onset of the current one.            |
| int_vec6                                        | int    | sixth value of interval vector  computed for all notes on the onset of the current one.            |
| M/m                                             | binary | is the interval vector equivalent to Major or Minor chord                                          |
| sus4                                            | binary | is the interval vector equivalent to a sus4 chord                                                  |
| M7                                              | binary | is the interval vector equivalent to a dominant 7 chord                                            |
| M7wo5                                           | binary | is the interval vector equivalent to a dominant 7 chord without the 5th                            |
| Mmaj7                                           | binary | is the interval vector equivalent to a major 7 chord                                               |
| Mmaj7maj9                                       | binary | is the interval vector equivalent to a major 7 chord with major 9                                  |
| M9                                              | binary | is the interval vector equivalent to a dominant chord with 9                                       |
| M9wo5                                           | binary | is the interval vector equivalent to a dominant chord with 9 without the 5th                       |
| m7                                              | binary | is the interval vector equivalent to a minor 7 chord                                               |
| m7wo5                                           | binary | is the interval vector equivalent to a minor 7 chord without the 5th                               |
| m9                                              | binary | is the interval vector equivalent to a minor chord with a major 9                                  |
| m9wo5                                           | binary | is the interval vector equivalent to a minor chord with a major 9 without the 5                    |
| m9wo7                                           | binary | is the interval vector equivalent to a minor chord with a major 9 without the 7                    |
| mmaj7                                           | binary | is the interval vector equivalent to a minor chord with a major 7                                  |
| Maug                                            | binary | is the interval vector equivalent to a augmented chord                                             |
| Maug7                                           | binary | is the interval vector equivalent to a augmented chord with 7                                      |
| mdim                                            | binary | is the interval vector equivalent to a diminshed chord                                             |
| mdim7                                           | binary | is the interval vector equivalent to a diminshed chord with 7                                      |
| is_maj_triad                                    | binary | is the set of notes present on the onset of the current note a major triad                         |
| is_pmaj_triad                                   | binary | is the set of notes present on the onset of the current note a perfect major triad                 |
| is_min_triad                                    | binary | is the set of notes present on the onset of the current note a minor triad                         |
| ped_note                                        | binary | is the current note a pedal note                                                                   |
| hv_7                                            | binary | is the highest voice of the chord a 7 of the chord compared to the lowest pitch                    |
| hv_5                                            | binary | is the highest voice of the chord a 5 of the chord compared to the lowest pitch                    |
| hv_3                                            | binary | is the highest voice of the chord a 3 of the chord compared to the lowest pitch                    |
| hv_1                                            | binary | is the highest voice of the chord an octave of the chord compared to the lowest pitch              |
| chord_has_2m                                    | binary | does the chord have a 2m                                                                           |
| chord_has_2M                                    | binary | does the chord have a 2M                                                                           |
| **Cadence Features**                            |        |                                                                                                    |
| perfect_triad                                   | binary | is the chord a perfect triad                                                                       |
| perfect_major_triad                             | binary | is the chord a perfect major                                                                       |
| is_sus4                                         | binary | is the chord a sus4                                                                                |
| in_perfect_triad_or_sus4                        | binary | is the chord a perfect triad or a sus4                                                             |
| highest_is_3                                    | binary | is the highest voice a 3rd compared to the lowest on this onset                                    |
| highest_is_1                                    | binary | is the highest voice a 8th compared to the lowest on this onset                                    |
| bass_compatible_with_I                          | binary | is the bass compatible with the Tonal of the scale                                                 |
| bass_compatible_with_I_scale                    | binary |                                                                                                    |
| one_comes_from_7                                | binary | does the one (compared to the lowest voice) comes from a leading tone (compared to previous onset) |
| one_comes_from_1                                | binary | was the one (compared to the lowest voice) present on the previous onset                           |
| one_comes_from_2                                | binary | does the one (compared to the lowest voice) comes from a second (compared to previous onset)       |
| three_comes_from_4                              | binary | does the third (compared to the lowest voice) comes from a fourth (compared to previous onset)     |
| five_comes_from_5                               | binary | was the fifth (compared to the lowest voice) present on the previous onset                         |
| strong_beat                                     | binary | is the current onset a strong beat                                                                 |
| sustained_note                                  | binary | is the current note a pedal note (more or equal to a bar's duration)                               |
| rest_highest                                    | binary | is there a rest on the highest voice                                                               |
| rest_lowest                                     | binary | is there a rest on any middle voice                                                                |
| rest_middle                                     | binary | is there a rest on the lowest voice                                                                |
| voice_ends                                      | binary | is there a voice end after the particular onset                                                    |
| v7                                              | binary | does the chord have a seventh                                                                      |
| v7-3                                            | binary | does the chord have a seventh and a third                                                          |
| has_7                                           | binary | does the chord have a seventh                                                                      |
| has_9                                           | binary | does the chord have a ninth                                                                        |
| bass_voice                                      | binary | is the bass on this voice/note                                                                     |
| bass_moves_chromatic                            | binary | does the bass (lowest pitch) move chromatically                                                    |
| bass_moves_octave                               | binary | does the bass (lowest pitch) move with an octave jump                                              |
| bass_compatible_v-i                             | binary | does the bass (lowest pitch) move with a V to I                                                    |
| bass_compatible_i-v                             | binary | does the bass (lowest pitch) move with a I to V                                                    |
| bass_moves_2M                                   | binary | does the bass (lowest pitch) does the mass move with a second major interval.                      |
