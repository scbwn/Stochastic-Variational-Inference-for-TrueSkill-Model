# Stochastic-Variational-Inference-for-TrueSkill-Model

## Project Overview

This repository implements Stochastic Variational Inference (SVI) for the TrueSkill model, a Bayesian player ranking system for competitive games. The goal is to estimate the posterior distribution of player skills, represented by continuous latent variables. (Part of course assignment for STA414/STA2014 and CSC412/CSC2506, University of Toronto, Winter 2020)

## Background

TrueSkill is a generalization of the Elo rating system, originally developed for Halo 2. This implementation is based on the 2007 NeurIPS paper by Herbrich et al. and Carl Rasmussen's probabilistic machine learning course at Cambridge.

## Model Definition

The simplified TrueSkill model assumes each player has an unknown skill $z_i \in \mathbb{R}$. The model considers $N$ players.

## Implementation

- Stochastic Variational Inference (SVI) for approximate Bayesian inference
- TrueSkill model implementation with continuous latent variables
- Customizable hyperparameters for experimentation
- Easy-to-use code for estimating player skills

## Features

- SVI using stochastic gradient descent
- Gaussian approximate posterior distribution
- Evidence lower bound (ELBO) optimization
- Support for multiple games and players

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib (for visualization)

## Usage

1. Clone repository
2. Install requirements
3. Run SVI algorithm
4. Estimate player skills using trained model

## Example Use Cases

- Player ranking in competitive games
- Team performance prediction
- Skill-based matchmaking

## References

- Herbrich et al. (2007) - TrueSkill: A Bayesian Skill Rating System (http://papers.nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system.pdf)
- Carl Rasmussen's probabilistic machine learning course (http://mlg.eng.cam.ac.uk/teaching/4f13/1920/)
