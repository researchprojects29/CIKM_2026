# Beyond Answer Accuracy: Diagnosing Perception Failures in Multimodal Mathematical Reasoning

This repository contains the code, prompts, datasets, and experimental artifacts accompanying our work on **perception-aware evaluation of Multimodal Large Language Models (MLLMs)** for mathematical reasoning.

## Repository Contents

* **`Prompts/`**
  Contains all prompts used throughout the project. The prompt file naming convention follows the terminology and identifiers used in the paper for easy correspondence.

* **`Fleiss Kappa Test/`**
  Includes the annotation protocol, inter-annotator agreement analysis, and detailed Fleiss' $\kappa$ evaluation results used to validate APS generation and reference triples.

* **`Detailed Experimental Result Analysis/`**
  Provides extended analyses, interpretations, and conclusions for the plots and experimental results presented in the paper.

* **`Scripts/`**
  Contains Python implementations for:

  * Noise injection
  * Auxiliary Perception Set (APS) evaluation using triples
  * Knowledge graph triple generation
  * Other experimental utilities used in the evaluation pipeline

* **`APS/`**
  Contains sample Auxiliary Perception Set (APS) questions, reference triples, and related examples illustrating the perception evaluation framework.

## Overview

The proposed framework separates **perception** from **reasoning** by representing multimodal understanding as structured knowledge graph triples and evaluating them through the **Auxiliary Perception Set (APS)**. This enables fine-grained diagnosis of perception failures that remain hidden when evaluating only final-answer accuracy.

The repository includes all necessary resources to reproduce the perception evaluation pipeline, generate structured triples, inject controlled perturbations, and analyze multimodal reasoning performance across different MLLMs.


Contains sample Auxiliary Perception Set (APS) questions, reference triples, and example annotations used for perception evaluation.
