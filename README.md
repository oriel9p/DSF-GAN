# Downstream Feedback GAN

## Overview

This repository contains the implementation of Downstream Feedback Generative Adversarial Network (DSF-GAN), a novel architecture designed to address the challenge of generating high-utility synthetic tabular data while also prioritizing privacy concerns. The primary innovation lies in incorporating feedback from a downstream prediction model mid-training to enhance the utility of synthetic samples.

### Abstract

Utility and privacy are two crucial measurements of synthetic tabular data. While privacy concerns have been dramatically improved with the use of Generative Adversarial Networks (GANs), generating high-utility synthetic samples remains challenging. To increase the samples’ utility, we propose a novel architecture called DownStream Feedback Generative Adversarial Network (DSF-GAN). This approach uses feedback from a downstream prediction model mid-training, to add valuable information to the generator’s loss function. Hence, DSF-GAN harnesses a downstream prediction task to increase the utility of the synthetic samples. To properly evaluate our method, we implemented and tested it using five data sets. Our experiments show better model performance when training on DSF-GAN-generated synthetic samples compared to synthetic data generated using another State-of-the-art (SOTA) GAN architecture. All code and datasets used in this research are openly available for ease of reproduction.

## Instructions

To run the DSF-GAN implementation, follow the steps below:

1. Install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your CSV datasets in the `/datasets/` folder.

3. Run the main script using the command:
   ```bash
   python main.py


### Submission to ICLR 2024 TinyPapers
This research project, including the DSF-GAN implementation and experimental datasets, has been submitted to ICLR 2024 TinyPapers. The aim is to contribute to the broader research community and facilitate reproducibility.

Feel free to explore the code and datasets to replicate our experiments or build upon our work. If you have any questions or feedback, please contact the authors.