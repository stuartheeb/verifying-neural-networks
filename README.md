# Verifying Neural Networks

This project is a verifier for neural networks. It can verify for (currently only a limit set of architectures) whether an input image perturbed by an epsilon will still produce the correct initial label. The architectures support the following layer:
- Flatten
- Linear
- Conv2D
- ReLU
- LeakyReLU

The transformers are based on the [DeepPoly paper](https://dl.acm.org/doi/10.1145/3290354) (An abstract domain for certifying neural networks). Another resource is the paper [On the Paradox of Certified Training](https://arxiv.org/abs/2102.06700).

## Authors

This project was part of the course [Reliable and Trustworthy Artificial Intelligence](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lang=en&semkez=2023W&ansicht=ALLE&lerneinheitId=172754&) at ETH Zürich and was completed with Angéline Pouget and Ioana Branescu-Raspop.
