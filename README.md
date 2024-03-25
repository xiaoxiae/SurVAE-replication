# SurVAE

A replication of "SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows", implemented as the final project for the "Generative Neural Networks" class of the University of Heidelberg, 2024.

[[Our Report](https://github.com/xiaoxiae/SurVAE-replication/blob/main/report/main.pdf)] [[Original Paper](https://arxiv.org/abs/2007.02731)]

![Overview image](overview.png)


## Structure

The repository contains the following:

```
├── assets     # relevant PDFs
├── report     # source code for the report
├── notebooks  # Jupyter notebooks
├── saves      # save states for MNIST
└── survae     # source code
```

## Running the code

1. install the requirements (`pip install -r requirements.txt`)
2. open and  run any notebook in `notebooks/`


## Tests

To run all of the tests, use the following command:

```
python3 -m survae.test
```
