# PB-SVRGE Algorithm Implementation

This repository contains an implementation of the PB-SVRGE (PowerBall Stochastic Variance Reduced Gradient with Enhancement)[^1] algorithm on selected datasets. The code allows sweeping over one of the following hyperparameters:

- **`gamma`**: Sweeps over different values of the $\gamma$ parameter.
- **`eta`**: Sweeps over different values of the learning rate ($\eta$).
- **`b`**: Sweeps over different values of the batch size ($b$).

## Workflow

For each dataset and hyperparameter value, the main function performs the following steps:

1. **Obtain Approximate Optimal Solution**: Uses the ADAM optimizer to compute an approximate optimal solution if the file `w_star.pkl` does not exist.
2. **Load Dataset**: Loads the dataset from a specified path.
3. **Run PB-SVRGE**: Executes the PB-SVRGE optimization algorithm with the specified parameter.
4. **Record Results**: Records the training history, final loss, and elapsed time.
5. **Save Convergence Curve**: Saves the convergence curve of the objective gap as PNG files in the `./img/` directory.

## How to Run

First, download the required datasets and extract them into the `dataset` folder. The datasets can be found in the [PKU Disk](https://disk.pku.edu.cn/link/AAFE324EB4001A4619BB35CC7EC759017A) (link expires on 2025-07-15 22:29) or [LIBSVM Data Library](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

To sweep over a specific hyperparameter, use one of the following commands:

```bash
python pbsvrge.py --para_type gamma
python pbsvrge.py --para_type b
python pbsvrge.py --para_type eta
```

## Requirements

- Python 3.9.16 was okay for this project.
- Required packages are listed in `requirements.txt`. Install them by running:

    ```bash
    pip install -r requirements.txt
    ```

## References

[^1]: Zhuang Yang. 2023. "Improved PowerBall stochastic optimization algorithms for large-scale machine learning." *Journal of Machine Learning Research*, 24(241), 1–29.


