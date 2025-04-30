# How to Run

There are two ways to run this program:

### 1. Batch Run (All Experiments on All Datasets)

Use the following command to run all experiments on **all** datasets:

```bash
python batch_run_test.py
```
**Warning:**
This method is less repetitive but may take a very long time to complete.
Some datasets, such as the HIV dataset (which contains over 30,000 graphs), significantly increase the runtime.

### 2. Selectively run datasets and feature combinations

To run the program with user input, allowing you to select specific datasets and feature combinations:
```bash
python full_multimodal_project.py
```
