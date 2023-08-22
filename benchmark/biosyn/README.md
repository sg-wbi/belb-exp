# BioSyn

Here are reported the steps necessary to run BioSyn with BELB data.

For each command make sure that (a) you have the correct virtual environment activated
and (b) you are in the right directory.
In the example the virtual environment name is signaled in brackets
and after that there is the folder (repository) name.

### 1. Setup BioSyn

You first need to clone the BioSyn repository:

```bash
(belb) home $ git clone https://github.com/dmis-lab/BioSyn
```

Then follow the instructions to setup the environment
including obtaining the original data provided by the authors.

You also need to follow the instruction to be able to run the `preprocess` modules.
See [here](https://github.com/dmis-lab/BioSyn/tree/master/preprocess#pre-processing-datasets-and-dictionaries).

| :warning: WARNING |
| :---------------- |

You should also comment out [these two lines](https://github.com/dmis-lab/BioSyn/blob/master/preprocess/query_preprocess.py#L256)

```python
abbr_dict = abbr_resolver.resolve(txt_file)
concept = apply_abbr_dict(concept, abbr_dict)
```

as we directly provide a way to resolve abbreviations (`--ab3p`)

Please also edit [this line](https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/data_loader.py#L60),
changing it from

```python
concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
```

to

```python
concept_files = sorted(glob.glob(os.path.join(data_dir, "*.concept")))
```

This is necessary because the code `eval.py` only stores the mentions text with the predictions,
but we need to connect them to the document they came from.

### 2. Sanity check

Place all the bash scripts you find in this folder into the BioSyn one:

```bash
(belb) belb-benchmark $ cp benchmark/biosyn/*.sh /path/to/BioSyn/
```

Make sure that the script `train_ncbi_disease_original.sh` (training with data provided by the authors)
runs without issues (at least until it starts the actual training).

### 3. Convert BELB data

Run:

```bash
(belb) belb-benchmark $ python -m benchmark.biosyn.biosyn --run input --in_dir data_directory --out_dir biosyn_directory --belb belb_directory
```

This will convert BELB data into the format required by BioSyn and store it into `in_dir`.

### 4. BioSyn preprocessing

Run:

```bash
(BioSyn) BioSyn $ ./create_belb_runs.sh data_directory
```

This will place each KB in the correct corpus directory
and pre-process the data as required by BioSyn.

Note that the path passed to `--dir` must be the same one passed to `--in_dir` in step 3.

### 5. Train

Edit the file `train_belb.sh`:

```bash
DIRECTORY="/path/to/biosyn/preprocessed/data"
```

`DIRECTORY` should point to the preprocessed data (steps 3 and 4).

Run:

```bash
(BioSyn) BioSyn $ ./train_belb.sh
```

to train a BioSyn model on each of the BELB corpora.
Modify the `CORPORA` variable to specify only a subset.

### 6. Predict

Run:

```bash
(BioSyn) BioSyn $ ./predict_belb.sh
```

### 7. Gather results

Run:

```bash
(belb) belb-benchmark $ python -m benchmark.biosyn.biosyn --run output --in_dir data_directory --out_dir biosyn_directory --belb belb_directory
```
