# arboEL

Here are reported the steps necessary to run arboEL with BELB data.

For each command make sure that (a) you have the correct virtual environment activated
and (b) you are in the right directory.
In the example the virtual environment name is signaled in brackets
and after that there is the folder (repository) name.

### 1. Setup arboEL

You first need to clone the arboEL repository:

```bash
(belb) home $ git clone https://github.com/dhdhagar/arboEL
```

Then follow the instructions to setup the environment
including obtaining the original data provided by the authors.

### 2. Sanity check

Place all the scripts you find in this folder into the arboEL one.

Make sure that the script `train_medmentions_original.sh` (training with data provided by the authors)
runs without issues (at least until it starts the actual training).

### 3. Convert BELB data

Run:

```bash
(belb) belb-benchmark $ python -m benchmark.arboel.arboel --run input --in_dir data_directory --out_dir arboel_directory --belb belb_directory
```

This will convert BELB data into the format required by arboEL and store it into `in_dir`.

### 4. arboEL preprocessing

Run:

```bash
(blink37) arboEL $ ./create_belb_runs.sh data_directory
```

This will place each KB in the correct corpus directory and
pre-process the data as required by arboEL.

Note that the path passed to `--dir` must be the same one passed to `--in_dir` in step 3.

Now you also need to run the following script to finalize the pre-processing:

```bash
(blink37) arboEL $ python preprocess_belb_run.py --dir data_directory
```

### 5. Get BioBERT

Run:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
(blink37) arboEL $ git lfs install
(blink37) arboEL $ git clone https://huggingface.co/dmis-lab/biobert-v1.1
```

this will download the BioBERT model weights.
We need to do this because arboEL uses an old version of the [transformers](https://github.com/huggingface/transformers) library,
which does not support direct download.

### 6. Train

Edit the file `train_belb.sh`:

```bash
DIRECTORY="/path/to/arboEL/preprocessed/data"
PLM="LOCAL/path/to/biobert"
```

`DIRECTORY` should point to the preprocessed data (steps 3 and 4) and  `PLM` to the *local* BioBERT weights (step 5)

Run:

```bash
(blink37) arboEL $ ./train_belb.sh
```

to train an arboEL model on each of the BELB corpora.
Modify the `CORPORA` variable to specify only a subset.

### 7. Setup predict

Edit the `predict_belb.sh` similarly to `train_belb.sh` (step 6).

Run:

```bash
(blink37) arboEL $ grep "Best performance in epoch" models/trained/belb/run1/**/pos_neg_loss/no_type/log.txt
```

which will print something like this:

```
models/trained/belb/run1/bc5cdr_chemical/pos_neg_loss/no_type/log.txt:04/17/2023 22:24:10 - INFO - Blink -   Best performance in epoch: 4
models/trained/belb/run1/bc5cdr_disease/pos_neg_loss/no_type/log.txt:04/17/2023 20:11:18 - INFO - Blink -   Best performance in epoch: 3
models/trained/belb/run1/linnaeus/pos_neg_loss/no_type/log.txt:04/18/2023 18:50:01 - INFO - Blink -   Best performance in epoch: 4
models/trained/belb/run1/ncbi_disease/pos_neg_loss/no_type/log.txt:04/17/2023 18:58:29 - INFO - Blink -   Best performance in epoch: 4
models/trained/belb/run1/nlm_chem/pos_neg_loss/no_type/log.txt:04/18/2023 06:24:46 - INFO - Blink -   Best performance in epoch: 4
models/trained/belb/run1/bc5cdr_disease/pos_neg_loss/no_type/log.txt:04/17/2023 20:11:18 - INFO - Blink -   Best performance in epoch: 3
models/trained/belb/run1/nlm_gene/pos_neg_loss/no_type/log.txt:04/19/2023 20:56:09 - INFO - Blink -   Best performance in epoch: 4
```

Edit the `predict_belb.sh` with the information from the previous step:

```bash
if [ "$CORPUS" == "gnormplus" ] || [ "$CORPUS" == "bc5cdr_disease" ]; then
    BEST="epoch_3"
else
    BEST="epoch_4"
fi
```

to ensure we are using the best set of weights (on the development set) for the predictions on the test set.

### 7. Predict

Run:

```bash
(blink37) arboEL $ ./predict_belb.sh
```

### 8. Gather results

Run:

```bash
(belb) belb-benchmark $ python -m benchmark.arboel.arboel --run output --in_dir data_directory --out_dir arboeL_directory --belb belb_directory
```
