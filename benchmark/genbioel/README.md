# GenBioEL

Here are reported the steps necessary to run GenBioEL with BELB data.

For each command make sure that (a) you have the correct virtual environment activated
and (b) you are in the right directory.
In the example the virtual environment name is signaled in brackets
and after that there is the folder (repository) name.

### 1. Setup GenBioEL

You first need to clone the GenBioEL repository:

```bash
(belb) home $ git clone https://github.com/Yuanhy1997/GenBioEL
```

Then follow the instructions to setup the environment 
including obtaining the original data provided by the authors.

### 2. Sanity check

Place all the scripts you find in this folder into the BioSyn one.

Make sure that the script `train_ncbi.sh` (training with data provided by the authors)
runs without issues (at least until it starts the actual training).

### 3. Convert BELB data

Run:

```bash
(belb) belb-benchmark $ python -m benchmark.genbioel.genbioel --run input --in_dir data_directory --out_dir genbioel_directory --belb belb_directory
```

This will convert BELB data into the format required by BioSyn and store it into `in_dir`.

### 4. GenBioEL preprocessing

Run:

```bash
(genbioel) GenBioEL $ ./create_belb_runs.sh data_directory
```

This will place each KB in the correct corpus directory 
and pre-process the data as required by GenBioEL.

Note that the path passed to `--dir` must be the same one passed to `--in_dir` in step 3.

### 5. Get BART

The training script downloads the BART weights at every call, 
which may throw this error if called repeatedly:

```bash
Traceback (most recent call last):
  File "/vol/home-vol3/wbi/gardasam/.venv/genbioel/lib/python3.9/site-packages/huggingface_hub/utils/_errors.py", line 259, in hf_raise_for_status
    response.raise_for_status()
  File "/vol/home-vol3/wbi/gardasam/.venv/genbioel/lib/python3.9/site-packages/requests/models.py", line 1021, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 504 Server Error: Gateway Time-out for url: https://huggingface.co/api/models/facebook/bart-large
```

To avoid this run:

```bash
(genbioel) GenBioEL $ git install lfs
(genbioel) GenBioEL $ git clone https://huggingface.co/facebook/bart-large 
```

to save locally the BART weights and tokenizer.

### 6. Train 

Edit the file `train_belb.sh`:

```bash
DIRECTORY="/path/to/genbioel/preprocessed/data"
# If you skipped step 6: 'facebook/bart-large' 
PLM="path/to/local/facebook/bart/weights" 
```

`DIRECTORY` should point to the preprocessed data (steps 3 and 4) and `PLM` to the BART large model/tokenizer (step 5)

Run:

```bash
(genbioel) GenBioEL $ ./train_belb.sh
```

to train a GenBioEL model on each of the BELB corpora. 
Modify the `CORPORA` variable to specify only a subset.

### 7. Predict 

You need to change this [line](https://github.com/Yuanhy1997/GenBioEL/blob/main/src/train.py#L388)

from:

```python
pickle.dump([results, results_score], f)
```

to:

```python
pickle.dump([results, cui_results], f)
```


This will avoid storing all beam search probabilities and save instead the name converted to entities atomic labels.

Run:

```bash
(genbioel) GenBioEL $ ./predict_belb.sh
```

### 8. Gather results 

Run:

```bash
(belb-venv) belb-benchmark $ python -m benchmark.genbioel.genbioel --run output --in_dir data_directory --out_dir genbioel_directory --belb belb_directory
```
