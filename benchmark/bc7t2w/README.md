# BioCreative VII Track 2 Winner

Here are reported the steps necessary to run BC7TW2 with BELB data.

For each command make sure that (a) you have the correct virtual environment activated
and (b) you are in the right directory.
In the example the virtual environment name is signaled in brackets
and after that there is the folder (repository) name.

Run this command to setup the model and its input

```bash
(belb) belb-benchmark$ ./bin/bc7t2w/run_bc7t2w_step1.sh <belb directory> <output directory>
```

Now head to to `<output directory>` and edit the `src/settings.yaml` file.

As we run the tool on NLM-Chem, we want to avoid using gold entity mentions from this corpus.

To do this you need to substitute these lines:

```yaml
corpus_for_expansion:
    - "datasets/NLMChem/BC7T2-NLMChem-corpus-train.BioC.json"
    - "datasets/NLMChem/BC7T2-NLMChem-corpus-dev.BioC.json"
    - "datasets/NLMChem/BC7T2-NLMChem-corpus-test.BioC.json"
```

with: 

```yaml
corpus_for_expansion: []
```

Now follow the instruction [here](https://github.com/bioinformatics-ua/biocreativeVII_track2)
to create a conda enviroment with the required dependencies. 

Then to run the model on the appropriate BELB corpora run:

```bash
(biocreative) belb-benchmark$ CUDA_VISIBLE_DEVICES=0 ./bin/bc7t2w/run_bc7t2w_step2.sh <belb directory> <output directory>
```

Now you can deactivate the conda enviroment and go back to the one for the `belb-benchmark` repository.

Finally, you can gather the results with:

```bash
(belb) belb-benchmark$ ./bin/bc7t2w/run_bc7t2w_step3.sh <belb directory> <output directory>
```


