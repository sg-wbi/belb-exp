# tmVar v3

To run tmVar you need to first create the input files with

## Input files

```bash
(belb-benchmark) user $ python -m benchmark.tmvar.tmvar --run input --in_dir path/to/gnormplus/tmvar_input --belb_dir /belb/directory
```

This will create BioC files without annotations.
Make sure that `--in_dir` points to the directory where you have install "GNormPlus" (you can run `bin/run_gnormplus.sh` for this).

## Run GNormPlus

This is because tmVar requires as input text pre-annotated with gene mentions.

In the GNormPlus directory run:

```bash
java -Xmx60G -Xms30G -jar GNormPlus.jar "./tmvar_input/snp" "./tmvar_input_gene/snp" setup_nlm_gene.txt 
java -Xmx60G -Xms30G -jar GNormPlus.jar "./tmvar_input/osiris" "./tmvar_input_gene/osiris" setup_nlm_gene.txt 
java -Xmx60G -Xms30G -jar GNormPlus.jar "./tmvar_input/tmvar" "./tmvar_input_gene/tmvar" setup_nlm_gene.txt 
```

## Run tmVar

On Linux tmVar v3 requires a CRF library to be install at the system level.
This is unwise and simply often not possible if you do not have administrator privileges.

This means that you need to turn to Windows (or Mac possibly, we did not test this).
You can follow the instruction on how to run tmVar on Windows with the `/tmvar_input_gene/*`

```bash
java -XX:ActiveProcessorCount=2 -Xmx10G -Xms5G -jar tmVar.jar input output
```

This will generate both BioC and PubTator files with tmVar predictions
