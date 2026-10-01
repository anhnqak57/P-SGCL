# BRCA R Enrichr stage

The R stage accepts selected_genes.csv from analysis.brca.selection. It does not use a hard-coded gene list.

Install the required R packages:

~~~bash
Rscript analysis/brca/r/requirements.R
~~~

Run the two-step hand-off:

~~~bash
python -m analysis.brca.selection \
  --gene-csv dataset/brca-v5/features/BRCA_mRNA_top.csv \
  --labels dataset/brca-v5/features/labels.csv \
  --label-source external \
  --allow-positional-labels \
  --output-dir outputs/brca_rf \
  --top-n 100 \
  --seed 777

Rscript analysis/brca/r/enrichr_brca.R \
  --genes outputs/brca_rf/selected_genes.csv \
  --output-dir outputs/brca_r_enrichr \
  --config analysis/brca/r/enrichr_config.json
~~~

The R command sends selected symbols to Enrichr. It uses KEGG_2021_Human and GO 2023 human libraries, adjusted P-value < 0.05, no background list, and writes raw responses, filtered terms, plots, a manifest, and session_info.txt to the requested output directory.

On the tested Windows host, the default enrichR package client could not connect. enrichr_brca.R therefore calls Enrichr addList/export REST endpoints through httr with IPv4 forced. This changes transport only; it does not change libraries or filtering. All enrichment outputs are ignored by Git because they are generated, time-dependent remote responses.

