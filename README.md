# Brain network study during resting states

![](https://scx2.b-cdn.net/gfx/news/hires/2018/anextgeneegc.jpg)

## Description
The human brain is composed by millions of neurons which are interconnected together
and which send and receive signals to initiate actions in the whole human body. To analyze and evaluate electrical activity and flow patterns in the brain, the EEG test is used. In this project, we analyze two datasets of EEG data recorded from 64 electrodes with subjects at rest in (i) eye-open and (ii) eyes-closed conditions, respectively. The analysis includes the topics of connectivity graphs, graph theory indices, motif analysis, and community detection. We also explore the differences of results obtained from the construction of the brain graph using different methods and densities, and we compute motifs and communities between channels in the brain using various algorithms.

A more comprehensive analysis can be found in the [project report](./report_BINM_proj_neuro_group-01.pdf).

## Repository structure

The repository organization is the following:

```bash
brain-network-analysis
├── .gitignore
├── README.md
├── data
│    ├── img                        # images obtained from the analysis
│    ├── S002                       # EEG data of subject 002
│    └── channel_locations.txt      # coordinates of electrodes locations
├── brain_analysis_tools            # tools for brain analysis
│    ├── connectivity_graph.py
│    ├── graph_theory_indices.py
│    └── motif_analyzer.py
├── main.ipynb                      # jupyter notebook of the project
├── BINM_proj2(neuro)_ay2021.pdf    # guidelines and questions for the project
├── report_BINM_proj_neuro_group-01 # report of the analysis
└── env.yml                         # conda environment yaml file
```