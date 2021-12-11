# Detecting Parkinsons with ML
[![Open Word-Level In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pshah123/parkinsons-AI/blob/master/Train.ipynb)

We'll use the data from UC Irvine's amazing dataset repository, specifically the [Parkinsons ML database](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/).

There are two datasets within this. The first is in the root folder (`parkinsons.data` which is included here too) and can be used to detect Parkinsons. The second is within the `telemonitoring/` directory and contains UDPR scores for us to predict.

## Approach

Parkinsons detection islikely best done with an XGBoost since outputs are 0 or 1 and it seems mostly linear.

The UDPR is very hard to fine tune with XGBoost. With an NN in Keras, we can fit much better. There are still some very bad apples in our data/predictions bur the performance is overall/on average much better.

Both approaches are provided in the Jupyter notebook for this repo (`Train.ipynb`). Run using `jupyter notebook` from this repo's root folder in the terminal.

You can find more info on the datasets in UCI's database. ([info for Parkinsons detection](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.names)) ([info for UDPR](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names))

## Requirements

- Python 3 (I **highly** recommend using Anaconda as this will save you a TON of time)
- XGBoost (`pip install xgboost`)
- Keras (`pip install keras`)
- sklearn (`pip install scikit-learn`)
- Jupyter ([instructions](http://jupyter.org/install))
- Pandas (`pip install pandas`)
- NumPy (`pip install numpy`)
- Tensorflow or another Keras backend (`pip install tensorflow` or for GPU assuming CUDA and CUDNN already installed and in PATH, `pip install tensorflow-gpu`)


## Credits

I don't own this dataset, it's provided in the link earlier.

Citations for the datasets used:

```
'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', 
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 
BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)
```

```
A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson.s disease progression by non-invasive speech tests',
IEEE Transactions on Biomedical Engineering (to appear).
```

**If you use this repo please cite it:**

```
Cuuupid 💔. (2018, April 4). Detecting Parkinsons with AI (Version DOI). Zenodo. 
http://doi.org/10.5281/zenodo.1211859
```

Bibtex:
```
@misc{cuuupid_loves_you_2018_1211859,
  author       = {Cuuupid 💔},
  title        = {Detecting Parkinsons with AI},
  month        = apr,
  year         = 2018,
  doi          = {10.5281/zenodo.1211859},
  url          = {https://doi.org/10.5281/zenodo.1211859}
}
```
