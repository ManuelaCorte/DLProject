# Use instructions

## Download the repo

```
git clone https://github.com/ManuelaCorte/DLProject
cd DLProject
```

If you have conda

```
conda env create -f environment.yml
conda activate dl
```

otherwise

```
pip install -r requirements.txt
```

## Download the dataset

The dataset can be downloaded from https://drive.google.com/file/d/1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq/view?usp=drive_link and shoud be placed in `data/raw/`

## Run the code

To run the training code, you can use the following command:

```
cd src
python -m vgproject.train
```

otherwise you can use the notebook in `notebooks/train.ipynb`
