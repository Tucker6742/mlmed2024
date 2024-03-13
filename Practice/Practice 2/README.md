# Usage

1. Run `pip install -r requirements.txt`
2. Downaload the [Fetal Head Circumference Dataset](https://zenodo.org/records/1327317)
3. Set up the fodler like this
   ```
    data
    ├───training_set
    ├───testing_set
    ├───test_set_pixel_size.csv
    └───training_set_pixel_size_and_HC.csv
   ```
   `images/test_set` is the folder of test image from the dataset, `images/training_set` is the folder include annoation and origin image
4. Run `Data processing and sampling.py` first to set up the directory tree then  `Data analysis.py` for generating labels data and data_df for train/val/test in `src/Python script` folder

## `Data processing and sampling.py`

Edit the `root_dir` in the `main` function to match the `data` directory tree

## `Data analysis.py`

Edit `root_dir`  in `main` fuction to match the directory tree
