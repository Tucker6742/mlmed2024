# Usage

1. Run `pip install -r requirements.txt`
2. Downaload the [Fetal Head Circumference Dataset](https://zenodo.org/records/1327317)
3. Set up the fodler like this
   ```
    data
    ├───images
    │   ├───test_set
    │   └───training_set
    └───labels
        └───training_set
   ```
   `images/test_set` is the folder of test image from the dataset, `images/training_set` is the folder include annoation and origin image, `labels/training_set` is an empty label

   Put 2 csv file in `images` folder
4. Run `Data analysis.py` or `Running model.py` for corresponding task in `src/Python script` folder

## `Data analysis.py`

Edit the `data_path_train` and `data_path_test` in the `main` function to match the `mitbih_train.csv` and `mitbih_test.csv` file and run the code

## `Running model.py`

Edit `path_train` and `path_test` in $\text{Read and process data}$ section and then run the code
