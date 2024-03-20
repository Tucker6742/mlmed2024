# Usage

1. Run `pip install -r requirements.txt`
2. Download the [Fetal Head Circumference Dataset](https://zenodo.org/records/1327317)
3. Set up the folder like this
   ```
    data
    ├───training_set
    ├───testing_set
    ├───test_set_pixel_size.csv
    └───training_set_pixel_size_and_HC.csv
   ```

   `test_set` is the folder of test image from the dataset, `training_set` is the folder include annotation and origin image
4. Run each file in the `src/Python script` folder in order for task that described below 

## `1. Data processing and sampling.py`

- Usage
    Edit the `root_dir` in the `main` function to match the `data` directory tree

- Behavior
    Set up the train/val/test split for data, create directory tree, and setup files order in folders

## `2. Data analysis.py`
- Usage
    Edit `root_dir` in `main` function to match the directory tree

- Behavior
    Setup train/val/test dataframe, make OBB labels for each split and plot ellipse + OBB for some image in train/val/test

## `3. Data validation.py`

- Usage
    Edit 
    - `split` for type of data split (train/val/test)
    - `index` for the index of image in the split
    - `root_dir` in `main` function to match the directory tree

- Behavior
    Plot the image with OBB and ellipse of an image in the split

## `4. Model training.py`

- Usage
    Edit
    - `data_root_dir` for the root directory of the data
    - `evaluation_dir` for the directory of the evaluation data
    - `test_dir` for the directory of the test data
    - `root_dir` in `main` function for the output directory of the model results

- Behavior
    Train the model with the training data, evaluate the model with the evaluation data, and predict the test data and evaluate on new data using the trained model `yolov8n-obb`