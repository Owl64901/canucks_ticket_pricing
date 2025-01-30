# 🏒 Ticket Pricing Analysis of the Vancouver Canucks
## 📝 Overview
This repository contains the code and configuration required for the pricing model for Canucks tickets. The project includes steps for reading raw data, preprocessing, feature engineering, model training, and prediction. The workflow is managed using a Makefile, which simplifies running different stages of the pipeline.

## 🛠️ Setup and Installation
- Clone the repository to your local machine.
```bash
git clone git@github.com:bi-canucks/canucks_mds_capstone.git
cd canucks_mds_capstone
```
- Ensure you have conda installed.
- Create the conda environment specified in the environment.yml file:
```bash
conda env create -f environment.yml
```
- Activate the environment:
```bash
conda activate canucks_pricing
```

## 🚀 Usage

### 🎯 Predict and Build Dashboard
1. Make sure `output/prediction/` contains csv files of the date to be predicted, along with 5 days of data prior.
2. To generate predictions and build the dashboard, run:
```bash
make predict
```
This command will execute:
- python src/result_prediction.py: Make predictions using the trained model.\
The predictions will be saved to output/prediction/predicted_tickets_sales_{the predicted date}.csv.
- python src/app.py: Start the dashboard application.\
The dashboard will be accessible at http://127.0.0.1:8050/.
![Dashboard Demo](https://github.com/bi-canucks/canucks_mds_capstone/blob/main/output/prediction/dashboard_demo.gif)

### 🔄 Retraining Process

1. Make sure all the data used for training are in `data/short_summary/`
2. Adjust the hyperparameters as needed in `utils.py` (for the transformer model) and the individual model files (for scikit-learn models).
3. Set the output directory to `RETRAINED_MODEL_DIR` in your training script:
```python
if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)
```
4. To retrain the model from scratch, starting from reading the raw data to training and evaluating various models, run:
```bash
make model
```
This command will execute the following steps:
- setup: Create necessary directories.
- It will ask for the training date range **(for the cross-validation to work, it would be best to at least include 1.5 years)
- read_data: Read the raw data.
- preprocess: Preprocess the data. 
- feature_engineering: Perform feature engineering.
- split: Split the data into training and testing sets.
- choose_model: Interactively choose a model to train and evaluate.

5. Update the model paths in your prediction script to point to the retrained models. In `utils.py`, modify:
```python
TRANSFORMER_MODEL_BOWL_0_PATH = 'output/model/retrained/model_bowl_0.pth'
TRANSFORMER_MODEL_BOWL_1_PATH = 'output/model/retrained/model_bowl_1.pth'
POISSON_MODEL_PATH = 'output/model/retrained/poisson_model.joblib'
```
6. Run the prediction script to test the performance of the retrained models:
```bash
make predict
```

### 🥇 Updating Canucks' Ranking Data

The script `src/vancouver_ranking.py` fetches weekly data from the NHL API, based on dates in `data/output/processed.parquet`.  Since the league standings do not fluctuate greatly day to day, it will only fetch Monday dates, along with the earliest date in `processed.parquet` to ensure full coverage. Additionally, it will only fetch dates that are not already in `vancouver_ranking.csv`. The feature engineering script, `src/feature_engineering.py`, will incorporate the data from `vancouver_ranking.csv` when the pipeline is run.

Upon incorporation of new ticket sales data, run the following command from the root of the directory to update `vancouver_ranking.csv`.

```bash
  python src/vancouver_ranking_data.py
```


## 📊 Model Training Data
The training data used for the default model (i.e. production) contains data with `event_date` from Feb 15, 2022 to Apr 18, 2024.

## 📂 Directory Structure
```plaintext
repository
├── data/
│   ├── output/                   # Processed data files
│   ├── prediction_input/         # Input data for predictions
│   └── short_summary/            # Subdirectories of original data
├── eda_notebooks/                # includes some trial on random search and stat models as well as demo of using nhl api for ranking
├── output/
│   ├── eda_img/                  # Images from Exploratory Data Analysis
│   ├── model/                    # Trained model files
│   └── prediction/               # Prediction results (CSV files)
├── reports/                      # Reports
├── src/                          # Source code
└── tests/                        # Test files
└── Makefile                      # Makefile for automating tasks
└── environment.yml               # Conda environment file
```

- src/: Contains all source code files for data processing, model training, and evaluation.
- output/: Directory where outputs, such as EDA images and model files, will be stored.
  - output/eda_img/: Images generated during exploratory data analysis.
  - output/prediction/: Directory where prediction results will be saved. **(Please make sure it contains csv files of the date to be predicted, along with 5 days of prior data to allow the transformer model to work)**
  - output/model/: Directory where trained model files will be stored.
- data/: Directory for input and output data files.
  - data/output/: Directory where processed data files will be stored.
  - data/prediction_input/: Directory for input data files used for predictions.
  - data/short_summary/: Directory containing subdirectories of the original data.
- reports/: Directory for report generation.
- tests/: Directory containing test files.

  ## 📋 Additional Commands
- Run Exploratory Data Analysis (EDA):
```bash
make eda
```
- Clean Generated Files:
```bash
make clean
```
This will remove all generated files and directories.

## 📚 Dependencies
All dependencies are listed in the [environment.yml file](environment.yml). Ensure to create and activate the conda environment as specified in the setup instructions.

## 📑 Reports

Reports detailing the methodologies and data science techniques proposed and implemented throughout this project can be accessed here:
- [Proposal Report](reports/_build/pdf/book.pdf)
- [MDS Final Report](reports/final_report.pdf)
- [Technical Report](reports/technical_report.pdf) 


## 🙋‍♂️ Team Members
This project is the result of our team. Here are the contributors (in alphabetical order):
- Bill Wan (klwanbill@gmail.com)
- Nicole Bidwell (nrbidwell@gmail.com)
- Yan Zeng (yzeng11@student.ubc.ca)

Feel free to reach out to us if any problem or clarification is needed, even after our submission! 😊
  
