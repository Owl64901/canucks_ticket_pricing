.PHONY: all setup read_data preprocess feature_engineering split linear_regression random_forest xgboost polynomial_model tabnet_model evaluate_tabnet transformer_model clean interactive

# Default values
FULL_CAPACITY_DATE ?= 2022-02-15
CUTOFF_DATE ?= 2024-04-18

all: setup read_data preprocess feature_engineering split choose_model
predict: setup read_data preprocess feature_engineering split dashboard

setup:
	mkdir -p output/eda_img data/output output/model/default output/model/retrained

# Read raw data
read_data: src/read_data.py
	python src/read_data.py

# Preprocess data with full capacity date
preprocess: src/preprocessing.py
	python src/preprocessing.py --full_capacity_date $(FULL_CAPACITY_DATE)

# Feature Engineering
feature_engineering: src/feature_engineering.py
	python src/feature_engineering.py

# Create plots
eda: src/eda.py
	python src/eda.py

# Choose model interactively
choose_model:
	@echo "Select a model to run:"
	@echo "1. Linear Regression"
	@echo "2. Random Forest"
	@echo "3. XGBoost"
	@echo "4. Polynomial Model"
	@echo "5. TabNet Model"
	@echo "6. Transformer Model"
	@echo "7. Poisson Model"
	@read choice; \
	case $$choice in \
		1) make linear_regression ;; \
		2) make random_forest ;; \
		3) make xgboost ;; \
		4) make polynomial_model ;; \
		5) make run_tabnet ;; \
		6) make transformer_model ;; \
		7) make poisson ;; \
		*) echo "Invalid selection"; exit 1 ;; \
	esac

# Train & test split with cutoff date
split: src/split_dataset.py
	python src/split_dataset.py --cutoff_date $(CUTOFF_DATE)

# Train and evaluate linear regression model
linear_regression: src/models/baseline_model.py
	python -m src.models.baseline_model

# Train and evaluate random forest model
random_forest: src/models/random_forest_model.py
	python -m src.models.random_forest_model

# Train and evaluate xgboost model
xgboost: src/models/xgboost_model.py
	python -m src.models.xgboost_model

# Train and evaluate polynomial model
polynomial_model: src/models/polynomial_model.py
	python -m src.models.polynomial_model

# Combined target for training and evaluating TabNet
run_tabnet: tabnet_model evaluate_tabnet

# Train tabnet model
tabnet_model: src/models/tabnet_model.py
	python -m src.models.tabnet_model

# Evaluate tabnet model
evaluate_tabnet: src/models/evaluate_tabnet.py
	python -m src.models.evaluate_tabnet

# Train and evaluate transformer model
transformer_model: src/models/transformer_model.py
	python -m src.models.transformer_model

# Train and evaluate poisson model
poisson: src/models/poisson_model.py
	python -m src.models.poisson_model

# Run prediction and application
dashboard: src/result_prediction.py src/app.py
	python -m src.result_prediction
	python -m src.app

# Clean
clean: 
	rm -rf reports/_build output/eda_img/* data/output/* 

# Interactive target to collect dates
model:
	@echo "Enter the date range for training data:"
	@echo "Note: Ensure the training data includes at least 1.5 years of data for cross-validation."
	@read -p "Training start date(default: $(FULL_CAPACITY_DATE)): " full_capacity_date; \
	read -p "Cutoff date for training/testing split (default: $(CUTOFF_DATE)): " cutoff_date; \
	full_capacity_date=$${full_capacity_date:-$(FULL_CAPACITY_DATE)}; \
	cutoff_date=$${cutoff_date:-$(CUTOFF_DATE)}; \
	echo "Using full capacity date: $$full_capacity_date"; \
	echo "Using cutoff date: $$cutoff_date"; \
	make setup read_data preprocess FULL_CAPACITY_DATE=$$full_capacity_date; \
	make feature_engineering; \
	make split CUTOFF_DATE=$$cutoff_date; \
	make choose_model
