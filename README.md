# House-Price-Prediction-System


This repository contains a Python-based machine learning project to predict house prices using a dataset of features such as location, size, and other property attributes. The system utilizes regression models to deliver accurate predictions.

## Features

- Preprocessing of data with feature scaling and one-hot encoding for categorical variables.
- Implementation of machine learning models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Model evaluation using metrics like Mean Squared Error (MSE) and R² Score.
- Designed to achieve high accuracy (R² score between 0.8 and 0.9).

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or later
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aryamandev14/House-Price-Prediction-System.git
   cd House-Price-Prediction-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Place your dataset file in the root directory of the project. The dataset should be a CSV file containing features and a target column (`Price`).

## Usage

1. Update the file path and target column name in the script (`main()` function):
   ```python
   file_path = 'your-dataset.csv'  # Replace with your dataset file path
   target_column = 'Price'  # Replace with the target column in your dataset
   ```

2. Run the script:
   ```bash
   python house_price_prediction.py
   ```

3. View model evaluation metrics in the console output.

## Project Structure

```
House-Price-Prediction-System/
├── house_price_prediction.py    # Main Python script for prediction
├── requirements.txt             # Dependencies for the project
├── README.md                    # Project documentation
└── your-dataset.csv             # Your dataset file (not included in the repo)
```

## Results

The project evaluates models using the test dataset and provides:

- Mean Squared Error (MSE)
- R² Score

The models are tuned to achieve an R² score of 0.8–0.9 for reliable predictions.

## Contributions

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

[Aryaman Dev Kumar](https://github.com/Aryamandev14)
