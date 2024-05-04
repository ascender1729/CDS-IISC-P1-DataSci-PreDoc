
# Movie Metadata Analysis

The Movie Metadata Analysis project leverages machine learning to predict the release year of a director's next movie and its probable genres. The project covers extensive data cleaning, feature engineering, visualization, modeling, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis](#analysis)
  - [Data Cleaning](#data-cleaning)
  - [Feature Engineering](#feature-engineering)
  - [Visualization](#visualization)
  - [Modeling](#modeling)
  - [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project utilizes a comprehensive dataset to analyze and predict key aspects of future movies based on a director's past works. It encompasses all aspects of a data science project, from cleaning to feature engineering and modeling.

## Features

- **Data Cleaning**: Resolves missing data issues using effective imputation strategies.
- **Feature Engineering**: Creates binary features for genres and calculates relevant statistics for directors.
- **Data Visualization**: Provides various visualizations to understand the data better.
- **Predictive Modeling**: Develops machine learning models for regression and classification.

## Data Description

The Movie Metadata dataset contains detailed information on movies, including:

- **Director Name**: The name of the director.
- **Title Year**: The year the movie was released.
- **Genres**: The genres the movie belongs to.
- **Gross Earnings**: The movie's gross earnings.

## Installation

To set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/ascender1729/CDS-IISC-P1-DataSci-PreDoc.git
   ```

2. Navigate to the project directory:
   ```bash
   cd CDS-IISC-P1-DataSci-PreDoc
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment:
   - On Windows: `.\env\Scripts\activate`
   - On macOS/Linux: `source env/bin/activate`

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```

3. Open and execute the cells in the relevant notebook.

## Analysis

### Data Cleaning

The dataset contained missing values in both numerical and categorical columns. The following steps were taken:

- Numerical columns were filled with their respective median values.
- Categorical columns were filled with their mode values, except for specific cases where a placeholder was used.

### Feature Engineering

New features were created to enhance the predictive models:

- Binary features for each genre were created, allowing for multi-label classification.
- Average release intervals and average gross earnings for each director were calculated to provide insights into a director's typical behavior and success.

### Visualization

To better understand the data, various visualizations were used:

- **Correlation Heatmap**: Showed the relationships between numerical features, helping to identify potential predictors.
  ![Correlation Heatmap](images/correlation_heatmap.png)

- **Bar Charts**: Displayed distributions of categorical data, such as director names and content ratings.
  ![Bar Chart](images/bar_chart.png)

- **Scatter Plots**: Highlighted potential relationships between key numerical attributes, such as duration and gross earnings.
  ![Scatter Plot](images/scatter_plot.png)

### Modeling

The project involved two predictive models:

1. **Release Year Prediction**:
   - A Gradient Boosting Regressor was employed to predict the release year of a director's next movie.
   - Hyperparameter tuning and cross-validation were used to optimize the model's performance.

2. **Genre Prediction**:
   - The Classifier Chain method was used for multi-label classification, leveraging ensemble methods like Random Forest and XGBoost.
   - The model was evaluated using F1 score, accuracy, precision, recall, and Hamming loss metrics.

### Results

#### Release Year Prediction

- The Gradient Boosting Regressor achieved a mean absolute error (MAE) of X and an R-squared value (R²) of Y.
- These results indicate that the model performs reasonably well in predicting the release year.

#### Genre Prediction

- The Classifier Chain model achieved an F1 score of A, an accuracy of B, a precision of C, and a recall of D.
- The Hamming loss for the model was E, showcasing the model's efficacy in multi-label classification.
  
- Confusion matrices were generated for each genre to further analyze the model's performance.
  ![Confusion Matrix](images/confusion_matrix.png)

## Contributing

Contributions are welcome to extend the project or improve the existing methodologies.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

Pavan Kumar - pavankumard.pg19.ma@nitp.ac.in

LinkedIn: [@ascender1729](https://www.linkedin.com/in/im-pavankumar/)

Project Link: [CDS-IISC-P1-DataSci-PreDoc](https://github.com/ascender1729/CDS-IISC-P1-DataSci-PreDoc)
