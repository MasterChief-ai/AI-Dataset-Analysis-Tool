AI Dataset Analysis Tool

An intelligent research assistant that automates dataset exploration and analysis. The tool classifies datasets into regression, classification, or clustering tasks, executes the appropriate workflow, and generates both structured results and narrative research insights.

🚀 Features

Automatic Task Classification – Detects whether the dataset corresponds to regression, classification, or clustering.

Task-Specific Analysis

Regression → builds predictive models and evaluates performance metrics.

Classification → trains and tests supervised learning models with accuracy reports.

Clustering → applies unsupervised methods to reveal hidden data structures.

Results & Visualization – Generates key performance metrics and charts to support findings.

AI-Powered Research Insights – Integrates with Gemini API to produce contextual, publication-ready interpretations of results (feature importance, correlations, trends).

🛠️ Tech Stack

Programming Language: Python

Libraries: scikit-learn, NumPy, Pandas, Matplotlib/Seaborn

LLM Integration: Gemini API (for narrative insights)

📂 Project Structure
AI-Dataset-Analyzer/
│
├── data/                # Sample datasets
├── notebooks/           # Jupyter notebooks for testing
├── src/                 # Core Python scripts
│   ├── classifier.py    # Task classification logic
│   ├── analyzer.py      # Regression, classification, clustering modules
│   ├── visualizer.py    # Visualization and metrics
│   └── gemini_api.py    # Gemini integration for research insights
│
├── README.md            # Project documentation
└── requirements.txt     # Dependencies

📖 Usage

Clone the repository:

git clone https://github.com/MasterChief-ai/AI-Dataset-Analyzer.git
cd AI-Dataset-Analyzer


Install dependencies:

pip install -r requirements.txt


Run the analyzer:

python src/analyzer.py --input data/your_dataset.csv

📊 Example Output

Task automatically identified as Regression.

Regression model trained → R² score: 0.87.

Feature importance ranked, top driver: Interest Rates.

Gemini API interpretation: “Interest rates account for 42% of the variance in the target variable, suggesting strong macroeconomic influence on outcomes.”

🎯 Applications

Academic research (data-driven papers, experiments)

Finance & macroeconomic analysis

Market prediction and business intelligence

Automated exploratory data analysis

📜 License

This project is released under the Apache 2.0 License.
