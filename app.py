import os
import re
import uuid
import pandas as pd
import numpy as np
import pdfplumber
from flask import Flask, request, render_template, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, 
                           silhouette_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from dbase import log_user_input
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_AI_STUDIO_GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')



app = Flask(__name__)
os.makedirs("Users/user/Documents/Nig Project/static/reports", exist_ok=True)

def determine_task_type(df, target_col):
    if target_col is None:
        return 'clustering'
    elif df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
        return 'classification'
    elif np.issubdtype(df[target_col].dtype, np.number):
        return 'regression'
    else:
        return 'clustering'

def generate_classification_charts(y_test, y_pred, report_id):
    filename = f"{report_id}_conf_matrix.png"
    full_path = f"Users/user/Documents/Nig Project/static/reports/{filename}"
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(full_path)
    plt.close()
    
    return f"reports/{filename}"

def generate_regression_charts(y_test, y_pred, report_id):
    filename = f"{report_id}_regression.png"
    full_path = f"Users/user/Documents/Nig Project/static/reports/{filename}"
    
    plt.figure(figsize=(8,6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3})
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(full_path)
    plt.close()
    
    return f"reports/{filename}"

def generate_clustering_charts(X, labels, report_id):
    filename = f"{report_id}_clusters.png"
    full_path = f"Users/user/Documents/Nig Project/static/reports/{filename}"
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=labels, palette='viridis', s=100)
    plt.title("Cluster Visualization (First Two Principal Components)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(full_path)
    plt.close()
    
    return f"reports/{filename}"

def generate_insights(df, target_col):
    insights = []
    if target_col and target_col in df.columns:
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numerical_features:
            numerical_features.remove(target_col)

        corr = df[numerical_features + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)
        if not corr.empty:
            top_features = corr.head(3)
            for feature, value in top_features.items():
                direction = "increase" if value > 0 else "decrease"
                insights.append(f"To increase {target_col}, consider to {direction} '{feature}' (correlation: {value:.2f}).")
            bottom_feature = corr.index[-1]
            insights.append(f"The least correlated feature with {target_col} is '{bottom_feature}' with a correlation of {corr[bottom_feature]:.2f}.")

        time_col = next((col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df['year'] = df[time_col].dt.year
            if 'year' in df:
                yearly_avg = df.groupby('year')[target_col].mean().dropna()
                if len(yearly_avg) > 1:
                    trend = yearly_avg.diff().mean()
                    direction = 'increased' if trend > 0 else 'decreased'
                    insights.append(f"On average, {target_col} has {direction} by {abs(trend):.2f} units per year.")
    else:
        insights.append("No target column specified, so trends and correlations cannot be analyzed.")

    return "\n".join(insights)

def strip_markdown(text):
    text = re.sub(
        r"(?i)(okay|sure|here's|this is).*?target column.*?\n+", 
        '', 
        text.strip()
    )
    # Replace bullet-style markdown with clean dashes
    text = re.sub(r'^\s*[\*\-]\s*', '- ', text, flags=re.MULTILINE)

    # Remove bold/italic/headers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Force each sentence or point to begin on a new line
    text = re.sub(r'(?<=[.!?])\s+(?=[A-Z\-])', '\n\n', text)

    # Add headers before sections
    text = re.sub(r"(1\.\s+Interpret why.+?)(\n+|\Z)", r"\n\nCorrelation Insights:\n", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"(2\.\s+Suggest.+?)(\n+|\Z)", r"\n\nActionable Suggestions:\n", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove leading/trailing whitespace and reduce multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Gemini Insight Generator
def generate_gemini_insight(corr_df, target_column, df):
    top_corr = corr_df[target_column].drop(target_column).abs().sort_values(ascending=False).head(3)
    related_cols = top_corr.index.tolist()
    related_vals = [corr_df[target_column][col] for col in related_cols]

    data_desc = df.describe().T.to_string()
    insight_prompt = f"""
You are an expert AI Analyst for an advanced AI dataset analysis tool.

The current analysis involves the target column: '{target_column}'.

The three most correlated variables to this target are:
{[(col, round(val, 3)) for col, val in zip(related_cols, related_vals)]}

Dataset statistics:
{data_desc}

Your task is to generate a clear, well arranged(possibly bullet the main points) human-readable insight report focusing on the **target column**.

Please:
1. Interpret why these variables may be highly correlated with the target.
2. Suggest 3 actionable changes that could improve the target column by adjusting the related variables.
3. Ensure the explanation is data-aware and written for a non-technical audience.

"""

    try:
        gemini_response = model.generate_content(insight_prompt)
        clean_text = strip_markdown(gemini_response.text)
        return clean_text
    except Exception as e:
        return f"Gemini Insight Error: {e}"

def generate_classification_explanation(y_test, y_pred, accuracy, target_col):
    explanation = []
    
    # Basic accuracy interpretation
    if accuracy > 0.9:
        explanation.append(f"The model shows excellent performance with an accuracy of {accuracy:.2f}.")
    elif accuracy > 0.7:
        explanation.append(f"The model shows good performance with an accuracy of {accuracy:.2f}.")
    elif accuracy > 0.5:
        explanation.append(f"The model shows moderate performance with an accuracy of {accuracy:.2f}.")
    else:
        explanation.append(f"The model's performance is below expectations with an accuracy of {accuracy:.2f}.")
    
    # Class distribution analysis
    class_dist = pd.Series(y_test).value_counts(normalize=True)
    explanation.append(f"\nClass distribution in test data:")
    for cls, prop in class_dist.items():
        explanation.append(f"- Class '{cls}': {prop:.1%}")
    
    # Confusion matrix insights
    cm = confusion_matrix(y_test, y_pred)
    if len(cm) == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        explanation.append("\nConfusion matrix details:")
        explanation.append(f"- True Positives: {tp}")
        explanation.append(f"- True Negatives: {tn}")
        explanation.append(f"- False Positives: {fp} (Type I errors)")
        explanation.append(f"- False Negatives: {fn} (Type II errors)")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        explanation.append(f"\nPrecision: {precision:.2f} (When predicted positive, how often correct)")
        explanation.append(f"Recall: {recall:.2f} (What proportion of actual positives was identified correctly)")
    
    return "\n".join(explanation)

def generate_clustering_explanation(labels, silhouette_score):
    explanation = []
    
    # Silhouette score interpretation
    if silhouette_score > 0.7:
        explanation.append(f"Excellent clustering structure (Silhouette Score: {silhouette_score:.2f}).")
    elif silhouette_score > 0.5:
        explanation.append(f"Reasonable clustering structure (Silhouette Score: {silhouette_score:.2f}).")
    elif silhouette_score > 0.25:
        explanation.append(f"Weak clustering structure (Silhouette Score: {silhouette_score:.2f}).")
    else:
        explanation.append(f"No substantial clustering structure (Silhouette Score: {silhouette_score:.2f}).")
    
    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    explanation.append("\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        explanation.append(f"- Cluster {cluster}: {count} samples ({count/len(labels):.1%})")
    
    # General advice
    explanation.append("\nRecommendations:")
    explanation.append("- Consider visualizing different feature combinations if clusters aren't well-separated")
    explanation.append("- Try different numbers of clusters if the silhouette score is low")
    explanation.append("- Examine cluster centers to understand what defines each group")
    
    return "\n".join(explanation)

def analyze_data(df, target_col=None):
    report_id = str(uuid.uuid4())[:8]
    task = determine_task_type(df, target_col)
    best_model = ""
    metrics = {}
    explanation = ""
    ai_insight = ""
    chart_path = ""
    corr_matrix = df.corr(numeric_only=True) 
    

    if task == 'clustering':
        X = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception:
            score = -1 # Handle cases where silhouette score cannot be computed
        chart_path = generate_clustering_charts(pd.DataFrame(X_scaled), labels, report_id)
        best_model = "KMeans"
        metrics = {"Silhouette Score": round(score, 4)}
        explanation = generate_clustering_explanation(labels, score)
        if target_col:
            ai_insight = generate_gemini_insight(corr_matrix, target_col, df)
        else:
            ai_insight = "No target column specified for AI insight generation." 
        
        

    else:
        X = df.drop(columns=[target_col])
        X = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if task == 'classification':
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForestClassifier": RandomForestClassifier()
            }
            scores = {}
            predictions = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                scores[name] = acc
                predictions[name] = y_pred
            best_model = max(scores, key=scores.get)
            chart_path = generate_classification_charts(y_test, predictions[best_model], report_id)
            metrics = {"Accuracy": scores[best_model]}
            explanation = generate_classification_explanation(y_test, predictions[best_model], scores[best_model], target_col)
            ai_insight = generate_gemini_insight(corr_matrix, target_col, df)
            
        elif task == 'regression':
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor()
            }
            scores = {}
            predictions = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                scores[name] = r2
                predictions[name] = y_pred
            best_model = max(scores, key=scores.get)
            chart_path = generate_regression_charts(y_test, predictions[best_model], report_id)
            mse = mean_squared_error(y_test, predictions[best_model])
            metrics = {"R2 Score": scores[best_model], "MSE": mse}
            ai_insight = generate_gemini_insight(corr_matrix, target_col, df)

            
            # Regression explanation
            target_mean = df[target_col].mean()
            target_std = df[target_col].std()
            if r2 > 0.8:
                explanation = f"The model fits the data very well (R2 Score = {scores[best_model]:.2f}). Lower MSE of {mse:.2f} indicates high accuracy in predictions. The mean of the target variable '{target_col}' is {target_mean:.2f}, with a standard deviation of {target_std:.2f}."
            elif r2 > 0.5:
                explanation = f"The model shows a moderate fit to the data (R2 Score = {scores[best_model]:.2f}). MSE of {mse:.2f} suggests some deviation in predictions. The mean of the target variable '{target_col}' is {target_mean:.2f}, with a standard deviation of {target_std:.2f}."
            else:
                explanation = f"The model may not be a good fit (R2 Score = {scores[best_model]:.2f}). High MSE of {mse:.2f} indicates large prediction errors. The mean of the target variable '{target_col}' is {target_mean:.2f}, with a standard deviation of {target_std:.2f}."
    

    # Generate insights
    insight_text = generate_insights(df, target_col)

    log_user_input(
        report_id=report_id,
        target_col=target_col,
        task_type=task,
        best_model=best_model,
        metrics=metrics,
        insights=insight_text,
        explanation=explanation,
        ai_insight=ai_insight
    )


    return {
        "task_type": task,
        "best_model": best_model,
        "metrics": metrics,
        "insights": insight_text,
        "chart_path": f"/static/{chart_path}",
        "explanation": explanation,
        "ai_insight": ai_insight
    }

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/extract_columns', methods=['POST'])
def extract_columns():
    file = request.files['dataset']
    filename = file.filename.lower()

    if filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    elif filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            first_page = pdf.pages[0]
            table = first_page.extract_table()
        if not table:
            return jsonify({"error": "No readable table found in the PDF."}), 400
        df = pd.DataFrame(table[1:], columns=table[0])
    else:
        return jsonify({"error": "Unsupported file format."}), 400

    return jsonify({"columns": list(df.columns)})

@app.route('/analyze', methods=['POST'])
def upload_and_analyze():
    file = request.files['dataset']
    target_col = request.form.get('target_column')
    filename = file.filename.lower()

    if filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    elif filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            first_page = pdf.pages[0]
            table = first_page.extract_table()
        if not table:
            return jsonify({"error": "No readable table found in the PDF."}), 400
        df = pd.DataFrame(table[1:], columns=table[0])
    else:
        return jsonify({"error": "Unsupported file format."}), 400

    results = analyze_data(df, target_col if target_col != 'None' else None)
    return render_template("report_template.html",
                           task_type=results["task_type"],
                           best_model=results["best_model"],
                           metrics=results["metrics"],
                           insights=results["insights"],
                           chart_path=results["chart_path"],
                           explanation=results["explanation"],
                           ai_insight=results["ai_insight"]
                           )
def send_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True)
