import os
import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score

def train_and_evaluate_svm(encodings_dir, results_path):
    # List all model directories
    model_dirs = [d for d in os.listdir(encodings_dir) if os.path.isdir(os.path.join(encodings_dir, d))]

    # DataFrame to store the results
    results_df = pd.DataFrame(columns=[
        'Model', 'Mean Accuracy', 'Std Accuracy', 'Top-1%', 'Top-3%', 'Top-5%',
        'Precision', 'Recall', 'F1 Score', 'Time Taken (s)'
    ])

    # Loop through each model directory, load features and labels, and perform classification
    for model_dir in model_dirs:
        print(f"Processing model: {model_dir}")
        
        # Define the paths for features and labels
        features_path = os.path.join(encodings_dir, model_dir, 'features.npy')
        labels_path = os.path.join(encodings_dir, model_dir, 'labels.npy')
        
        # Load the features and labels from the .npy files
        features = np.load(features_path)
        labels = np.load(labels_path)

        # Normalize the feature vectors (SVM benefits from normalized data)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Set up repeated stratified cross-validation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

        # Initialize lists to store performance metrics
        accuracy_list, top1_list, top3_list, top5_list = [], [], [], []
        precision_list, recall_list, f1_list = [], [], []
        
        start_time = time.time()

        # Perform cross-validation manually to collect additional metrics
        for train_idx, test_idx in cv.split(features, labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Define the SVM classifier
            svm = SVC(kernel='linear', probability=True)  # 'probability=True' to enable probability predictions

            # Train the SVM model
            svm.fit(X_train, y_train)
            
            # Make predictions
            y_pred = svm.predict(X_test)
            y_prob = svm.predict_proba(X_test)  # Probability predictions for top-k accuracy

            # Calculate metrics
            accuracy_list.append(accuracy_score(y_test, y_pred))
            top1_list.append(top_k_accuracy_score(y_test, y_prob, k=1, labels=np.unique(labels)))
            top3_list.append(top_k_accuracy_score(y_test, y_prob, k=3, labels=np.unique(labels)))
            top5_list.append(top_k_accuracy_score(y_test, y_prob, k=5, labels=np.unique(labels)))
            precision_list.append(precision_score(y_test, y_pred, average='weighted'))
            recall_list.append(recall_score(y_test, y_pred, average='weighted'))
            f1_list.append(f1_score(y_test, y_pred, average='weighted'))

        end_time = time.time()
        time_taken = end_time - start_time

        # Calculate mean and standard deviation of the metrics
        mean_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        top1_accuracy = np.mean(top1_list)
        top3_accuracy = np.mean(top3_list)
        top5_accuracy = np.mean(top5_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_f1 = np.mean(f1_list)

        # Add the results to the DataFrame
        results_df = results_df.append({
            'Model': model_dir,
            'Mean Accuracy': mean_accuracy,
            'Std Accuracy': std_accuracy,
            'Top-1%': top1_accuracy,
            'Top-3%': top3_accuracy,
            'Top-5%': top5_accuracy,
            'Precision': mean_precision,
            'Recall': mean_recall,
            'F1 Score': mean_f1,
            'Time Taken (s)': time_taken
        }, ignore_index=True)

        print(f"Model {model_dir}: Mean Accuracy = {mean_accuracy:.4f}, Top-1% = {top1_accuracy:.4f}, Time = {time_taken:.2f} s")

    # Save the results to an Excel file
    excel_path = os.path.join(results_path, 'svm_classification_results.xlsx')
    results_df.to_excel(excel_path, index=False)

    print(f"Results saved to {excel_path}")

# Example usage
if __name__ == "__main__":
    encodings_dir = "encodings1"  # Change this to the path to your actual encodings directory
    results_path = "results"  # Change this to the path where you want to save the results
    train_and_evaluate_svm(encodings_dir, results_path)
