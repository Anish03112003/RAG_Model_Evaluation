README: Execution Guide for Answer Quality Evaluation Framework

This guide provides step-by-step instructions to set up and run the Answer Quality Evaluation Framework in Google Colab. The framework evaluates model-generated answers based on token-level and semantic similarities with ground truth answers.

Prerequisites
Google Colab Account: Ensure you have a Google account and are signed into Google Colab.

Step 1: Set Up Dependencies
Run the following commands to install necessary packages for NLP and machine learning.
# Install required libraries
!pip install spacy transformers torch scikit-learn nltk matplotlib
# Download necessary language models and data
!python -m spacy download en_core_web_sm
import nltk
nltk.download('punkt')
nltk.download('stopwords')

Step 2: Prepare Input Data
Create text files named questionEX1.txt, answerEX1.txt, contextEX1.txt, and groundtruthEX1.txt (and similarly for EX2, EX3, EX4, EX5) in the Colab environment. Each file should contain the text data for each example.
question: Contains the original question.
answer: Contains the model-generated answer.
context (optional): Provides additional context (e.g., passage text).
ground_truth: Contains the reference answer for evaluation.
To upload files in Colab, go to the Files tab on the left, then click Upload.

Step 3: Initialize Classes
Define the classes for data handling, semantic scoring, hybrid scoring, evaluation, and plotting.

Step 4: Execute the Main Evaluation
Run the following cell to execute the evaluation. This will load each example, compute the scores, and print the results.
# Main execution
results = []
for ex in ['EX1', 'EX2', 'EX3', 'EX4', 'EX5']:
    data_handler = DataHandler(ex)
    semantic_scorer = SemanticScorer()
    hybrid_scorer = HybridScorer()
    evaluator = Evaluator(data_handler, semantic_scorer, hybrid_scorer)
    final_score_hybrid, final_score_semantic, final_score_combined = evaluator.evaluate()
    results.append((ex, final_score_hybrid, final_score_semantic, final_score_combined))
    print(f"{ex} - Final Token Score: {round(final_score_hybrid, 4) * 100}, Final Semantic Score: {round(final_score_semantic, 4) * 100}, Final Combined Score: {round(final_score_combined, 4) * 100}")

Step 5: Visualize Results
Plot the scores to visualize performance across different examples by running the code below.
# Plot results
Plotter.plot_results(results)

Step 6: Interpret Output
Printed Scores: After running the evaluation code, you’ll see token, semantic, and combined scores for each example.
Plot: The plot will show a graphical comparison of token scores, semantic scores, and combined scores across examples.

Notes
BERT Model: This evaluation uses a BERT-based approach for calculating semantic similarity, which may take additional time for computation in Colab.
Custom Adjustments: You can adjust the SemanticScorer and HybridScorer classes for other scoring methods as needed.
By following these steps, you should be able to set up, run, and evaluate the answer quality using the provided code in Google Colab.
