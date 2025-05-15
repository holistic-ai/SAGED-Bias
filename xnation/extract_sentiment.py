import pandas as pd
from saged import FeatureExtractor

def extract_sentiment():
    # Load the CSV file
    file_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\pseudo_generations\pseudo_routed_responses_20250515_144221.csv"
    df = pd.read_csv(file_path)
    
    # Create FeatureExtractor instance
    extractor = FeatureExtractor(
        benchmark=df,
        generations=['pseudo_routed_responses'],  # Using actual column names from the CSV
        calibration=True,
        baseline='baseline'
    )
    
    # Apply sentiment classification
    df_with_sentiment = extractor.sentiment_classification()
    
    # Save the results
    output_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\pseudo_generations\routed_pre_generated_responses_with_sentiment_20250515_144221.csv"
    df_with_sentiment.to_csv(output_path, index=False)
    
    print(f"Sentiment analysis complete. Results saved to: {output_path}")
    return df_with_sentiment

if __name__ == "__main__":
    extract_sentiment()
