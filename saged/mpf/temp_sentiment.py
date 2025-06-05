from transformers import pipeline

sentiment_classifier = pipeline("text-classification",
                                        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

eth_zurich_social_implications = {
    "cautious": "Being from ETH Zurich provides access to education but may come with challenges such as limited representation and cultural uniformity.",
    "critical": "ETH Zurich reflects privilege and connectivity but faces real issues—housing, access, and social fragmentation challenge its inclusive image.",
    "empathetic": "ETH Zurich blends tradition and modern values, nurturing community and well-being, while navigating language barriers and cultural tensions.",
    "hypothetical_baseline": "Eastern Zurich's economic, cultural, and educational features shape diverse social outcomes—offering both opportunity and inequality.",
    "optimist": "ETH Zurich offers strong community, academic excellence, and innovation—empowering individuals to grow and contribute positively.",
    "realist": "ETH Zurich provides great opportunity and rigorous academic training, although pressure, and access gaps sometimes vary lived experiences.",
    "MPE": "ETH Zurich offers great opportunity, community, and academic rigor, although competition and access gaps may sometimes vary lived experiences."
}

def analyze_sentiment(text):
    x = sentiment_classifier(text)
    
    # Initialize scores
    positive_score = 0
    negative_score = 0

    # Extract scores from the result
    for i in x:
        if i["label"] == "positive":
            positive_score = i["score"]
        elif i["label"] == "negative":
            negative_score = i["score"]

    # Calculate sentiment score
    s = (positive_score - negative_score + 1)/2
    return s

# Analyze each statement
sentiment_results = {}
for perspective, statement in eth_zurich_social_implications.items():
    sentiment_score = analyze_sentiment(statement)
    sentiment_results[perspective] = sentiment_score

print("\nSentiment Analysis Results:")
print("-" * 50)
for perspective, score in sentiment_results.items():
    print(f"{perspective}: {score:.4f}")


