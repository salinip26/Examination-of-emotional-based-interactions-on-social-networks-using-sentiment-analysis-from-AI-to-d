import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
from textblob import TextBlob

# Step 1: preprocess dataset
file_path = '/Users/deeks/Desktop/goemotions_1.csv'  # Replace with the actual path
raw_data = pd.read_csv(file_path)

# Create a proxy for length of the text 
raw_data['text_length'] = raw_data['text'].apply(len)  # Engagement = length of the text

# Map emotions to sentiment scores
emotion_to_sentiment_mapping = {
    'admiration': 1, 'amusement': 1, 'approval': 1, 'caring': 1, 'excitement': 1,
    'gratitude': 1, 'joy': 1, 'love': 1, 'optimism': 1, 'pride': 1, 'relief': 1,
    'anger': -1, 'annoyance': -1, 'disappointment': -1, 'disapproval': -1, 
    'disgust': -1, 'embarrassment': -1, 'fear': -1, 'grief': -1, 
    'nervousness': -1, 'remorse': -1, 'sadness': -1,
    'confusion': 0, 'curiosity': 0, 'realization': 0, 'surprise': 0, 'neutral': 0
}

def compute_sentiment(row):
    for emotion, sentiment in emotion_to_sentiment_mapping.items():
        if row[emotion] == 1:  
            return sentiment
    return 0  
raw_data['sentiment_score'] = raw_data.apply(compute_sentiment, axis=1)

# Text sentiment preprocessing
raw_data['text_sentiment'] = raw_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Step 2: Create a network graph
interaction_graph = nx.DiGraph()

for _, row in raw_data.iterrows():
    interaction_graph.add_edge(
        row['parent_id'],  # Parent as source node
        row['author'],     # Author as target node
        sentiment=row['sentiment_score'],
        text_sentiment=row['text_sentiment'],
        text_length=row['text_length']
    )

# Step 3: Create a Subgraph
# Using only a subset of the graph for faster computation because the dataset is big
subgraph_nodes = list(interaction_graph.nodes())[:500]  
interaction_subgraph = interaction_graph.subgraph(subgraph_nodes)

# Step 4: compute centrality measures for the subgraph
degree_centrality = nx.degree_centrality(interaction_subgraph)
closeness_centrality = nx.closeness_centrality(interaction_subgraph)

# Step 5: Extract features for predictive modeling
feature_list = []
for source, target, attrs in interaction_subgraph.edges(data=True):
    feature_list.append({
        'source': source,
        'target': target,
        'sentiment': attrs.get('sentiment', 0),
        'text_sentiment': attrs.get('text_sentiment', 0),
        'text_length': attrs.get('text_length', 0),
        'degree_centrality_source': degree_centrality.get(source, 0),
        'degree_centrality_target': degree_centrality.get(target, 0),
        'closeness_centrality_source': closeness_centrality.get(source, 0),
        'closeness_centrality_target': closeness_centrality.get(target, 0)
    })

# Convert to DataFrame
feature_df = pd.DataFrame(feature_list)

# Step 6: Train-Test Split
X = feature_df.drop(columns=['text_length'])
y = feature_df['text_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#numeric features
categorical_columns = X_train.select_dtypes(include=['object']).columns
X_train = pd.get_dummies(X_train, columns=categorical_columns)
X_test = pd.get_dummies(X_test, columns=categorical_columns)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Step 7: Train the  Model
print("Starting model training...")
start_time = time.time()

# Train RandomForest with parameters
regressor_model = RandomForestRegressor(
    n_estimators=50,  # Fewer trees for faster training
    max_depth=10,     # Limit tree depth
    n_jobs=-1,        # Utilize all CPU cores
    random_state=42
)
regressor_model.fit(X_train, y_train)

end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds")

# Step 8: Evaluate the Model
print("Evaluating the model...")

#testing for 1000 samples for faster computation
X_test = X_test[:1000]  
y_test = y_test[:1000]

# Make predictions
start_eval = time.time()
y_pred = regressor_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

end_eval = time.time()
print(f"Model evaluation completed in {end_eval - start_eval:.2f} seconds")

# Subgraph with fewer nodes and edges
subgraph_nodes = list(interaction_graph.nodes())[:500]  # Limit to 500 nodes
interaction_subgraph = interaction_graph.subgraph(subgraph_nodes)

# Generate positions using fewer iterations
pos = nx.spring_layout(interaction_subgraph, iterations=10)


# Plot with labels
plt.figure(figsize=(10, 10))
node_sizes = [len(list(interaction_subgraph.neighbors(n))) * 100 for n in interaction_subgraph.nodes()]
nx.draw(
    interaction_subgraph,
    pos,
    with_labels=True,  # Enable labels
    node_size=node_sizes,
    node_color="lightblue",
    edge_color="gray",
    alpha=0.7,
    font_size=8  # Adjust font size for better readability
)
plt.title("Optimized User Interaction Network with Labels")
plt.show()
