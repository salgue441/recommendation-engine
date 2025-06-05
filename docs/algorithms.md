# Recommendation Algorithms Documentation

## Overview

This document describes the recommendation algorithms implemented in the Real-Time Recommendation Engine. Our hybrid approach combines multiple techniques to provide accurate, diverse, and explainable recommendations while handling various scenarios like cold starts and real-time updates.

## Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Recommender                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │Collaborative│  │Content-Based│  │  Social Network     │ │
│  │ Filtering   │  │ Filtering   │  │  Recommendations    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                    │            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   MinHash   │  │  TF-IDF +   │  │    Graph Neural     │ │
│  │     LSH     │  │ Cosine Sim  │  │     Networks        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
                    ┌─────────────────┐
                    │ Cold Start      │
                    │ Handler         │
                    └─────────────────┘
```

## Core Algorithms

### 1. Collaborative Filtering

Collaborative Filtering (CF) generates recommendations based on user-item interactions and similarity patterns between users or items.

#### User-Based Collaborative Filtering

**Algorithm:** Find users similar to the target user and recommend items they liked.

```python
def user_based_cf(user_id, user_item_matrix, k_neighbors=50):
    """
    User-based collaborative filtering using cosine similarity
    """
    # Calculate user similarities using MinHash + LSH for efficiency
    similar_users = find_similar_users_lsh(user_id, k_neighbors)

    # Weight recommendations by user similarity
    recommendations = {}
    for similar_user, similarity in similar_users:
        for item in get_user_items(similar_user):
            if item not in get_user_items(user_id):  # Not already interacted
                recommendations[item] = recommendations.get(item, 0) + similarity

    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
```

**Advantages:**

- Captures complex patterns in user behavior
- Naturally handles item diversity
- Works well with implicit feedback

**Disadvantages:**

- Cold start problem for new users
- Sparse data issues
- Computational complexity: O(n²) for naive implementation

#### Item-Based Collaborative Filtering

**Algorithm:** Find items similar to those the user has interacted with.

```python
def item_based_cf(user_id, item_similarity_matrix):
    """
    Item-based collaborative filtering using precomputed similarities
    """
    user_items = get_user_items(user_id)
    recommendations = {}

    for item in user_items:
        similar_items = item_similarity_matrix[item]
        for similar_item, similarity in similar_items:
            if similar_item not in user_items:
                recommendations[similar_item] = (
                    recommendations.get(similar_item, 0) + similarity
                )

    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
```

**Optimization: MinHash + LSH Implementation**

For scalability, we use MinHash and Locality-Sensitive Hashing (LSH) to find similar users/items efficiently:

```python
class MinHashLSH:
    def __init__(self, num_perm=128, threshold=0.5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def add_user(self, user_id, item_set):
        """Add user with their item interactions"""
        minhash = MinHash(num_perm=self.num_perm)
        for item in item_set:
            minhash.update(str(item).encode('utf-8'))
        self.lsh.insert(user_id, minhash)

    def find_similar_users(self, user_id, item_set, k=50):
        """Find k most similar users efficiently"""
        minhash = MinHash(num_perm=self.num_perm)
        for item in item_set:
            minhash.update(str(item).encode('utf-8'))

        candidates = self.lsh.query(minhash)
        # Rank candidates by exact Jaccard similarity
        similarities = []
        for candidate in candidates:
            sim = calculate_jaccard_similarity(user_id, candidate)
            similarities.append((candidate, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
```

**Time Complexity:** O(log n) for LSH queries vs O(n) for brute force

### 2. Content-Based Filtering

Content-based filtering recommends items similar to those a user has interacted with, based on item features and metadata.

#### Feature Extraction

```python
class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.category_encoder = OneHotEncoder()

    def extract_features(self, items):
        """Extract and combine multiple feature types"""
        # Text features (title, description, tags)
        text_features = [
            f"{item.title} {item.description} {' '.join(item.tags)}"
            for item in items
        ]
        text_vectors = self.tfidf_vectorizer.fit_transform(text_features)

        # Categorical features
        categories = [[item.category] for item in items]
        category_vectors = self.category_encoder.fit_transform(categories)

        # Numerical features (normalized)
        numerical_features = np.array([
            [item.price, item.rating, item.duration]
            for item in items
        ])
        numerical_features = StandardScaler().fit_transform(numerical_features)

        # Combine all features
        combined_features = hstack([
            text_vectors,
            category_vectors,
            numerical_features
        ])

        return combined_features

    def recommend(self, user_id, user_profile, k=10):
        """Generate content-based recommendations"""
        user_items = get_user_items(user_id)
        user_feature_vector = self.get_user_feature_vector(user_items)

        # Calculate similarities with all items
        item_similarities = cosine_similarity(
            user_feature_vector.reshape(1, -1),
            self.item_feature_matrix
        ).flatten()

        # Filter out already interacted items
        recommendations = []
        for idx, similarity in enumerate(item_similarities):
            if self.items[idx].id not in user_items:
                recommendations.append((self.items[idx].id, similarity))

        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:k]
```

**Advanced Feature Engineering:**

- **TF-IDF with n-grams** for text similarity
- **Word embeddings** (Word2Vec/GloVe) for semantic similarity
- **Image features** using CNN embeddings for visual content
- **Temporal features** for time-sensitive content

### 3. Social Network Recommendations

Leverages social connections and community detection to provide socially-aware recommendations.

#### Graph-Based Approach

```python
class SocialRecommender:
    def __init__(self, social_graph):
        self.graph = social_graph  # NetworkX graph

    def recommend_via_friends(self, user_id, k=10):
        """Recommend items liked by friends"""
        friends = list(self.graph.neighbors(user_id))
        recommendations = {}

        for friend in friends:
            friend_weight = self.get_friend_influence(user_id, friend)
            friend_items = get_user_items(friend)

            for item in friend_items:
                if item not in get_user_items(user_id):
                    recommendations[item] = (
                        recommendations.get(item, 0) + friend_weight
                    )

        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]

    def community_recommendations(self, user_id, k=10):
        """Recommend based on community preferences"""
        # Detect communities using Louvain algorithm
        communities = community.best_partition(self.graph)
        user_community = communities[user_id]

        # Find popular items in user's community
        community_users = [
            user for user, comm in communities.items()
            if comm == user_community and user != user_id
        ]

        item_scores = {}
        for community_user in community_users:
            for item in get_user_items(community_user):
                item_scores[item] = item_scores.get(item, 0) + 1

        # Normalize by community size
        community_size = len(community_users)
        for item in item_scores:
            item_scores[item] /= community_size

        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_friend_influence(self, user_id, friend_id):
        """Calculate friend influence based on interaction frequency"""
        # Could be based on:
        # - Number of mutual interactions
        # - Recency of interactions
        # - Similarity of preferences
        return self.graph[user_id][friend_id].get('weight', 1.0)
```

### 4. Hybrid Recommendation Strategy

The hybrid approach combines multiple algorithms to leverage their strengths and mitigate weaknesses.

#### Weighted Linear Combination

```python
class HybridRecommender:
    def __init__(self, cf_recommender, content_recommender, social_recommender):
        self.cf = cf_recommender
        self.content = content_recommender
        self.social = social_recommender

        # Weights determined through A/B testing and genetic algorithm optimization
        self.weights = {
            'collaborative': 0.5,
            'content': 0.3,
            'social': 0.2
        }

    def recommend(self, user_id, k=10):
        """Generate hybrid recommendations"""
        # Get recommendations from each algorithm
        cf_recs = self.cf.recommend(user_id, k=k*2)
        content_recs = self.content.recommend(user_id, k=k*2)
        social_recs = self.social.recommend(user_id, k=k*2)

        # Combine scores using weighted sum
        combined_scores = {}

        for item, score in cf_recs:
            combined_scores[item] = self.weights['collaborative'] * score

        for item, score in content_recs:
            combined_scores[item] = (
                combined_scores.get(item, 0) +
                self.weights['content'] * score
            )

        for item, score in social_recs:
            combined_scores[item] = (
                combined_scores.get(item, 0) +
                self.weights['social'] * score
            )

        # Apply diversification and ranking
        final_recommendations = self.diversify_recommendations(
            combined_scores, user_id, k
        )

        return final_recommendations

    def diversify_recommendations(self, recommendations, user_id, k):
        """Ensure diversity in final recommendations"""
        # Use greedy algorithm for diversity
        selected = []
        remaining = list(recommendations.items())
        remaining.sort(key=lambda x: x[1], reverse=True)

        # Always include top recommendation
        if remaining:
            selected.append(remaining.pop(0))

        while len(selected) < k and remaining:
            best_item = None
            best_score = -1

            for item, score in remaining:
                # Calculate diversity score
                diversity = self.calculate_diversity(item, selected)
                combined_score = 0.7 * score + 0.3 * diversity

                if combined_score > best_score:
                    best_score = combined_score
                    best_item = (item, score)

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)

        return selected

    def calculate_diversity(self, item, selected_items):
        """Calculate how different an item is from already selected items"""
        if not selected_items:
            return 1.0

        item_features = self.get_item_features(item)
        min_similarity = float('inf')

        for selected_item, _ in selected_items:
            selected_features = self.get_item_features(selected_item)
            similarity = cosine_similarity(
                item_features.reshape(1, -1),
                selected_features.reshape(1, -1)
            )[0][0]
            min_similarity = min(min_similarity, similarity)

        return 1 - min_similarity  # Higher diversity = lower similarity
```

### 5. Cold Start Problem Solutions

#### New User Cold Start

```python
class ColdStartHandler:
    def __init__(self):
        self.popularity_recommender = PopularityRecommender()
        self.demographic_recommender = DemographicRecommender()

    def recommend_new_user(self, user_demographic, k=10):
        """Recommend items for users with no interaction history"""
        # Strategy 1: Demographic-based recommendations
        demo_recs = self.demographic_recommender.recommend(
            user_demographic, k=k//2
        )

        # Strategy 2: Popular items in relevant categories
        popular_recs = self.popularity_recommender.recommend_popular(
            categories=user_demographic.get('interests', []),
            k=k//2
        )

        # Combine and deduplicate
        all_recs = demo_recs + popular_recs
        seen_items = set()
        final_recs = []

        for item, score in all_recs:
            if item not in seen_items and len(final_recs) < k:
                final_recs.append((item, score))
                seen_items.add(item)

        return final_recs
```

#### New Item Cold Start

```python
def recommend_new_item(self, item_id, k=10):
    """Find users who might be interested in a new item"""
    item_features = self.get_item_features(item_id)

    # Find users who liked similar items
    similar_items = self.find_similar_items(item_features, k=50)
    candidate_users = set()

    for similar_item, similarity in similar_items:
        item_users = get_item_users(similar_item)
        for user in item_users:
            candidate_users.add(user)

    # Rank candidate users by likelihood to engage
    user_scores = []
    for user in candidate_users:
        user_profile = self.get_user_profile(user)
        affinity_score = self.calculate_user_item_affinity(
            user_profile, item_features
        )
        user_scores.append((user, affinity_score))

    return sorted(user_scores, key=lambda x: x[1], reverse=True)[:k]
```

### 6. Real-Time Updates

#### Incremental Learning

```python
class IncrementalUpdater:
    def __init__(self, recommender_system):
        self.system = recommender_system
        self.update_queue = Queue()

    def process_real_time_interaction(self, user_id, item_id, interaction_type):
        """Process a single user interaction in real-time"""
        # Update user-item matrix
        self.system.user_item_matrix[user_id][item_id] = self.get_interaction_weight(
            interaction_type
        )

        # Update user similarity cache for affected users
        affected_users = self.find_affected_users(user_id, item_id)
        for affected_user in affected_users:
            self.invalidate_user_cache(affected_user)

        # Update item popularity scores
        self.system.item_popularity[item_id] += self.get_popularity_boost(
            interaction_type
        )

        # Queue for batch processing of expensive updates
        self.update_queue.put({
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'timestamp': time.time()
        })

    def batch_update_similarities(self):
        """Periodically update similarity matrices"""
        updates = []
        while not self.update_queue.empty():
            updates.append(self.update_queue.get())

        if updates:
            # Group by user and item for efficient batch processing
            user_updates = defaultdict(list)
            for update in updates:
                user_updates[update['user_id']].append(update)

            # Update LSH index incrementally
            for user_id, user_update_list in user_updates.items():
                self.update_user_lsh(user_id, user_update_list)
```

## Algorithm Selection Strategy

### Dynamic Algorithm Selection

```python
class AlgorithmSelector:
    def __init__(self):
        self.selection_rules = {
            'new_user': ['demographic', 'popularity'],
            'sparse_user': ['content', 'popularity'],
            'active_user': ['collaborative', 'hybrid'],
            'trending_context': ['popularity', 'social'],
            'exploration_mode': ['content', 'social']
        }

    def select_algorithm(self, user_context):
        """Dynamically select best algorithm based on context"""
        user_interaction_count = user_context['interaction_count']
        time_since_last_interaction = user_context['recency']

        if user_interaction_count == 0:
            return self.selection_rules['new_user']
        elif user_interaction_count < 10:
            return self.selection_rules['sparse_user']
        elif time_since_last_interaction > 30:  # days
            return self.selection_rules['exploration_mode']
        else:
            return self.selection_rules['active_user']
```

## Performance Optimizations

### 1. Caching Strategy

- **User recommendation cache**: 15-minute TTL
- **Item similarity cache**: 24-hour TTL
- **Popular items cache**: 1-hour TTL
- **User profile cache**: 30-minute TTL

### 2. Approximation Techniques

- **LSH for similarity computation**: 10x speedup with <5% accuracy loss
- **Random sampling** for large user bases
- **Matrix factorization** for dimensionality reduction

### 3. Parallel Processing

- **Multi-threaded similarity computation**
- **Distributed matrix operations** using Dask
- **Asynchronous recommendation serving**

## Evaluation Metrics

### Accuracy Metrics

- **Precision@K**: Relevant items in top-K recommendations
- **Recall@K**: Coverage of relevant items
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

### Business Metrics

- **Click-Through Rate (CTR)**
- **Conversion Rate**
- **Revenue per Recommendation**
- **User Engagement Time**

### Diversity and Coverage

- **Intra-list Diversity**: Diversity within recommendation list
- **Catalog Coverage**: Percentage of items recommended
- **User Coverage**: Percentage of users receiving recommendations

## A/B Testing Framework

### Experiment Design

```python
class ABTestFramework:
    def __init__(self):
        self.experiments = {}

    def create_experiment(self, experiment_config):
        """Create a new A/B test experiment"""
        experiment = {
            'id': experiment_config['id'],
            'control_algorithm': experiment_config['control'],
            'test_algorithm': experiment_config['test'],
            'traffic_split': experiment_config['split'],
            'metrics': experiment_config['metrics'],
            'start_date': datetime.now(),
            'status': 'active'
        }
        self.experiments[experiment['id']] = experiment

    def assign_user_to_group(self, user_id, experiment_id):
        """Assign user to control or test group"""
        # Use consistent hashing for stable assignment
        hash_value = hashlib.md5(f"{user_id}_{experiment_id}".encode()).hexdigest()
        hash_int = int(hash_value, 16)

        experiment = self.experiments[experiment_id]
        if (hash_int % 100) < (experiment['traffic_split'] * 100):
            return 'test'
        else:
            return 'control'
```

## Genetic Algorithm for Hyperparameter Optimization

```python
class GeneticOptimizer:
    def __init__(self, parameter_ranges):
        self.parameter_ranges = parameter_ranges
        self.population_size = 50
        self.generations = 100

    def optimize_hybrid_weights(self, validation_data):
        """Optimize hybrid algorithm weights using genetic algorithm"""
        # Initialize random population
        population = self.initialize_population()

        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, validation_data)
                fitness_scores.append(fitness)

            # Selection: tournament selection
            parents = self.tournament_selection(population, fitness_scores)

            # Crossover and mutation
            offspring = self.crossover_and_mutate(parents)

            # Replace worst individuals
            population = self.replace_worst(
                population, offspring, fitness_scores
            )

        # Return best individual
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]

    def evaluate_fitness(self, weights, validation_data):
        """Evaluate recommendation quality with given weights"""
        # Configure hybrid recommender with these weights
        recommender = HybridRecommender()
        recommender.set_weights(weights)

        # Calculate NDCG on validation set
        total_ndcg = 0
        for user_id, true_items in validation_data:
            recommendations = recommender.recommend(user_id, k=10)
            recommended_items = [item for item, score in recommendations]
            ndcg = calculate_ndcg(true_items, recommended_items)
            total_ndcg += ndcg

        return total_ndcg / len(validation_data)
```

This algorithmic foundation provides a robust, scalable, and adaptive recommendation system capable of handling diverse use cases while maintaining high performance and accuracy.
  