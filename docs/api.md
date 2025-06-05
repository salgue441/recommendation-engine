# Real-Time Recommendation Engine API Documentation

## Overview

The Recommendation Engine API provides endpoints for generating personalized recommendations, managing user profiles, tracking interactions, and analyzing recommendation performance. All endpoints return JSON responses and follow RESTful conventions.

**Base URL:** `http://localhost:8000/api/v1`

**Authentication:** Bearer token (JWT) required for most endpoints

## Quick Start

```bash
# Get recommendations for a user
curl -X GET "http://localhost:8000/api/v1/recommendations/user/12345" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Track a user interaction
curl -X POST "http://localhost:8000/api/v1/interactions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"user_id": 12345, "item_id": 67890, "interaction_type": "view"}'
```

## Endpoints

### Recommendations

#### GET /recommendations/user/{user_id}

Get personalized recommendations for a user.

**Parameters:**

- `user_id` (path): User identifier
- `limit` (query, optional): Number of recommendations (default: 10, max: 100)
- `algorithm` (query, optional): Specific algorithm to use (`collaborative`, `content`, `hybrid`, `social`)
- `explain` (query, optional): Include explanation for recommendations (default: false)

**Response:**

```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "item_id": 67890,
      "score": 0.95,
      "rank": 1,
      "explanation": {
        "reason": "Users with similar preferences also liked this",
        "algorithm": "collaborative_filtering",
        "confidence": 0.89
      }
    }
  ],
  "algorithm_used": "hybrid",
  "generated_at": "2025-06-05T10:30:00Z",
  "cache_hit": false
}
```

#### GET /recommendations/item/{item_id}/similar

Get items similar to a specific item.

**Parameters:**

- `item_id` (path): Item identifier
- `limit` (query, optional): Number of similar items (default: 10)
- `method` (query, optional): Similarity method (`content`, `collaborative`, `hybrid`)

**Response:**

```json
{
  "item_id": 67890,
  "similar_items": [
    {
      "item_id": 11111,
      "similarity_score": 0.87,
      "similarity_type": "content_based"
    }
  ],
  "method_used": "hybrid"
}
```

#### POST /recommendations/batch

Get recommendations for multiple users in a single request.

**Request Body:**

```json
{
  "user_ids": [12345, 12346, 12347],
  "limit": 5,
  "algorithm": "hybrid"
}
```

**Response:**

```json
{
  "recommendations": {
    "12345": [...],
    "12346": [...],
    "12347": [...]
  },
  "processing_time_ms": 245
}
```

### User Management

#### GET /users/{user_id}

Get user profile information.

**Response:**

```json
{
  "user_id": 12345,
  "created_at": "2025-01-01T00:00:00Z",
  "last_active": "2025-06-05T09:45:00Z",
  "preferences": {
    "categories": ["technology", "science"],
    "tags": ["machine-learning", "data-science"]
  },
  "interaction_count": 1250,
  "recommendation_stats": {
    "total_served": 5000,
    "click_through_rate": 0.12,
    "average_rating": 4.2
  }
}
```

#### POST /users

Create a new user profile.

**Request Body:**

```json
{
  "user_id": 12345,
  "preferences": {
    "categories": ["technology"],
    "demographic": {
      "age_group": "25-34",
      "location": "US"
    }
  }
}
```

#### PUT /users/{user_id}/preferences

Update user preferences.

**Request Body:**

```json
{
  "categories": ["technology", "science"],
  "tags": ["ai", "machine-learning"],
  "explicit_feedback": {
    "liked_items": [111, 222],
    "disliked_items": [333]
  }
}
```

### Item Management

#### GET /items/{item_id}

Get item details and metadata.

**Response:**

```json
{
  "item_id": 67890,
  "title": "Advanced Machine Learning Course",
  "description": "Comprehensive course covering deep learning...",
  "categories": ["education", "technology"],
  "tags": ["machine-learning", "deep-learning", "python"],
  "features": {
    "duration_hours": 40,
    "difficulty": "advanced",
    "rating": 4.8
  },
  "created_at": "2025-03-15T00:00:00Z",
  "popularity_score": 0.76
}
```

#### POST /items

Add a new item to the catalog.

**Request Body:**

```json
{
  "item_id": 67890,
  "title": "New Course Title",
  "description": "Course description...",
  "categories": ["education"],
  "tags": ["programming"],
  "features": {
    "duration_hours": 20,
    "difficulty": "beginner"
  }
}
```

#### GET /items/search

Search items with filters.

**Parameters:**

- `q` (query): Search query
- `categories` (query): Comma-separated categories
- `tags` (query): Comma-separated tags
- `limit` (query): Number of results

### Interaction Tracking

#### POST /interactions

Track a user-item interaction.

**Request Body:**

```json
{
  "user_id": 12345,
  "item_id": 67890,
  "interaction_type": "view",
  "context": {
    "source": "homepage",
    "position": 1,
    "session_id": "abc123"
  },
  "timestamp": "2025-06-05T10:30:00Z"
}
```

**Interaction Types:**

- `view`: User viewed item
- `click`: User clicked on item
- `like`: User liked/favorited item
- `dislike`: User disliked item
- `purchase`: User purchased/enrolled
- `share`: User shared item
- `rating`: User rated item (include `rating` field: 1-5)

#### POST /interactions/batch

Track multiple interactions at once.

**Request Body:**

```json
{
  "interactions": [
    {
      "user_id": 12345,
      "item_id": 67890,
      "interaction_type": "view",
      "timestamp": "2025-06-05T10:30:00Z"
    }
  ]
}
```

#### GET /interactions/user/{user_id}

Get user's interaction history.

**Parameters:**

- `limit` (query): Number of interactions
- `interaction_type` (query): Filter by interaction type
- `since` (query): ISO timestamp for interactions since date

### Analytics

#### GET /analytics/user/{user_id}/stats

Get detailed analytics for a user.

**Response:**

```json
{
  "user_id": 12345,
  "period": "last_30_days",
  "stats": {
    "recommendations_served": 150,
    "clicks": 18,
    "conversions": 3,
    "click_through_rate": 0.12,
    "conversion_rate": 0.02,
    "avg_session_duration": 300,
    "favorite_categories": ["technology", "science"]
  }
}
```

#### GET /analytics/item/{item_id}/stats

Get analytics for a specific item.

**Response:**

```json
{
  "item_id": 67890,
  "period": "last_30_days",
  "stats": {
    "times_recommended": 1000,
    "unique_users_recommended": 850,
    "clicks": 120,
    "conversions": 25,
    "average_rating": 4.5,
    "recommendation_algorithms": {
      "collaborative": 0.4,
      "content": 0.3,
      "hybrid": 0.3
    }
  }
}
```

#### GET /analytics/system/performance

Get system-wide performance metrics.

**Response:**

```json
{
  "period": "last_24_hours",
  "metrics": {
    "total_recommendations": 50000,
    "avg_response_time_ms": 45,
    "cache_hit_rate": 0.78,
    "algorithm_performance": {
      "collaborative": { "accuracy": 0.85, "coverage": 0.92 },
      "content": { "accuracy": 0.79, "coverage": 0.95 },
      "hybrid": { "accuracy": 0.88, "coverage": 0.94 }
    },
    "real_time_updates": 12000
  }
}
```

### A/B Testing

#### GET /experiments

List active A/B experiments.

**Response:**

```json
{
  "experiments": [
    {
      "id": "rec_algo_test_1",
      "name": "Collaborative vs Hybrid Algorithm",
      "status": "active",
      "traffic_split": 0.5,
      "start_date": "2025-06-01T00:00:00Z",
      "metrics": {
        "control_group": { "ctr": 0.12, "conversion": 0.03 },
        "test_group": { "ctr": 0.14, "conversion": 0.035 }
      }
    }
  ]
}
```

#### POST /experiments/{experiment_id}/assign

Assign a user to an experiment group.

**Request Body:**

```json
{
  "user_id": 12345
}
```

**Response:**

```json
{
  "user_id": 12345,
  "experiment_id": "rec_algo_test_1",
  "group": "test",
  "algorithm": "hybrid_v2"
}
```

## Error Handling

The API uses conventional HTTP response codes:

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing/invalid token)
- `404` - Not Found (resource doesn't exist)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

**Error Response Format:**

```json
{
  "error": {
    "code": "INVALID_USER_ID",
    "message": "User ID must be a positive integer",
    "details": {
      "field": "user_id",
      "value": "invalid_id"
    }
  },
  "timestamp": "2025-06-05T10:30:00Z",
  "request_id": "req_123456"
}
```

## Rate Limiting

- **General endpoints**: 1000 requests per hour per user
- **Recommendation endpoints**: 100 requests per minute per user
- **Analytics endpoints**: 50 requests per minute per user
- **Interaction tracking**: 10000 requests per hour per user

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1620000000
```

## SDK Examples

### Python

```python
import requests

class RecommendationClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def get_recommendations(self, user_id, limit=10):
        response = requests.get(
            f"{self.base_url}/recommendations/user/{user_id}",
            params={"limit": limit},
            headers=self.headers
        )
        return response.json()

    def track_interaction(self, user_id, item_id, interaction_type):
        data = {
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction_type
        }
        response = requests.post(
            f"{self.base_url}/interactions",
            json=data,
            headers=self.headers
        )
        return response.json()

# Usage
client = RecommendationClient("http://localhost:8000/api/v1", "your_token")
recommendations = client.get_recommendations(12345)
```

### JavaScript

```javascript
class RecommendationAPI {
  constructor(baseURL, token) {
    this.baseURL = baseURL
    this.headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    }
  }

  async getRecommendations(userId, limit = 10) {
    const response = await fetch(
      `${this.baseURL}/recommendations/user/${userId}?limit=${limit}`,
      { headers: this.headers }
    )
    return response.json()
  }

  async trackInteraction(userId, itemId, interactionType) {
    const response = await fetch(`${this.baseURL}/interactions`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        user_id: userId,
        item_id: itemId,
        interaction_type: interactionType,
      }),
    })
    return response.json()
  }
}
```

## Webhooks

The system can send webhooks for important events:

### Webhook Events

- `recommendation.generated` - New recommendations generated
- `user.interaction` - User interaction tracked
- `model.updated` - Recommendation model retrained
- `experiment.completed` - A/B test completed

### Webhook Payload Example

```json
{
  "event": "recommendation.generated",
  "timestamp": "2025-06-05T10:30:00Z",
  "data": {
    "user_id": 12345,
    "recommendation_count": 10,
    "algorithm": "hybrid",
    "session_id": "abc123"
  }
}
```

To configure webhooks, contact support or use the admin dashboard.
