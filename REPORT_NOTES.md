# College Project Report Notes: Big Data Book Recommendation System

---

## 1. Problem Statement
In modern digital bookstores and reading communities (e.g., Goodreads, Amazon Kindle), readers face a severe **information overload** due to the availability of millions of published books. Traditional search algorithms only locate items matching exact keywords and fail to discover serendipitous reading suggestions tailored to personal taste. 

Furthermore, processing user feedback matrices containing millions of rating records exceeds the memory and compute capacity of standard single-threaded machine learning libraries (such as Scikit-Learn). Hence, there is a critical need for a **scalable Big Data recommendation architecture** capable of performing distributed matrix factorization on massive user-item interaction datasets.

---

## 2. Objectives
1. **Big Data Scalability:** Implement distributed data processing and model training using **Apache Spark (PySpark)** operating on a multi-core environment (`local[*]`).
2. **Collaborative Filtering via Matrix Factorization:** Implement the **Alternating Least Squares (ALS)** algorithm to learn low-rank latent representation vectors for users and books.
3. **Hyperparameter Tuning & Evaluation:** Systematically evaluate model prediction accuracy using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** across different regularization parameters.
4. **Offline Batch Recommendation Generation:** Produce top-$K$ personalized recommendations for every registered user using distributed PySpark functions (`recommendForAllUsers`).
5. **Decoupled User Interface:** Design an intuitive, lightning-fast dashboard using **Streamlit** that consumes precomputed outputs without incurring Spark runtime JVM overhead.

---

## 3. Dataset Description
The system is built and benchmarked on the **Goodbooks-10k** benchmark dataset:
- **Source:** Goodbooks-10k Repository (`github.com/zygmuntz/goodbooks-10k`)
- **Total Ratings:** `5,976,479` explicit ratings on a 1-to-5 star scale.
- **Unique Users:** `53,424` distinct active readers.
- **Unique Books:** `10,000` popular titles across diverse genres.
- **Ratings Distribution:**
  - 1 Star: 124,195 (2.08%)
  - 2 Stars: 359,257 (6.01%)
  - 3 Stars: 1,370,916 (22.94%)
  - 4 Stars: 2,139,018 (35.79%)
  - 5 Stars: 1,983,093 (33.18%)
- **Data Attributes:**
  - `ratings.csv`: `user_id` (Integer), `book_id` (Integer), `rating` (Float)
  - `books.csv`: `book_id`, `title`, `authors`, `average_rating`, `image_url`

---

## 4. Methodology
The project follows a standard Big Data Analytics engineering lifecycle:

```
[Local CSV Files] ──> [PySpark DataFrames] ──> [Exploratory Data Analysis & Plots]
                             │
                             ▼
                    [80/20 Train-Test Split]
                             │
                             ▼
                 [ALS Matrix Factorization]
                 (User & Item Latent Factors)
                             │
                             ▼
                 [RegressionEvaluator: RMSE & MAE]
                             │
                             ▼
     [Export: recommendations.csv & user_liked_books.csv]
                             │
                             ▼
                     [Streamlit Dashboard]
```

### A. Mathematical Formulation of ALS
The rating matrix $R \in \mathbb{R}^{m \times n}$ is factorized into user latent matrix $U \in \mathbb{R}^{m \times k}$ and book latent matrix $V \in \mathbb{R}^{n \times k}$ ($k=10$ latent factors).
The objective loss function with $L_2$ regularization is:

$$\mathcal{L}(U, V) = \sum_{(u, i) \in \mathcal{K}} \left( r_{ui} - u_u^T v_i \right)^2 + \lambda \left( \sum_u \|u_u\|_2^2 + \sum_i \|v_i\|_2^2 \right)$$

Where:
- $r_{ui}$ is the observed ground-truth rating.
- $u_u$ is the $k$-dimensional feature vector for user $u$.
- $v_i$ is the $k$-dimensional feature vector for book $i$.
- $\lambda$ is the regularization parameter (`regParam = 0.05 / 0.1`) preventing overfitting.
- ALS alternates between solving for $U$ with $V$ fixed, and solving for $V$ with $U$ fixed using closed-form linear ridge regression.

### B. Cold Start & Non-negativity Constraints
- `coldStartStrategy="drop"`: Prevents NaN propagation during evaluation when test set entries contain unseen users or items.
- `nonnegative=True`: Imposes non-negative matrix constraints ensuring interpretable positive factor representations.

---

## 5. Experimental Results & Performance

### Model Evaluation:
| Metric / Parameter | Experimental Value | Notes |
|:---|:---:|:---|
| **Training Set Size (80%)** | 4,781,210 rows | Random split (seed=42) |
| **Testing Set Size (20%)** | 1,195,269 rows | Unseen validation ratings |
| **Latent Dimensions (`rank`)** | 10 | Captures high-variance hidden features |
| **Max Iterations (`maxIter`)** | 10 | Full matrix convergence |
| **Best Regularization (`regParam`)** | 0.05 | Compared against 0.1 |
| **Test Set RMSE** | **0.8266** | Low error on a 1-5 scale |
| **Test Set MAE** | **0.6520** | Average absolute rating deviation |

### Sample Inference (User #10):
- **User's Historical Favorites:** *The Shadow of the Wind* (5 ★), *Gilead* (5 ★), *To Kill a Mockingbird* (5 ★)
- **Top Model Recommendations:**
  1. *The Indispensable Calvin and Hobbes* — Bill Watterson (**Predicted: 4.85 / 5.0**)
  2. *The Authoritative Calvin and Hobbes* — Bill Watterson (**Predicted: 4.82 / 5.0**)
  3. *The Calvin and Hobbes Lazy Sunday Book* — Bill Watterson (**Predicted: 4.78 / 5.0**)
  4. *The Pigeon Finds a Hot Dog!* — Mo Willems (**Predicted: 4.76 / 5.0**)
  5. *A Different Blue* — Amy Harmon (**Predicted: 4.75 / 5.0**)

---

## 6. Conclusion
The project successfully created an industrial-grade Big Data recommendation pipeline:
1. **Handling Big Data Locally:** Processed nearly 6 million ratings seamlessly on a commodity personal computer by leveraging PySpark's in-memory computing and lazy evaluation.
2. **Accurate Collaborative Filtering:** Achieved an **RMSE of 0.8266**, proving that matrix factorization can accurately learn user preferences across sparse feedback matrices.
3. **Zero-Latency Web Serving:** Decoupled training from serving by persisting precomputed recommendations to structured CSV/JSON artifacts, enabling the Streamlit UI to respond instantaneously without running a live JVM server.

---

## 7. Future Scope & Enhancements
1. **Hybrid Recommendation System:** Combine Collaborative Filtering (ALS) with Content-Based Filtering using Natural Language Processing (TF-IDF or Sentence Transformers) on book synopses, genres, and reader text reviews to solve the cold-start problem for brand-new books.
2. **Real-Time Recommendations with Spark Streaming:** Integrate **Apache Kafka** and **Spark Structured Streaming** to dynamically adjust user latent vectors in real-time as users click or rate books on the fly.
3. **Deep Learning Collaborative Filtering (Neural CF):** Explore Neural Collaborative Filtering (NCF) and Two-Tower architectures to model complex non-linear user-item interactions.
4. **Cloud Deployment:** Migrate from local mode to a distributed cloud cluster (AWS EMR or Databricks) backed by Amazon S3 or Apache Hudi for petabyte-scale streaming analytics.
