# 📚 Big Data Book Recommendation System using PySpark ALS & Streamlit

An end-to-end Big Data machine learning project implementing Collaborative Filtering on the **Goodbooks-10k** dataset (~6 Million ratings) using **Apache Spark (PySpark ALS)** and an interactive **Streamlit** user interface.

---

## 🌟 Key Features
- **Big Data Scale:** Processes nearly 6,000,000 ratings across 53,424 users and 10,000 books locally using multi-threaded PySpark (`local[*]`).
- **Matrix Factorization:** Employs the **Alternating Least Squares (ALS)** algorithm for Collaborative Filtering.
- **Model Evaluation:** Evaluates predictions using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** with hyperparameter tuning.
- **Decoupled Architecture:** Spark handles heavy data transformations, model training, and batch recommendation generation. The Streamlit UI loads precomputed CSV/JSON artifacts instantly without the overhead of starting a JVM at runtime.
- **Interactive UI:** Dynamic user ID input, recommendation count slider, user history view, metric performance badges, and exploratory data visualizations.

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------+
|                      DATA INGESTION                         |
|   raw.githubusercontent.com/zygmuntz/goodbooks-10k          |
|         ratings.csv (6M rows) & books.csv (10K rows)        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   BIG DATA PROCESSING & ML                  |
|                 Apache Spark (local[*] mode)                |
|  - Data Exploration & Matplotlib Visualizations             |
|  - 80/20 Train-Test Split (seed=42)                         |
|  - Matrix Factorization via PySpark ALS                     |
|  - Evaluation: RMSE (0.8266) & MAE (0.6520)                 |
|  - Full Batch Inference: recommendForAllUsers(10)           |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     SAVED ARTIFACTS                         |
|   outputs/recommendations.csv     outputs/metrics.json      |
|   outputs/user_liked_books.csv    outputs/*.png charts      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   STREAMLIT WEB INTERFACE                   |
|   Instant pandas loading, user profile inspection,          |
|   custom recommendations, model KPI cards, and EDA charts   |
+-------------------------------------------------------------+
```

---

## 📂 Project Structure

```
book-recommender/
│
├── data/
│   ├── ratings.csv            # 5,976,479 user ratings
│   └── books.csv              # Metadata for 10,000 books
│
├── outputs/
│   ├── recommendations.csv    # Precomputed top 10 recommendations for all users
│   ├── user_liked_books.csv   # Historical 4 & 5-star books per user
│   ├── metrics.json           # Evaluation metrics (RMSE, MAE, dataset stats)
│   ├── rating_distribution.png# Rating frequency chart (1 to 5 stars)
│   └── top_rated_books.png    # Top 10 most-rated books chart
│
├── download_data.py           # Automated downloader for Goodbooks-10k
├── train.py                   # PySpark pipeline: EDA, ALS training, recommendations
├── app.py                     # Interactive Streamlit UI
├── requirements.txt           # Project dependencies
├── REPORT_NOTES.md            # Comprehensive project documentation for reports/viva
└── README.md                  # Project overview and instructions
```

---

## 🧠 How ALS (Alternating Least Squares) Works (In Simple Words)

In recommendation systems, we have a huge **User-Item Matrix** where rows represent users, columns represent books, and each cell contains a rating (1 to 5). Because most users only read a small fraction of all books, this matrix is **extremely sparse** (over 98% empty).

1. **Latent Factors:** Instead of comparing users directly, ALS assumes that preferences and books can be represented by a small set of hidden concepts or "latent features" (e.g., sci-fi intensity, writing style, pace, fantasy elements).
2. **Matrix Factorization:** ALS decomposes the rating matrix $R$ ($U \times I$) into two smaller low-rank matrices:
   - A **User Factor Matrix** $P$ ($U \times k$)
   - An **Item Factor Matrix** $Q$ ($I \times k$)
   Such that $R \approx P \times Q^T$.
3. **The "Alternating" Trick:** Solving both matrices simultaneously is computationally non-linear. ALS makes this simple:
   - Fix $Q$ and solve for $P$ using standard linear least squares regression.
   - Fix $P$ and solve for $Q$ using standard linear least squares regression.
   - Alternate between these steps back and forth until the error converges!
4. **Prediction:** To predict user $u$'s rating for book $i$, we simply take the dot product: $\hat{r}_{ui} = p_u \cdot q_i$.

---

## ⚙️ Prerequisites & Setup

### 1. Java JDK
Apache Spark requires **Java (JDK 8, 11, 17, or 21)**.
Check if Java is installed:
```powershell
java -version
```

### 2. Python Environment
Python 3.10, 3.11, or 3.12 is recommended.
Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Execution Guide (Commands in Order)

### Step 1: Download the Dataset
Downloads `ratings.csv` (~68 MB) and `books.csv` (~3 MB) into `data/`:
```powershell
python download_data.py
```

### Step 2: Train the Model & Export Recommendations
Runs the complete Big Data pipeline in PySpark (EDA, ALS training, grid search, metrics evaluation, batch recommendations):
```powershell
python train.py
```
*Expected training time on a multi-core laptop: ~1 to 2 minutes.*

### Step 3: Launch the Streamlit App
Starts the web application:
```powershell
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

---

## 📊 Sample Output (User #10)

### Top-5 Recommendations Generated for User 10:
| Rank | Book Title | Author(s) | Predicted Rating |
|:---:|:---|:---|:---:|
| #1 | *The Indispensable Calvin and Hobbes* | Bill Watterson | **4.85 / 5.0** |
| #2 | *The Authoritative Calvin and Hobbes* | Bill Watterson | **4.82 / 5.0** |
| #3 | *The Calvin and Hobbes Lazy Sunday Book* | Bill Watterson | **4.78 / 5.0** |
| #4 | *The Pigeon Finds a Hot Dog!* | Mo Willems | **4.76 / 5.0** |
| #5 | *A Different Blue* | Amy Harmon | **4.75 / 5.0** |

### Model Performance Metrics:
- **Test Set RMSE:** `0.8266`
- **Test Set MAE:** `0.6520`
- **Total Ratings Evaluated:** `5,976,479`
- **Train / Test Split:** 80% (`4,781,210` samples) / 20% (`1,195,269` samples)
