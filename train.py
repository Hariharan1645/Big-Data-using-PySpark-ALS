"""
PySpark ALS Book Recommendation System - Model Training & Evaluation
======================================================================
This script performs:
1. Environment setup (auto-detects valid Java JDK).
2. Data loading from local CSV files (ratings.csv, books.csv) via PySpark.
3. Exploratory Data Analysis (EDA) & saving visual charts.
4. Train/Test split (80% / 20%, seed=42).
5. Collaborative Filtering model training using PySpark ALS.
6. Hyperparameter tuning & evaluation (RMSE & MAE metrics saved to JSON).
7. Top-10 book recommendations generated for all users (saved to CSV).
8. Highly-rated books (rating >= 4) per user extracted (saved to CSV).

Designed for clear explanation in Viva and practical execution on a laptop.
"""

import os
import sys
import json
import shutil
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless chart generation
import matplotlib.pyplot as plt

# ==============================================================================
# Step 0: Ensure Valid Java Environment for PySpark
# ==============================================================================
def setup_environment():
    """Ensure JAVA_HOME points to an existing, valid JDK directory."""
    current_java_home = os.environ.get("JAVA_HOME", "")
    if not current_java_home or not os.path.exists(current_java_home):
        java_bin = shutil.which("java")
        if java_bin:
            # Resolves parent of 'bin' directory as JAVA_HOME
            detected_java_home = os.path.dirname(os.path.dirname(os.path.abspath(java_bin)))
            os.environ["JAVA_HOME"] = detected_java_home
            print(f"[ENVIRONMENT] JAVA_HOME set to: {detected_java_home}")
        else:
            print("[WARNING] 'java' not found in PATH. PySpark may fail without a valid JDK.")
    else:
        print(f"[ENVIRONMENT] Using existing JAVA_HOME: {current_java_home}")

setup_environment()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def create_spark_session():
    """
    Initializes a PySpark session optimized for a local multi-core laptop.
    Using local[*] utilizes all available CPU cores.
    JVM options allow Java 17/21 compatibility.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Initializing PySpark Session (local[*] mode)")
    print("=" * 70)
    
    spark = (
        SparkSession.builder
        .appName("BookRecommender-ALS")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.driver.extraJavaOptions",
            "-Djava.security.manager=allow "
            "--add-opens=java.base/java.lang=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
            "--add-opens=java.base/java.io=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/java.util=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED"
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[SUCCESS] Spark Session initialized! Spark version: {spark.version}")
    return spark

def load_data(spark):
    """
    Reads local ratings.csv and books.csv directly into PySpark DataFrames.
    Cleans null values and casts columns to appropriate types.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Loading CSV Data into PySpark DataFrames")
    print("=" * 70)
    
    ratings_file = os.path.join(DATA_DIR, "ratings.csv")
    books_file = os.path.join(DATA_DIR, "books.csv")

    if not os.path.exists(ratings_file) or not os.path.exists(books_file):
        raise FileNotFoundError(
            f"Missing dataset files! Please run 'python download_data.py' first.\n"
            f"Expected: {ratings_file} and {books_file}"
        )

    # 1. Load Ratings
    print(f"Reading ratings from: {ratings_file}")
    ratings_df = (
        spark.read.csv(ratings_file, header=True, inferSchema=True)
        .select(
            F.col("user_id").cast("integer"),
            F.col("book_id").cast("integer"),
            F.col("rating").cast("float")
        )
        .na.drop()
    )

    # 2. Load Books
    print(f"Reading books from: {books_file}")
    raw_books = spark.read.csv(books_file, header=True, inferSchema=True)
    
    # Check for image_url if present
    has_image = "image_url" in raw_books.columns
    select_cols = [
        F.col("book_id").cast("integer"),
        F.col("title").cast("string"),
        F.col("authors").cast("string"),
        F.col("average_rating").cast("float")
    ]
    if has_image:
        select_cols.append(F.col("image_url").cast("string"))

    books_df = raw_books.select(*select_cols).na.drop(subset=["book_id", "title"])
    
    # Cache for reuse
    ratings_df.cache()
    books_df.cache()
    
    return ratings_df, books_df

def explore_data_and_generate_charts(ratings_df, books_df):
    """
    Calculates summary statistics and exports visual charts:
    - rating_distribution.png
    - top_rated_books.png
    """
    print("\n" + "=" * 70)
    print("STEP 3: Exploratory Data Analysis (EDA) & Visualizations")
    print("=" * 70)

    total_ratings = ratings_df.count()
    total_users = ratings_df.select("user_id").distinct().count()
    total_books = ratings_df.select("book_id").distinct().count()
    
    overall_avg_rating = ratings_df.select(F.avg("rating")).first()[0]
    overall_avg_rating = round(float(overall_avg_rating), 2) if overall_avg_rating else 0.0

    print(f"[*] Total Ratings in Dataset : {total_ratings:,}")
    print(f"[*] Total Unique Users       : {total_users:,}")
    print(f"[*] Total Unique Books       : {total_books:,}")
    print(f"[*] Overall Average Rating   : {overall_avg_rating} / 5.0")

    # Rating distribution
    print("\n--- Rating Distribution (1 to 5 Stars) ---")
    dist_df = ratings_df.groupBy("rating").count().orderBy("rating").toPandas()
    for _, row in dist_df.iterrows():
        pct = (row['count'] / total_ratings) * 100
        print(f"  Rating {int(row['rating'])} stars : {int(row['count']):10,d} ({pct:5.2f}%)")

    # Save Rating Distribution Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        dist_df["rating"].astype(int), 
        dist_df["count"], 
        color="#4F46E5", 
        edgecolor="#312E81",
        alpha=0.85
    )
    plt.title("Distribution of Book Ratings (Goodbooks-10k)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Rating (Stars)", fontsize=12)
    plt.ylabel("Number of Ratings", fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (total_ratings*0.01), f"{int(yval):,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart1_path = os.path.join(OUTPUTS_DIR, "rating_distribution.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"[CHART SAVED] {chart1_path}")

    # Top 10 Most-Rated Books
    print("\n--- Top 10 Most-Rated Books ---")
    top_books_spark = (
        ratings_df.groupBy("book_id")
        .count()
        .orderBy(F.col("count").desc())
        .limit(10)
        .join(books_df, on="book_id", how="inner")
        .select("book_id", "title", "authors", "count")
        .orderBy(F.col("count").desc())
    )
    top_books_pd = top_books_spark.toPandas()

    for idx, r in top_books_pd.iterrows():
        print(f"  {idx+1:2d}. {r['title']} by {r['authors']} ({r['count']:,} ratings)")

    # Save Top Rated Books Chart
    plt.figure(figsize=(10, 6))
    short_titles = [t[:35] + "..." if len(t) > 35 else t for t in top_books_pd["title"]]
    # Invert order for nice top-to-bottom bar chart
    y_pos = range(len(short_titles))
    plt.barh(y_pos, top_books_pd["count"][::-1], color="#06B6D4", edgecolor="#0891B2", alpha=0.85)
    plt.yticks(y_pos, short_titles[::-1], fontsize=10)
    plt.xlabel("Number of Ratings", fontsize=12)
    plt.title("Top 10 Most-Rated Books in Dataset", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    chart2_path = os.path.join(OUTPUTS_DIR, "top_rated_books.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"[CHART SAVED] {chart2_path}")

    return {
        "total_ratings": int(total_ratings),
        "total_users": int(total_users),
        "total_books": int(total_books),
        "overall_avg_rating": float(overall_avg_rating)
    }

def train_and_evaluate_als(ratings_df, books_df, eda_stats):
    """
    Splits data (80% train / 20% test).
    Trains PySpark ALS model with collaborative filtering.
    Evaluates with RMSE & MAE and saves metrics to outputs/metrics.json.
    Generates recommendations and liked books.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Train / Test Split (80% Train, 20% Test, seed=42)")
    print("=" * 70)
    
    train_data, test_data = ratings_df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"[*] Training ratings count : {train_count:,}")
    print(f"[*] Testing ratings count  : {test_count:,}")

    print("\n" + "=" * 70)
    print("STEP 5: PySpark ALS Collaborative Filtering Model Training")
    print("=" * 70)
    print("ALS Parameters:")
    print(" - Latent factors (rank)  : 10")
    print(" - Maximum iterations     : 10")
    print(" - Regularization param   : 0.1")
    print(" - Cold start strategy    : drop (drops NaN predictions on unseen items)")
    print(" - Nonnegative factors    : True (ensures interpretable positive factors)")

    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    # Hyperparameter comparison: compare regParam=0.1 vs regParam=0.05
    candidate_params = [
        {"rank": 10, "maxIter": 10, "regParam": 0.1},
        {"rank": 10, "maxIter": 10, "regParam": 0.05}
    ]

    best_model = None
    best_rmse = float("inf")
    best_mae = float("inf")
    best_params = candidate_params[0]

    for p in candidate_params:
        print(f"\n[TRAINING] Evaluating ALS config: rank={p['rank']}, maxIter={p['maxIter']}, regParam={p['regParam']}...")
        als = ALS(
            userCol="user_id",
            itemCol="book_id",
            ratingCol="rating",
            rank=p["rank"],
            maxIter=p["maxIter"],
            regParam=p["regParam"],
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42
        )
        model = als.fit(train_data)
        predictions = model.transform(test_data)
        rmse_val = evaluator_rmse.evaluate(predictions)
        mae_val = evaluator_mae.evaluate(predictions)
        print(f"  -> Result: Test RMSE = {rmse_val:.4f} | Test MAE = {mae_val:.4f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_mae = mae_val
            best_model = model
            best_params = p

    print("\n" + "-" * 70)
    print(f"[BEST MODEL SELECTED]: regParam={best_params['regParam']} | RMSE={best_rmse:.4f} | MAE={best_mae:.4f}")
    print("-" * 70)

    # Save metrics to outputs/metrics.json
    metrics_info = {
        "rmse": round(float(best_rmse), 4),
        "mae": round(float(best_mae), 4),
        "best_rank": best_params["rank"],
        "best_maxIter": best_params["maxIter"],
        "best_regParam": best_params["regParam"],
        "train_samples": int(train_count),
        "test_samples": int(test_count),
        "total_ratings": eda_stats["total_ratings"],
        "total_users": eda_stats["total_users"],
        "total_books": eda_stats["total_books"],
        "overall_avg_rating": eda_stats["overall_avg_rating"]
    }
    metrics_file = os.path.join(OUTPUTS_DIR, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_info, f, indent=4)
    print(f"[METRICS SAVED] Saved model evaluation metrics to {metrics_file}")

    # ==============================================================================
    # Step 6: Generate Top 10 Recommendations for All Users
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Generating Top 10 Recommendations for All Users")
    print("=" * 70)
    print("Calling best_model.recommendForAllUsers(10)...")
    
    # recommendForAllUsers returns DataFrame: [user_id: int, recommendations: array<struct<book_id, rating>>]
    user_recs_df = best_model.recommendForAllUsers(10)

    # Explode recommendations array into individual rows
    exploded_recs = (
        user_recs_df.withColumn("rec", F.explode("recommendations"))
        .select(
            F.col("user_id"),
            F.col("rec.book_id").alias("book_id"),
            F.round(F.col("rec.rating"), 2).alias("predicted_rating")
        )
    )

    # Join with books dataframe to fetch book title and author
    recs_with_details = (
        exploded_recs.join(
            books_df.select("book_id", "title", "authors"), 
            on="book_id", 
            how="left"
        )
        .select("user_id", "book_id", "title", "authors", "predicted_rating")
        .orderBy("user_id", F.col("predicted_rating").desc())
    )

    recs_csv_path = os.path.join(OUTPUTS_DIR, "recommendations.csv")
    print(f"Exporting recommendations table to {recs_csv_path}...")
    recs_pd = recs_with_details.toPandas()
    recs_pd.to_csv(recs_csv_path, index=False)
    print(f"[RECOMMENDATIONS SAVED] Successfully exported {len(recs_pd):,} recommendation rows!")

    # ==============================================================================
    # Step 7: Save Each User's Highly Rated Books (rating >= 4)
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Extracting Each User's Highly Rated Books (Rating >= 4)")
    print("=" * 70)

    # Filter ratings >= 4 and rank within user partition to keep top 10 liked books per user
    window_spec = Window.partitionBy("user_id").orderBy(F.col("rating").desc())
    
    liked_books_spark = (
        ratings_df.filter(F.col("rating") >= 4)
        .withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") <= 10)
        .drop("rank")
        .join(
            books_df.select("book_id", "title", "authors"),
            on="book_id",
            how="left"
        )
        .select("user_id", "book_id", "title", "authors", "rating")
        .orderBy("user_id", F.col("rating").desc())
    )

    liked_csv_path = os.path.join(OUTPUTS_DIR, "user_liked_books.csv")
    print(f"Exporting user liked books to {liked_csv_path}...")
    liked_pd = liked_books_spark.toPandas()
    liked_pd.to_csv(liked_csv_path, index=False)
    print(f"[LIKED BOOKS SAVED] Successfully exported {len(liked_pd):,} liked book records!")

    # Sample output for viva demonstration: User 10
    print("\n" + "=" * 70)
    print("SAMPLE DEMONSTRATION: RECOMMENDATIONS FOR USER 10")
    print("=" * 70)
    user_10_recs = recs_pd[recs_pd["user_id"] == 10]
    if not user_10_recs.empty:
        for i, (_, row) in enumerate(user_10_recs.iterrows(), 1):
            print(f"  {i:2d}. {row['title']} by {row['authors']} (Predicted Rating: {row['predicted_rating']} / 5)")
    else:
        print("  User 10 not found in recommendations table.")
    print("=" * 70)

def main():
    spark = create_spark_session()
    try:
        ratings_df, books_df = load_data(spark)
        eda_stats = explore_data_and_generate_charts(ratings_df, books_df)
        train_and_evaluate_als(ratings_df, books_df, eda_stats)
        print("\n" + "=" * 70)
        print("TRAINING & PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("Artifacts generated in outputs/:")
        print(" - rating_distribution.png")
        print(" - top_rated_books.png")
        print(" - metrics.json")
        print(" - recommendations.csv")
        print(" - user_liked_books.csv")
        print("=" * 70)
    finally:
        spark.stop()
        print("[CLEANUP] SparkSession stopped.")

if __name__ == "__main__":
    main()
