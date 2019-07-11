// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # DATA 612 - Final Project
// MAGIC By Michael D'Acampora and Calvin Wong
// MAGIC 
// MAGIC In this walkthrough we compare and apply three different algorithms to two datasets used for recommendation systems. The datasets are from the [MovieLens repository](https://grouplens.org/datasets/movielens/), and the ones we chose are the small version with 100k records, and the largest version with over 27m records. The data consists of four columns, `userId`, `movieId`, `rating`, and `timestamp`. 
// MAGIC 
// MAGIC We decided to split the work up into two parts, 1) Spark Scala and 2) Python. Since Spark has an impressive Alternating Least Squares Matrix Factorization algorithm, we decided to employ this method to get experience in the Databricks environment and distributed computing in general. There is a shift in approach that is required when using Spark that is different than the similiarities that exist between R and Python. Fist, the base data structure to understand is the Resilient Distributed Dataset, or RDD. Spark takes these RDD's and distributes work across however many nodes you have dedicated to the job. The RDD is designed so Spark can quickly partition pieces as necessary. There are also DataFrames simliar to R and Python, and they exist as part of SparkSQL. Manipulating the data is a bit different, and since Spark was written in Scala, we felt it best to use the origin language and see its paradigm to better understand how Pyspark, Rspark and sparklyr work.
// MAGIC 
// MAGIC Scala is mostly a functional programming language and one can quickly see its fingerprints as it pertains to Spark. There is plenty of use of the `map()` function, which is used to manipulate data contained in RDDs and Dataframes. It is an anyonymous function similar to `lambda` functions in Python. Interestingly enough, after looking at Pyspark project you see that `lambda`s are often used.
// MAGIC 
// MAGIC #### Variable names:
// MAGIC 
// MAGIC Small -> 100k <br>
// MAGIC Medium -> 1m <br>
// MAGIC Large -> 10m <br>
// MAGIC XL -> 27m <br>

// COMMAND ----------

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## First up: Spark Scala with ALS
// MAGIC We created an Azure Databricks cluster with up to 8 worker nodes for this project. After loading our imports we can take a look at the filepaths of the files contained in workspace.

// COMMAND ----------

display(dbutils.fs.ls("/FileStore/tables/"))

// COMMAND ----------

// MAGIC %md
// MAGIC From here we load the small and large datasets as Spark DataFrames

// COMMAND ----------

// Load data as Spark DataFrame  

// 100k records
val dfSmall = spark.read
  .format("csv")
  .option("header", true)
  .load("dbfs:/FileStore/tables/ratings_small.csv")

// 1m records
val dfMed = spark.read
  .format("csv")
  .option("header", true)
  .load("dbfs:/FileStore/tables/ratings_medium_clean.csv")

// 10m records
val dfLarge = spark.read
  .format("csv")
  .option("header", true)
  .load("dbfs:/FileStore/tables/ratings_large10m_clean.csv")

// 27m records
val dfXl = spark.read
  .format("csv")
  .option("header", true)
  .load("dbfs:/FileStore/tables/ratings_large.csv")


// COMMAND ----------

// MAGIC  %md
// MAGIC #### Data prep 

// COMMAND ----------

// Select appropriate columns
val ratingsDFsmall = dfSmall.select("userId", "movieId", "rating")
              
val ratingsDFmedium = dfMed.select("userId", "movieId", "rating")

val ratingsDFlarge = dfLarge.select("userId", "movieId", "rating") 

val ratingsDFxl = dfXl.select("userId", "movieId", "rating") 

// COMMAND ----------

// MAGIC  %md
// MAGIC 
// MAGIC  After the dataframes are prepped, we split them up into training and test sets at an 80/20 ratio. 

// COMMAND ----------

// Create your training and test data                      
val splitsSmall = ratingsDFsmall.randomSplit(Array(0.8, 0.2), seed = 123L)
val splitsMedium = ratingsDFmedium.randomSplit(Array(0.8, 0.2), seed = 123L)
val splitsLarge = ratingsDFlarge.randomSplit(Array(0.8, 0.2), seed = 123L)
val splitsXl = ratingsDFxl.randomSplit(Array(0.8, 0.2), seed = 123L)

val (trainingDataSmall, testDataSmall) = (splitsSmall(0), splitsSmall(1))
val (trainingDataMedium, testDataMedium) = (splitsMedium(0), splitsMedium(1))
val (trainingDataLarge, testDataLarge) = (splitsLarge(0), splitsLarge(1))
val (trainingDataXl, testDataXl) = (splitsXl(0), splitsXl(1))

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC Since the dataframes were all string values we have to convert to numeric values for this exercise. Note the use of the `map()` function that grabs the column values and converts within the `Rating` class in Mllib. These training and test sets are created for both the small and large datasets.

// COMMAND ----------

// spark implicits needed here for serializing 
import spark.implicits._

// CONVERT TEST SETS TO NUMERIC
// ==========================================
val trainingSetSmall = trainingDataSmall.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val trainingSetMedium = trainingDataMedium.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val trainingSetLarge = trainingDataLarge.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val trainingSetXl= trainingDataXl.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})


// CONVERT TEST SETS TO NUMERIC
// ==========================================
val testSetSmall = testDataSmall.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val testSetMedium = testDataMedium.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val testSetLarge = testDataLarge.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val testSetXl = testDataXl.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

// COMMAND ----------

// MAGIC %md
// MAGIC #### Model creation
// MAGIC 
// MAGIC Now that we are ready to feed a model, we create on. As mentioned above we are using the Alternating Least Squares (ALS) Matrix Factorization model. `val als` instantiates the model with our parameters, and afterwards is fit to the small and large training data.
// MAGIC 
// MAGIC Once the training data is fit, we use that model to predict against the test data. 

// COMMAND ----------

// Create model
val als = new ALS()
.setMaxIter(5)
.setRegParam(0.01)
.setUserCol("user")
.setItemCol("product")
.setRatingCol("rating")

// Fit models
val smallModel = als.fit(trainingSetSmall)
val mediumModel = als.fit(trainingSetMedium)
val largeModel = als.fit(trainingSetLarge)
val xlModel = als.fit(trainingSetXl)

// Predict on the testSets
val smallPredictions = smallModel.transform(testSetSmall).na.drop()
val mediumPredictions = smallModel.transform(testSetMedium).na.drop()
val largePredictions = largeModel.transform(testSetLarge).na.drop()
val xlPredictions = xlModel.transform(testSetXl).na.drop()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Evaluation
// MAGIC 
// MAGIC We calculated Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) for each of the three datasets with the outputs below. 

// COMMAND ----------

// Evaluate models
val evaluatorRMSE = new RegressionEvaluator()
.setMetricName("rmse")
.setLabelCol("rating")
.setPredictionCol("prediction")

val evaluatorMAE = new RegressionEvaluator()
.setMetricName("mae")
.setLabelCol("rating")
.setPredictionCol("prediction")

// Calculate RMSE and MAE on both predictions
val rmseSmall = evaluatorRMSE.evaluate(smallPredictions)
val rmseMedium = evaluatorRMSE.evaluate(mediumPredictions)
val rmseLarge = evaluatorRMSE.evaluate(largePredictions)
val rmseXL = evaluatorRMSE.evaluate(xlPredictions)

val maeSmall = evaluatorMAE.evaluate(smallPredictions)
val maeMedium = evaluatorMAE.evaluate(mediumPredictions)
val maeLarge = evaluatorMAE.evaluate(largePredictions)
val maeXL = evaluatorMAE.evaluate(xlPredictions)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC As you can see there is a significant drop in RMSE and MAE (i.e. increase in accuracy) as the model gets to feed on more data. We observed 23.8% better accuracy for RMSE between the small (100k) and large (1m) datasets. Similary for the same two datasets we saw a 22.0% improvement for MAE.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Next up: Modeling in Python
// MAGIC 
// MAGIC Let's compare Spark's ALS alogrithm with some others from Python's sci-kit surprise library. We have chosen Singular Vector Decomposition (SVD), Non-negative Matrix Factorization (NMF), and k-Nearest Neighbors (KNN). Considering memory constraints we can only run the models on the small (100k), medium (1m), and large (10m) datasets. Each model is run twice to see if we can improve the error score; any more than that and we will encounter memory issues. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Import the data
// MAGIC Interesting to note that some of these files were `.csv` and other were `.dat`, and were separated by `::` instead of a comma, which was difficult to parse in spark. What we did was use pandas to parse the `.dat` file and saved it to Databricks be used in the spark section above.

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC 
// MAGIC # Load data -> small, medium and large ratings data
// MAGIC pdSmall = pd.read_csv("/dbfs/FileStore/tables/ratings_small.csv")
// MAGIC 
// MAGIC 
// MAGIC pdMedium = pd.read_csv("/dbfs/FileStore/tables/ratingsMedium.dat", 
// MAGIC                        sep="::", 
// MAGIC                        names=['userId',  'movieId', 'rating', 'timestamp'])
// MAGIC 
// MAGIC pdLarge = pd.read_csv("/dbfs/FileStore/tables/ratings_large10m.dat", 
// MAGIC                        sep="::", 
// MAGIC                        names=['userId',  'movieId', 'rating', 'timestamp'])
// MAGIC 
// MAGIC #pdXL = pd.read_csv("/dbfs/FileStore/tables/ratings_large.csv")
// MAGIC 
// MAGIC pdMedium.head()

// COMMAND ----------

// MAGIC %python
// MAGIC # Take a quick look at the size of each dataset to make sure our imports are correct
// MAGIC print(pdSmall.shape, pdMedium.shape, pdLarge.shape)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Serialize the data for scikit-surprise
// MAGIC 
// MAGIC The data needs to be formatted a certain way in order for the scikit surprise's algorithms to process them, which is what we are doing in the code chunk below.

// COMMAND ----------

// MAGIC %python
// MAGIC from surprise import Reader
// MAGIC from surprise import Dataset
// MAGIC from surprise import SVD
// MAGIC from surprise import NMF
// MAGIC from surprise import KNNBasic
// MAGIC from surprise.model_selection import cross_validate
// MAGIC 
// MAGIC 
// MAGIC # Convert files back to string values for Scikit-surprise API
// MAGIC pdSmall = pdSmall[['userId', 'movieId', 'rating']]
// MAGIC pdMedium = pdMedium[['userId', 'movieId', 'rating']]
// MAGIC pdLarge = pdLarge[['userId', 'movieId', 'rating']]
// MAGIC 
// MAGIC # Define a reader for a custom dataset
// MAGIC reader = Reader(rating_scale=(1,5))
// MAGIC 
// MAGIC # These datasets are the model-ready 
// MAGIC smallData = Dataset.load_from_df(pdSmall, reader)
// MAGIC mediumData = Dataset.load_from_df(pdMedium, reader)
// MAGIC largeData = Dataset.load_from_df(pdLarge, reader)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Singular Value Decomposition
// MAGIC Equivalent to [Probabilistic Matrix Factorization](http://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)

// COMMAND ----------

// MAGIC %python
// MAGIC import random
// MAGIC random.seed(123)
// MAGIC 
// MAGIC svd = SVD()
// MAGIC 
// MAGIC print("====================")
// MAGIC print("DATASET: SMALL (100K)")
// MAGIC cv_svd_small = cross_validate(svd, smallData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC print("====================" + "\n")
// MAGIC 
// MAGIC print("DATASET: MEDIUM (1M)")
// MAGIC cv_svd_medium = cross_validate(svd, mediumData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC print("====================" + "\n")
// MAGIC 
// MAGIC print("DATASET: LARGE (10M)")
// MAGIC cv_svd_large = cross_validate(svd, largeData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC print("====================")

// COMMAND ----------

// MAGIC %md
// MAGIC #### Non-negtative Matrix Factorization collaborative filtering algorithm. 
// MAGIC Similar to SVD.

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC nmf = NMF()
// MAGIC 
// MAGIC print("====================")
// MAGIC print("DATASET: SMALL (100K)")
// MAGIC cv_nmf_small = cross_validate(nmf, smallData, measures=['RMSE', 'MAE'], cv=2, verbose=True) 
// MAGIC print("====================" + "\n")
// MAGIC 
// MAGIC print("DATASET: MEDIUM (1M)")
// MAGIC cv_nmf_medium = cross_validate(nmf, mediumData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC print("====================" + "\n")
// MAGIC 
// MAGIC print("DATASET: LARGE (10M)")
// MAGIC cv_nmf_large = cross_validate(nmf, largeData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC print("====================")

// COMMAND ----------

// MAGIC %md
// MAGIC #### k-NN basic collaborative filtering algorithm

// COMMAND ----------

// MAGIC %python
// MAGIC # Instantiate KNN algo
// MAGIC knn = KNNBasic()
// MAGIC 
// MAGIC # Cross validate and score model on both datasets
// MAGIC print("====================")
// MAGIC print("DATASET: SMALL (100K)")
// MAGIC cv_knn_small = cross_validate(knn, smallData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC print("====================" + "\n")
// MAGIC 
// MAGIC # print("DATASET: MEDIUM (1M)")
// MAGIC # cv_knn_medium = cross_validate(knn, mediumData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC # print("====================" + "\n")
// MAGIC 
// MAGIC # print("DATASET: LARGE (10M)")
// MAGIC # cv_knn_large = cross_validate(knn, largeData, measures=['RMSE', 'MAE'], cv=2, verbose=True)
// MAGIC # print("====================")

// COMMAND ----------

// MAGIC %md
// MAGIC ### The Results are in...

// COMMAND ----------

// MAGIC %md
// MAGIC #### RMSE Table
// MAGIC | Model  |100K | 1M  | 10M | 20M|
// MAGIC |-------|--------|--------|-----|----|
// MAGIC | ALS  | 1.0797 | 1.5841  |0.8229 | 0.8415|
// MAGIC | SVD  | 0.8908 | 0.9030  |0.8260 | NA|
// MAGIC | NMF  | 0.9577 | 0.9274  |0.8784 |NA|
// MAGIC | KNN  | 0.9769 | NA      |NA     |NA|
// MAGIC 
// MAGIC #### MAE Table
// MAGIC | Model  |100K | 1M  | 10M | 20M|
// MAGIC |-------|--------|--------|-----|----|
// MAGIC | ALS  | 0.8121 | 1.2197 | 0.6337 | 0.6422|
// MAGIC | SVD  | 0.6862 | 0.7115 | 0.6352 | NA|
// MAGIC | NMF  | 0.7357 | 0.7321 |0.6788  |NA|
// MAGIC | KNN  | 0.7496 | NA     |NA      |NA|

// COMMAND ----------

spark.stop()

// COMMAND ----------

// MAGIC %md 
// MAGIC #### References
// MAGIC []()
// MAGIC []()
// MAGIC []()