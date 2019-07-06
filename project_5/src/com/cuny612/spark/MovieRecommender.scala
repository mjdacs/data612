package com.cuny612.spark


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS



object MovieRecommender {
  def main(args: Array[String]) {
    
    // Initiate session
    val spark = SparkSession
    .builder
    .appName("MovieLens Recommendation")
    .master("local[*]")
    .getOrCreate()
      
    // Path to data
    val ratingsFile = "/Users/deborahgemellaro/Programming/612/project_5/ratings.csv"
    val moviesFile = "/Users/deborahgemellaro/Programming/612/project_5/movies.csv"
  
    
    // Load data as Spark DataFrame  
    val df1 = spark.read.format("csv")
      .option("header", true)
      .load(ratingsFile)
           
    val df2 = spark.read.format("csv")
      .option("header", true)
      .load(moviesFile)
      
    // Select appropriate columns
    val ratingsDF = df1.select(df1.col("userId"), 
                           df1.col("movieId"), 
                           df1.col("rating"), 
                           df1.col("timestamp"))
 
    val moviesDF = df2.select(df2.col("movieId"), 
                          df2.col("title"),
                          df2.col("genres"))
                            
    
    // Create your training and test data                      
    val splits = ratingsDF.randomSplit(Array(0.75, 0.25), seed = 123L)
    val (trainingData, testData) = (splits(0), splits(1))
  
    // Print the counts
    println("Counts for -- Training Data: " + trainingData.count() + ", Test Data: " + testData.count())                 
    
    // spark implicits needed here for serializing 
    import spark.implicits._
    
    // Convert training data from strings to numerics 
    val trainingSet = trainingData.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
  })
    
    // Convert test data from strings to numerics
    val testSet = testData.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
  })
    
    // Create model
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("user")
      .setItemCol("product")
      .setRatingCol("rating")
  
    // Fit model
    val model = als.fit(trainingSet)
    
    // Predict on the testSet
    val predictions = model.transform(testSet).na.drop()
    
    // Evaluate model
    val evaluatorRMSE = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("rating")
        .setPredictionCol("prediction")
    
    val evaluatorMSE = new RegressionEvaluator()
        .setMetricName("mse")
        .setLabelCol("rating")
        .setPredictionCol("prediction")
    
    val evaluatorMAE = new RegressionEvaluator()
        .setMetricName("mae")
        .setLabelCol("rating")
        .setPredictionCol("prediction")
    
    val rmse = evaluatorRMSE.evaluate(predictions)
    val mse = evaluatorMSE.evaluate(predictions)
    val mae = evaluatorMAE.evaluate(predictions)
    
    println(f"RMSE = $rmse")
    println(f"MSE = $mse")
    println(f"MAE = $mae")
  
  spark.stop()

  } 
}