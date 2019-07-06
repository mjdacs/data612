package com.cuny612.spark


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import sqlContext.implicits._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
import org.apache.spark.rdd.RDD


object MovieRecommender {
  def main(args: Array[String]) {
    
    println("testing object")
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val sc = new SparkContext("local[*]", "MovieRecommender")
    
    val lines = sc.textFile("/Users/deborahgemellaro/Programming/612/project_5/ml-latest-small/ratings.csv")
    
    val ratings = lines.map(x => x.toString().split(",")(2))
    
    val results = ratings.countByValue()
    
    val sortedResults = results.toSeq.sortBy(_._1)
    lines.take(6).foreach(println)
   
  
    val df = lines.toDF()
    
  }
}