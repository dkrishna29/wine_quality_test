import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class RandomForestPrediction {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .master("local")
                .getOrCreate();

        // Load validation dataset
        Dataset<Row> val = spark.read().format("csv").option("header", "true").option("sep", ";").load(args[0]);
        val.printSchema();
        val.show();

        // Change the 'quality' column name to 'label'
        for (String colName : val.columns()) {
            val = val.withColumnRenamed(colName, colName.replace("\"", ""));
        }
        val = val.withColumnRenamed("quality", "label");

        // Convert features and labels to Java lists
        List<double[]> featuresList = new ArrayList<>();
        List<Double> labelList = new ArrayList<>();
        for (Row row : val.collectAsList()) {
            double[] features = new double[val.columns().length - 1];
            for (int i = 1; i < val.columns().length; i++) {
                features[i - 1] = Double.parseDouble(row.getString(i));
            }
            featuresList.add(features);
            labelList.add(row.getDouble(0));
        }

        // Create the feature vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(val.columns())
                .setOutputCol("features");
        Dataset<Row> dfTr = assembler.transform(val).select("features", "label");

        // Create labeled points and convert to RDD
        List<LabeledPoint> labeledPoints = new ArrayList<>();
        for (int i = 0; i < featuresList.size(); i++) {
            Vector featuresVector = new DenseVector(featuresList.get(i));
            LabeledPoint labeledPoint = new LabeledPoint(labelList.get(i), featuresVector);
            labeledPoints.add(labeledPoint);
        }
        JavaRDD<LabeledPoint> dataset = spark.sparkContext().parallelize(labeledPoints);

        // Load the model from S3
        RandomForestRegressionModel rfModel = RandomForestRegressionModel.load("/winepredict/trainingmodel.model/");

        System.out.println("Model loaded successfully");

        // Make predictions
        JavaRDD<Tuple2<Object, Object>> predictionsAndLabels = dataset.map(lp ->
                new Tuple2<>(rfModel.predict(lp.features()), lp.label()));

        // Convert RDD to DataFrame and then to Pandas DataFrame
        Dataset<Row> labelPred = spark.createDataFrame(predictionsAndLabels, Tuple2.class).toDF("Prediction", "label");
        labelPred.show();
        List<Row> labelPredRows = labelPred.collectAsList();

        // Calculate F1 score
        MulticlassMetrics metrics = new MulticlassMetrics(predictionsAndLabels.rdd());
        double f1Score = metrics.weightedFMeasure();
        System.out.println("F1-score: " + f1Score);

        // Print confusion matrix and other metrics
        System.out.println(metrics.confusionMatrix());
        System.out.println(metrics.toString());

        // Calculate accuracy
        double accuracy = metrics.accuracy();
        System.out.println("Accuracy: " + accuracy);

        // Calculate test error
        double testErr = predictionsAndLabels.filter(lp -> !lp._1().equals(lp._2())).count() / (double) dataset.count();
        System.out.println("Test Error = " + testErr);
    }
}
