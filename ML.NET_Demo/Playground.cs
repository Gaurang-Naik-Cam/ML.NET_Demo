using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.NET_Demo
{
    internal class Playground
    {
        /// <summary>
        /// Trains an ordinary least squares regression model to predict an insurance premium from age, then writes the
        /// prediction to the console.
        /// </summary>
        /// <param name="age">The age used to predict the insurance premium. Defaults to 0.</param>
        internal static void Lab1(int age = 0)
        {            
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable<InsuranceData>(DataRegression.GetInsuranceData());

            //pipeline where we define FLAT
            var pipeline = mlContext.Transforms
                                    .Concatenate("Features", "Age") //Can add more columns here like BP1, BP2, IsDiebetic, etc
                                    .Append(mlContext.Regression.Trainers.Ols( labelColumnName: "Premium", featureColumnName: "Features"));
            var model = pipeline.Fit(data);
            var pe = mlContext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);

            var p = pe.Predict(new InsuranceData() { Age = age });
            Console.WriteLine($"The predicted premium for age {age} is {p.PredictedPremium}");
        }

        /// <summary>
        /// Fits an ordinary least squares regression model to the insurance data and writes the predicted premium to
        /// the console. Also calculates the slope (weight) and intercept (bias) of the regression line and writes them to the console.
        /// </summary>
        /// <param name="age">The age to use when predicting the insurance premium.</param>
        internal static void Lab2(int age)
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromEnumerable<InsuranceData>(DataRegression.GetInsuranceData());
            //pipeline where we define FLAT
            var pipeline = mlContext.Transforms
                                    .Concatenate("Features", "Age") //Can add more columns here like BP1, BP2, IsDiebetic, etc
                                    .Append(mlContext.Regression.Trainers.Ols(labelColumnName: "Premium", featureColumnName: "Features"));
            var model = pipeline.Fit(data); //training the model
            var pe = mlContext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var p = pe.Predict(new InsuranceData() { Age = age }); //inference 
            Console.WriteLine($"The predicted premium for age {age} is {p.PredictedPremium}");

            var weights = model.LastTransformer.Model.Weights;
            var bias = model.LastTransformer.Model.Bias;

            Console.WriteLine($"The slope (weight) of the regression line is {weights[0]}");
            Console.WriteLine($"The intercept (bias) of the regression line is {bias}");
        }
    }
}
