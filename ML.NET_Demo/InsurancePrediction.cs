using System;
using Microsoft.ML.Data;

namespace ML.NET_Demo
{
    public class InsurancePrediction
    {
        [ColumnName("Score")]
        public float PredictedPremium { get; set; }
    }
}
