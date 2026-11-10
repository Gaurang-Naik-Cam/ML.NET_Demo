using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ML.NET_Demo
{
    public class InsuranceData
    {
        [LoadColumn(0)]
        public float Age { get; set; }
        [LoadColumn(1)]
        public float Premium { get; set; }
    }
}
