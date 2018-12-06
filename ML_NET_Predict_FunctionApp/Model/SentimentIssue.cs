using System;
using System.Collections.Generic;
using System.Text;

namespace ML_NET_Predict_FunctionApp.Model
{
    public class SentimentIssue
    {
        public bool Label { get; set; }
        public string Text { get; set; }
    }
}
