using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.Azure.WebJobs.Extensions.Storage;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms.Text;
using ML_NET_Predict_FunctionApp.Model;

namespace ML_NET_Predict_FunctionApp
{
    public static class PredictSentiment
    {
        private static string ModelPath => "C:/Users/kapeltol/Source/Repos/NETMLAppTest/NETMLAppTest/Data/functionmodel.zip"; //Url to blob
        [FunctionName("PredictSentiment")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req,
            [Blob("models/test", FileAccess.Read, Connection = "AzureWebJobsStorage")] Stream serializedModel,
            Binder binder,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string text = req.Query["Text"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            text = text ?? data?.Text;

            //Create MLContext to be shared across the model creation workflow objects 
            //Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            SentimentIssue sampleStatement = new SentimentIssue { Text = text };
            
            ITransformer trainedModel;
            trainedModel = mlContext.Model.Load(serializedModel);
            /*using (var stream = new FileStream(modelblob, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream);
            }*/

            // Create prediction engine related to the loaded trained model
            var predFunction = trainedModel.MakePredictionFunction<SentimentIssue, SentimentPrediction>(mlContext);

            //Score
            var resultprediction = predFunction.Predict(sampleStatement);
            log.LogInformation($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Nice")} sentiment | Probability: {resultprediction.Probability} ");

            return text != null
                ? (ActionResult)new OkObjectResult($"{(Convert.ToBoolean(resultprediction.Prediction) ? "0" : "1")}")
                : new BadRequestObjectResult("Please pass a Text in the request body");
        }
    }
}
