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
        [FunctionName("PredictSentiment")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous,"post", Route = null)] HttpRequest req,
            [Blob("models/demomodel", FileAccess.Read)] Stream serializedModel,
            Binder binder,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            string text = data?.Text;

            if (text == null)
                return new BadRequestObjectResult("Please pass sentiment text [text] in the request body");

            //Create MLContext to be shared across the model creation workflow objects 
            //Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            SentimentIssue sampleStatement = new SentimentIssue { Text = text };
            
            ITransformer trainedModel = mlContext.Model.Load(serializedModel);

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
