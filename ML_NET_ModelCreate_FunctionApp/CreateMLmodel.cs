using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms.Text;

namespace ML_NET_ModelCreate_FunctionApp
{
    public static class CreateMLmodel
    {

        [FunctionName("CreateMLmodel")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req,
            [Blob("models/demomodel", FileAccess.Write)] Stream modelstream,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processesing a request.");

            //Create MLContext to be shared across the model creation workflow objects 
            //Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            var reader = new TextLoader(mlContext,
                            new TextLoader.Arguments()
                            {
                                Separator = "tab",
                                HasHeader = true,
                                Column = new[]
                                {
                                    new TextLoader.Column("Label", DataKind.Bool, 0),
                                    new TextLoader.Column("Text", DataKind.Text, 1)
                                }
                            });
                        
            log.LogInformation("Reading training data...");
            //var trainingDataView = reader.Read("D:/home/site/wwwroot/Data/wikipedia-detox-250-line-data.tsv");
            var trainingDataView = reader.Read("D:/home/site/wwwroot/Data/wikipedia-detox-250-line-dataSentiment_tweets.tsv");
            log.LogInformation("Done loading training data.");


            // STEP 2: Common data process configuration with pipeline data transformations          
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Text", "Features");


            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(label: "Label", features: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            log.LogInformation("Starting training...");

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
            log.LogInformation("Done!");

            // STEP 5: Evaluate model with training set
            /*IDataView testDataView = reader.Read("D:/home/site/wwwroot/Data//Data/wikipedia-detox-250-line-test.tsv");
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label", "Score");
            log.LogInformation($"Model's Accuracy: {metrics.Accuracy:P2}");*/

            // STEP 6: Save/persist the trained model to blob
            mlContext.Model.Save(trainedModel, modelstream);
            log.LogInformation("Saved model to blob.");

            return (ActionResult)new OkObjectResult($"Model trained sucessfully!");
        }

    }
}
