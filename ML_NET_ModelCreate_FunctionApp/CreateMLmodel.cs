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
using ML_NET_ModelCreate_FunctionApp.Model;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms.Text;
using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Blob;
using System;
using System.IO;
using System.Threading.Tasks;

namespace ML_NET_ModelCreate_FunctionApp
{
    public static class CreateMLmodel
    {
        private static string ModelPath => "C:/Users/kapeltol/Source/Repos/NETMLAppTest/NETMLAppTest/Data/functionmodel.zip"; //Url to blob

        [FunctionName("CreateMLmodel")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req,
            [Blob("models/test", FileAccess.Write)] Stream modelstream,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            var file = data?.fileUrl;
            var modelname = data?.name;

            if( file == null || modelname == null )
                return new BadRequestObjectResult("Please pass trainingdata file url and new model name in the request body");

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

            //Load training data from url given in request
            var trainingDataView = reader.Read(new MultiFileSource("C:/Users/kapeltol/Source/Repos/NETMLAppTest/NETMLAppTest/Data/wikipedia-detox-250-line-data.tsv")); // file));

          
            // STEP 2: Common data process configuration with pipeline data transformations          
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Text", "Features");

            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(label: "Label", features: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

            IDataView testDataView = reader.Read("C:/Users/kapeltol/Source/Repos/NETMLAppTest/NETMLAppTest/Data/wikipedia-detox-250-line-test.tsv");
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label", "Score");
            log.LogInformation($"Model's Accuracy: {metrics.Accuracy:P2}");

            // STEP 6: Save/persist the trained model to a .ZIP file
            //using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                //mlContext.Model.Save(trainedModel, fs);

            //modelstream
            mlContext.Model.Save(trainedModel, modelstream);

            
            return (ActionResult)new OkObjectResult($"Model created sucessfully with name: {(modelname)}");
        }

    }
}
