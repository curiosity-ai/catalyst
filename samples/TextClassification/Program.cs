using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst;
using Catalyst.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Mosaik.Core;

namespace TextClassification
{
    class Program
    {
        private static ILogger Logger = ApplicationLogging.CreateLogger<Program>();

        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            ApplicationLogging.SetLoggerFactory(LoggerFactory.Create(lb => lb.AddConsole()));

            //Configures the model storage to use the online repository backed by the local folder ./catalyst-models/
            Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));


            //Download the Reuters corpus if necessary
            var (train, test) = await Corpus.Reuters.GetAsync();

            //Parse the documents using the English pipeline, as the text data is untokenized so far
            var nlp = Pipeline.For(Language.English);

            var trainDocs = nlp.Process(train).ToArray();
            var testDocs  = nlp.Process(test).ToArray();

            //Train a FastText supervised classifier with a multi-label loss (OneVsAll)
            var fastText = new FastText(Language.English, 0, "Reuters-Classifier");

            fastText.Data.Type = FastText.ModelType.Supervised;
            fastText.Data.Loss = FastText.LossType.OneVsAll;
            fastText.Data.LearningRate = 1f;
            fastText.Data.Dimensions = 256;
            fastText.Data.Epoch = 100;
            fastText.Data.MinimumWordNgramsCounts = 5;
            fastText.Data.MaximumWordNgrams = 3;
            fastText.Data.MinimumCount = 5;

            fastText.Train(trainDocs);

            //You can also auto-tune the model using the algorithm from https://ai.facebook.com/blog/fasttext-blog-post-open-source-in-brief/
            fastText.AutoTuneTrain(trainDocs, testDocs, new FastText.AutoTuneOptions());

            //Compute predictions
            Dictionary<IDocument, Dictionary<string, float>> predTrain, predTest;
            using (new Measure(Logger, "Computing train-set predictions", trainDocs.Length))
            {
                predTrain = trainDocs.AsParallel().Select(d => (Doc: d, Pred: fastText.Predict(d))).ToDictionary(d => d.Doc, d => d.Pred);
            }

            using (new Measure(Logger, "Computing test set predictions", testDocs.Length))
            {
                predTest = testDocs.AsParallel().Select(d => (Doc: d, Pred: fastText.Predict(d))).ToDictionary(d => d.Doc, d => d.Pred);
            }

            var resultsTrain = ComputeStats(predTrain);
            var resultsTest  = ComputeStats(predTest);

            Console.WriteLine("\n\n\n--- Results ---\n\n\n");
            foreach (var res in resultsTrain.Zip(resultsTest))
            {
                Console.WriteLine($"\tScore cutoff: {res.First.Cutoff:n2} Train: F1={res.First.F1:n2} P={res.First.Precision:n2} R={res.First.Recall:n2} Test: F1={res.Second.F1:n2} P={res.Second.Precision:n2} R={res.Second.Recall:n2}");
            }

            Console.ReadLine();
        }

        private static (float Cutoff, float Precision, float Recall, float F1)[] ComputeStats(Dictionary<IDocument, Dictionary<string, float>> predictions)
        {
            return Enumerable.Range(40, 20)
                             .Select(i => i / 100f) //Cutoff range: 0.4f to 0.6f
                             .Select(cutoff =>
                                (Cutoff: cutoff,
                                Predictions:  predictions.Select(kv =>
                                     {
                                         return (TruePositive: kv.Key.Labels.Count(lbl => kv.Value.TryGetValue(lbl, out var score) && score > cutoff),
                                                 FalsePositive: kv.Value.Count(pred => pred.Value > cutoff && !kv.Key.Labels.Contains(pred.Key)),
                                                 Count: kv.Key.Labels.Count);
                                     })
                                     .ToArray()
                                )
                             )
                             .Select(results =>
                             {
                                 var TP = (float)results.Predictions.Sum(r => r.TruePositive);
                                 var FP = (float)results.Predictions.Sum(r => r.FalsePositive);
                                 var precision = TP / (TP + FP);
                                 var recall = TP / (results.Predictions.Sum(r => r.Count));
                                 return (Cutoff: results.Cutoff, Precision: precision, Recall: recall, F1: 2 * (precision * recall) / (precision + recall));
                             })
                             .ToArray();
        }
    }
}
