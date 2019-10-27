using System;
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
            fastText.Data.LearningRate = 0.7f;
            fastText.Data.Dimensions = 128;
            fastText.Data.Epoch = 100;
            fastText.Data.MinimumWordNgramsCounts = 5;
            fastText.Data.MaximumWordNgrams = 3;

            fastText.Train(trainDocs);


            //Compute precision and Recall for the Train Set
            float TP = 0f, FP = 0f, R = 0f, cutoff = 0.5f;

            foreach (var doc in trainDocs)
            {
                var pred = fastText.Predict(doc).Where(kv => kv.Value >cutoff).ToArray();
                foreach (var p in pred)
                {
                    if (doc.Labels.Contains(p.Key))
                    {
                        TP++;
                    }
                    else
                    {
                        FP++;
                    }
                }

                R += doc.Labels.Count;
            }

            var precision = TP / (TP + FP);
            var recall = TP / (R);

            Console.WriteLine($"Train: F1={2 * (precision * recall) / (precision + recall):n2} P={precision:n2} R={recall:n2}");

            //Compute precision and Recall for the Test Set
            TP = 0f, FP = 0f, R = 0f,
            foreach (var doc in testDocs)
            {
                var pred = fastText.Predict(doc).Where(kv => kv.Value > cutoff).ToArray();
                foreach(var p in pred)
                {
                    if (doc.Labels.Contains(p.Key))
                    {
                        TP++;
                    }
                    else
                    {
                        FP++;
                    }
                }

                R += doc.Labels.Count;
            }

            precision = TP / (TP + FP);
            recall    = TP / (R);

            Console.WriteLine($"Test: F1={2*(precision*recall)/(precision+recall):n2} P={precision:n2} R={recall:n2}");
            Console.ReadLine();
        }
    }
}
