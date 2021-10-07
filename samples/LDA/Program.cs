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

namespace Catalyst.Samples
{
    class Program
    {
        static async Task Main(string[] args)
        {

            Console.OutputEncoding = Encoding.UTF8;
            ApplicationLogging.SetLoggerFactory(LoggerFactory.Create(lb => lb.AddConsole()));

            //Need to register the languages we want to use first
            Catalyst.Models.English.Register();

            //Configures the model storage to use the local folder ./catalyst-models/
            Storage.Current = new DiskStorage("catalyst-models");

            //Download the Reuters corpus if necessary
            var (train, test) = await Corpus.Reuters.GetAsync();

            //Parse the documents using the English pipeline, as the text data is untokenized so far
            var nlp = Pipeline.For(Language.English);

            var trainDocs = nlp.Process(train).ToArray();
            var testDocs = nlp.Process(test).ToArray();


            //Train an LDA topic model on the trainind dateset
            using (var lda = new LDA(Language.English, 0, "reuters-lda"))
            {
                lda.Data.NumberOfTopics = 20; //Arbitrary number of topics
                lda.Train(trainDocs, Environment.ProcessorCount);
                await lda.StoreAsync();
            }

            using (var lda = await LDA.FromStoreAsync(Language.English, 0, "reuters-lda"))
            {
                foreach (var doc in testDocs)
                {
                    if (lda.TryPredict(doc, out var topics))
                    {
                        var docTopics = string.Join("\n", topics.Select(t => lda.TryDescribeTopic(t.TopicID, out var td) ? $"[{t.Score:n3}] => {td.ToString()}" : ""));

                        Console.WriteLine("------------------------------------------");
                        Console.WriteLine(doc.Value);
                        Console.WriteLine("------------------------------------------");
                        Console.WriteLine(docTopics);
                        Console.WriteLine("------------------------------------------\n\n");
                    }
                }
            }
        }
    }
}
