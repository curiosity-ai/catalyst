using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Version = Mosaik.Core.Version;
using P = Catalyst.PatternUnitPrototype;
using System.Text;

namespace Catalyst.Samples.LanguageDetection
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            ApplicationLogging.SetLoggerFactory(LoggerFactory.Create(lb => lb.AddConsole()));

            //This example shows the two language detection models available on Catalyst. 
            //The first is derived from the Chrome former language detection code Compact Language Detector 2 (https://github.com/CLD2Owners/cld2)
            //and the newer model is derived from Facebook's FastText language detection dataset (see: https://fasttext.cc/blog/2017/10/02/blog-post.html)

            //Configures the model storage to use the online repository backed by the local folder ./catalyst-models/
            Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));

            Console.WriteLine("Loading models... This might take a bit longer the first time you run this sample, as the models have to be downloaded from the online repository");
            var cld2LanguageDetector     = await LanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");
            var fastTextLanguageDetector = await FastTextLanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");

            //We show bellow the detection on short and longer samples. You can expect lower precision on shorter texts, as there is less information for the model to work with
            //It's also interesting to see the kind of mistakes these models make, such as detecting Welsh as Gaelic_Scottish_Gaelic

            foreach (var (lang, text) in Data.ShortSamples)
            {
                var doc = new Document(text);
                fastTextLanguageDetector.Process(doc);

                var doc2 = new Document(text);
                cld2LanguageDetector.Process(doc2);

                Console.WriteLine(text);
                Console.WriteLine($"Actual:\t{lang}\nFT:\t{doc.Language}\nCLD2\t{doc2.Language}");
                Console.WriteLine();
            }

            foreach (var (lang, text) in Data.LongSamples)
            {
                var doc = new Document(text);
                fastTextLanguageDetector.Process(doc);

                var doc2 = new Document(text);
                cld2LanguageDetector.Process(doc2);

                Console.WriteLine(text);
                Console.WriteLine($"Actual:\t{lang}\nFT:\t{doc.Language}\nCLD2\t{doc2.Language}");
                Console.WriteLine();
            }

            // You can also access all predictions via the Predict method:
            var allPredictions = fastTextLanguageDetector.Predict(new Document(Data.LongSamples[Language.Spanish]));
            
            Console.WriteLine($"\n\nTop 10 predictions and scores for the Spanish sample:");
            foreach (var kv in allPredictions.OrderByDescending(kv => kv.Value).Take(10))
            {
                Console.WriteLine($"{kv.Key.ToString().PadRight(40)}\tScore: {kv.Value:n2}");
            }
        }
    }
}
