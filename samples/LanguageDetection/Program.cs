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
            //This example shows the two language detection models available on Catalyst. 
            //The first is derived from the Chrome language detection code
            //and the second from Facebook's FastText language detection model

            //Configures the model storage to use the online repository backed by the local folder ./catalyst-models/
            Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));

            //var chLangDetect = await LanguageDetector.FromStoreAsync(Language.Any, Version.Latest, ""); // Missing model training code

            var ftLangDetect = await FastTextLanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");

            //We show bellow the detection on short and longer samples. You can expect lower precision on shorter texts, as there is less information for the model to work with
            //It's also interesting to see the kind of mistakes these models make, such as detecting Welsh as Gaelic_Scottish_Gaelic

            foreach (var (lang, text) in Data.ShortSamples)
            {
                var doc = new Document(text);
                ftLangDetect.Process(doc);
                Console.WriteLine($"{text}\n\nActual\t\t{lang}\nPredicted\t{doc.Language}\n\n");
            }

            foreach (var (lang, text) in Data.LongSamples)
            {
                var doc = new Document(text);
                ftLangDetect.Process(doc);
                Console.WriteLine($"{text}\n\nActual\t\t{lang}\nPredicted\t{doc.Language}\n\n");
            }

            // You can also access all predictions via the Predict method:
            var allPredictions = ftLangDetect.Predict(new Document(Data.LongSamples[Language.English]));
            
            Console.WriteLine($"\n\nTop 10 predictions and scores for the English sample:");
            foreach (var kv in allPredictions.OrderByDescending(kv => kv.Value).Take(10))
            {
                Console.WriteLine($"{kv.Key.PadRight(40)}\tScore: {kv.Value:n2}");
            }
        }
    }
}
