using Mosaik.Core;
using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.Training
{
    public static class TrainLanguageDetector
    {
        public static void Train(string pathToSentences)
        {
            var docs = File.ReadAllLines(pathToSentences).Shuffle()
                           .Where(txt => !string.IsNullOrWhiteSpace(txt))
                           .Select(txt => txt.Split('\t'))
                           .Where(s => s.Length == 3 && Languages.IsValid3LetterCode(s[1]))
                           .Select(s =>
                           {
                               var doc = new Document(s[2]);
                               doc.Labels.Add(Languages.EnumToCode(Languages.ThreeLetterCodeToEnum(s[1])));
                               return doc as IDocument;
                           });


            var pipe = new Pipeline();
            pipe.Add(new SpaceTokenizer());

            var ldm = new FastTextLanguageDetector(0);
            ldm.Train(pipe.Process(docs).WithCaching(Language.Any,0,"language-detector-corpus", 100_000));
            ldm.StoreAsync().Wait();
        }

        public static void Test(string pathToSentences)
        {
            var vectorizer = FastTextLanguageDetector.FromStoreAsync(Language.Any, 0, null).WaitResult();

            var pipe = new Pipeline();
            pipe.Add(new SpaceTokenizer());


            var docs = File.ReadAllLines(pathToSentences).Shuffle()
               .Where(txt => !string.IsNullOrWhiteSpace(txt))
               .Select(txt => txt.Split('\t'))
               .Where(s => s.Length == 3 && Languages.IsValid3LetterCode(s[1]))
               //.GroupBy(l => l[1])
               //.Where(g => g.Count() > 10_000)
               //.SelectMany(g => g)
               .Select(s =>
               {
                   var doc = new Document(s[2]);
                   doc.Labels.Add(Languages.EnumToCode(Languages.ThreeLetterCodeToEnum(s[1])));
                   return doc as IDocument;
               });


            docs = pipe.Process(docs).WithCaching(Language.Any, 0, "language-detector-corpus", 100_000).ToList();


            int TP = 0, FP = 0, FN = 0;
            int k = 0;
            var sw = Stopwatch.StartNew();
            Parallel.ForEach(docs, (doc) =>
            {
                k++;
                vectorizer.Process(doc);
                if (doc.Language == Languages.CodeToEnum(doc.Labels.First()))
                {
                    Interlocked.Increment(ref TP);
                }
                else
                {
                    Interlocked.Increment(ref FP);
                    Interlocked.Increment(ref FN);
                }
            });

            sw.Stop();

            var precision = (double)TP / (double)(TP + FN);
            var recall = (double)TP / (double)(TP + FN);

            var f1 = 2 * (precision * recall) / (precision + recall);

            Console.WriteLine($"F1= {f1 * 100:0.0}% P= {precision * 100:0.0}% R={recall * 100:0.0}% in {sw.Elapsed.TotalSeconds:0.00}s or {k / sw.Elapsed.TotalSeconds} doc/s");
        }
    }
}
