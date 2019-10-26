using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Version = Mosaik.Core.Version;
using P = Catalyst.PatternUnitPrototype;

namespace Catalyst.Samples.EntityRecognition
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            //This example uses the WikiNER model, trained on the data provided by the paper "Learning multilingual named entity recognition from Wikipedia", Artificial Intelligence 194 (DOI: 10.1016/j.artint.2012.03.006)
            //The training data was sourced from the following repository: https://github.com/dice-group/FOX/tree/master/input/Wikiner

            //Configures the model storage to use the online repository backed by the local folder ./catalyst-models/
            Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));

            //Create a new pipeline for the english language, and add the WikiNER model to it
            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(await AveragePerceptronEntityRecognizer.FromStoreAsync(language: Language.English, version: Version.Latest, tag: "WikiNER"));

            //For processing a single document, you can call nlp.ProcessSingle
            var doc = new Document(Data.Sample_1, Language.English);
            nlp.ProcessSingle(doc);

            //For processing a multiple documents in parallel (i.e. multithreading), you can call nlp.Process on an IEnumerable<IDocument> enumerable
            var docs = nlp.Process(MultipleDocuments());

            //This will print all recognized entities. You can also see how the WikiNER model makes a mistake on recognizing Amazon as a location on Data.Sample_1
            PrintDocumentEntities(doc);
            foreach (var d in docs) { PrintDocumentEntities(d); }

            //For correcting Entity Recognition mistakes, you can use the Neuralyzer class. 
            //This class uses the Pattern Matching entity recognition class to perform "forget-entity" and "add-entity" 
            //passes on the document, after it has been processed by all other proceses in the NLP pipeline
            var neuralizer = new Neuralyzer(Language.English, 0, "WikiNER-sample-fixes");

            //Teach the Neuralyzer class to forget the match for a single token "Amazon" with entity type "Location"
            neuralizer.TeachForgetPattern("Location",  "Amazon", mp => mp.Add(new PatternUnit(P.Single().WithToken("Amazon").WithEntityType("Location"))));
            
            //Teach the Neuralyzer class to add the entity type Organization for a match for the single token "Amazon"
            neuralizer.TeachAddPattern("Organization", "Amazon", mp => mp.Add(new PatternUnit(P.Single().WithToken("Amazon"))));

            //Add the Neuralyzer to the pipeline
            nlp.UseNeuralyzer(neuralizer);

            //Now you can see that "Amazon" is correctly recognized as the entity type "Organization"
            var doc2 = new Document(Data.Sample_1, Language.English);
            nlp.ProcessSingle(doc2);
            PrintDocumentEntities(doc2);
        }

        private static void PrintDocumentEntities(IDocument doc)
        {
            Console.WriteLine($"Input text:\n\t'{doc.Value}'\n\nTokenized Value:\n\t'{doc.TokenizedValue}'\n\nEntities: \n{string.Join("\n", doc.SelectMany(span => span.GetEntities()).Select(e => $"\t{e.Value} [{e.EntityType.Type}]"))}");
        }

        static IEnumerable<IDocument> MultipleDocuments()
        {
            yield return new Document(Data.Sample_2, Language.English);
            yield return new Document(Data.Sample_3, Language.English);
            yield return new Document(Data.Sample_4, Language.English);
        }
    }
}
