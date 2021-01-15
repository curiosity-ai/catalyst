using System;
using System.Threading.Tasks;
using Mosaik.Core;
using Catalyst;

namespace Catalyst.Spacy_Test
{
    class Program
    {
        public static async Task Main(string[] args)
        {
            using (await Spacy.Initialize(Spacy.ModelSize.Small, Language.Any, Language.English))
            {
                var nlp = Spacy.For(Spacy.ModelSize.Small, Language.English);

                var doc = new Document("This is a test of integrating Spacy and Catalyst", Language.English);

                nlp.ProcessSingle(doc);

                Console.WriteLine(doc.ToJson());
            }
        }
    }
}
