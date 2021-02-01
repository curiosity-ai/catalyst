using System;
using System.Threading.Tasks;
using Mosaik.Core;
using Catalyst;
using System.Linq;

namespace Catalyst.Spacy_Test
{
    class Program
    {
        public static async Task Main(string[] args)
        {
            foreach (var syn in WordNet.Nouns.GetSynonyms("car").OrderBy(w => w.Word))
            {
                Console.WriteLine(syn);
            }

            foreach (var pointer in WordNet.Nouns.GetPointers("car").OrderBy(w => w.Word))
            {
                Console.WriteLine(pointer);
            }
        }
    }
}
