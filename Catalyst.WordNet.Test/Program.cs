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
            foreach (var syn in WordNet.Nouns.GetSynonyms("car"))
            {
                Console.WriteLine(syn);
            }

            foreach (var pointer in WordNet.Nouns.GetPointers("car"))
            {
                Console.WriteLine(pointer);
            }
        }
    }
}
