using System;
using System.Threading.Tasks;
using Mosaik.Core;
using Catalyst;
using System.Linq;
using Catalyst.Models;

namespace Catalyst.Spacy_Test
{
    class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("===SYNONYMS FOR CAR===");
            foreach (var syn in WordNet.Nouns.GetSynonyms("car").OrderBy(w => w.Word))
            {
                Console.WriteLine(syn);
            }

            Console.WriteLine("===POINTERS FOR CAR===");
            foreach (var pointer in WordNet.Nouns.GetPointers("car").OrderBy(w => w.Word))
            {
                Console.WriteLine(pointer);
            }

            //
            // Dutch Examples
            Dutch.Register();

            var dutchWordNet = await Dutch.GetWordNetAsync();
            Console.WriteLine("===DUTCH SYNONYMS FOR CAR===");
            foreach (var syn in dutchWordNet.Nouns.GetSynonyms("auto").OrderBy(w => w.Word))
            {
                Console.WriteLine(syn);
            }

            Console.WriteLine("===DUTCH POINTERS FOR CAR===");
            foreach (var pointer in dutchWordNet.Nouns.GetPointers("auto").OrderBy(w => w.Word))
            {
                Console.WriteLine(pointer);
            }

            //
            // French Examples
            French.Register();

            var frenchWordNet = await French.GetWordNetAsync();
            Console.WriteLine("===French SYNONYMS FOR CAR===");
            foreach (var syn in frenchWordNet.Nouns.GetSynonyms("voiture").OrderBy(w => w.Word))
            {
                Console.WriteLine(syn);
            }

            Console.WriteLine("===French POINTERS FOR CAR===");
            foreach (var pointer in frenchWordNet.Nouns.GetPointers("voiture").OrderBy(w => w.Word))
            {
                Console.WriteLine(pointer);
            }
        }
    }
}
