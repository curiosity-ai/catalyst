using System;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using Xunit;
using static Catalyst.WordNet;

namespace Catalyst.Tests
{
    public class WordNetMappingTest
    {
        public WordNetMappingTest()
        {
            Dutch.Register();
        }

        private static async Task<WordNetMapping> GetWordNetMappingAsync()
        {
            return (WordNetMapping)await Dutch.GetWordNetAsync();
        }

        [Theory]
        [InlineData("auto", PartOfSpeech.NOUN, "car")]
        [InlineData("fiets", PartOfSpeech.NOUN, "bicycle")]
        [InlineData("pad", PartOfSpeech.NOUN, "driveway", "path", "frog", "trail")]
        public async void TestGetMapping(string lemma, PartOfSpeech partOfSpeech, params string[] mapping)
        {
            var wordNetMapping = await GetWordNetMappingAsync();
            var actual = wordNetMapping.GetMapping(lemma, partOfSpeech).ToList();
            Assert.Equal(mapping, actual.Select(x => x.Word));
        }

        [Theory]
        [InlineData("auto", PartOfSpeech.NOUN, "kar", "auto", "automobiel", "wagen")]
        [InlineData("praten", PartOfSpeech.VERB, "converseren", "hebben", "praten", "spreken")]
        public async void TestGetSynonyms(string lemma, PartOfSpeech partOfSpeech, params string[] synonyms)
        {
            var wordNetMapping = await GetWordNetMappingAsync();
            var actual = wordNetMapping.GetSynonyms(lemma, partOfSpeech).Select(x => x.Word).ToList();
            Assert.Equal(synonyms, actual);
        }

        [Theory]
        [InlineData("auto", PointerSymbol.Hyponym, "taxi")]
        [InlineData("boot", PointerSymbol.Hypernym, "vaartuig")]
        public async void TestGetPointers(string word, PointerSymbol pointerSymbol, string targetWord)
        {
            var wordNetMapping = await GetWordNetMappingAsync();
            var actual = wordNetMapping.GetPointers(word).ToList();
            Assert.Contains(actual, pointer => pointer.Symbol == pointerSymbol && pointer.TargetWord == targetWord);
        }
    }
}
