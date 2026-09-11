using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    // Checks the spotters' matching walk against a brute-force oracle over randomly generated models and
    // documents. The walk is the part of the model that changed shape - one sorted dictionary of whole
    // entries instead of a hash set plus one hash set per word position - so multi-token capture is what
    // needs proving, not single-token capture.
    //
    // The contract being pinned, which is the one the hash-table implementation had:
    //  - at each position the longest entry starting there wins, and the walk resumes after its last token;
    //  - a token that both starts a longer entry and is an entry on its own carries Begin and Single;
    //  - the walk only extends while some stored entry continues past what it has.
    public class SpotterMatchingOracleTests
    {
        private const int VOCABULARY = 26;

        private static string Word(int i) => "w" + i.ToString("00");

        // Words that are prefixes of one another, so a lookup cannot get away with comparing only as far as
        // the shorter one - the case a fixed-width vocabulary never produces.
        private static readonly string[] PREFIX_VOCABULARY = { "a", "ab", "abc", "abcd", "b", "ba", "bab", "baba", "c", "ca", "cab", "cabb", "x", "xy", "xyz" };

        private static async Task<Pipeline> PlainPipelineAsync()
        {
            English.Register();
            return await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
        }

        private static List<(int index, EntityTag tag)> ActualTags(IDocument doc, string captureTag)
        {
            var tags   = new List<(int, EntityTag)>();
            var tokens = doc.SelectMany(s => s.Tokens).ToArray();

            for (int i = 0; i < tokens.Length; i++)
            {
                foreach (var entity in tokens[i].EntityTypes.Where(e => e.Type == captureTag))
                {
                    tags.Add((i, entity.Tag));
                }
            }
            return tags;
        }

        // The same decisions the engine makes, written the slow obvious way.
        private static List<(int index, EntityTag tag)> ExpectedTags(string[] tokens, HashSet<string> entries)
        {
            var expected = new List<(int, EntityTag)>();
            var prefixes = new HashSet<string>();

            foreach (var entry in entries)
            {
                var words = entry.Split(' ');
                for (int n = 1; n < words.Length; n++) { prefixes.Add(string.Join(" ", words.Take(n))); }
            }

            for (int i = 0; i < tokens.Length; i++)
            {
                var key       = tokens[i];
                bool single   = entries.Contains(key);
                bool extend   = prefixes.Contains(key);
                int  last     = i;
                int  j        = i;

                while (extend && j + 1 < tokens.Length)
                {
                    j++;
                    key = key + " " + tokens[j];
                    if (entries.Contains(key)) { last = j; }
                    extend = prefixes.Contains(key);
                }

                if (last > i)
                {
                    expected.Add((i, EntityTag.Begin));
                    for (int m = i + 1; m < last; m++) { expected.Add((m, EntityTag.Inside)); }
                    expected.Add((last, EntityTag.End));
                }

                if (single) { expected.Add((i, EntityTag.Single)); }

                i = last;
            }

            // The engine tags Begin/Inside/End as it walks and Single afterwards, so order by token index and
            // compare as a set per index rather than by discovery order.
            return expected.OrderBy(t => t.Item1).ThenBy(t => (int)t.Item2).ToList();
        }

        private static (HashSet<string> entries, List<string[]> documents) Generate(Random rng, int entryCount, int documentCount)
        {
            var entries = new HashSet<string>();
            while (entries.Count < entryCount)
            {
                int words = rng.Next(1, 5);
                var entry = string.Join(" ", Enumerable.Range(0, words).Select(_ => Word(rng.Next(VOCABULARY))));
                entries.Add(entry);
            }

            var documents = new List<string[]>(documentCount);
            for (int d = 0; d < documentCount; d++)
            {
                int length = rng.Next(4, 25);
                documents.Add(Enumerable.Range(0, length).Select(_ => Word(rng.Next(VOCABULARY))).ToArray());
            }

            return (entries, documents);
        }

        [Fact]
        public async Task Spotter_MatchesTheOracleOverGeneratedModelsAndDocuments()
        {
            int multiTokenMatches = 0;

            for (int seed = 0; seed < 25; seed++)
            {
                var rng = new Random(1000 + seed);
                var (entries, documents) = Generate(rng, entryCount: 60, documentCount: 20);

                var spotter = new Spotter(Language.English, 0, "", "Entity");
                foreach (var entry in entries) { spotter.AddEntry(entry); }

                var pipeline = await PlainPipelineAsync();
                pipeline.Add(spotter);

                foreach (var tokens in documents)
                {
                    var text = string.Join(" ", tokens);
                    var doc  = new Document(text, Language.English);
                    pipeline.ProcessSingle(doc);

                    var actual   = ActualTags(doc, "Entity").OrderBy(t => t.index).ThenBy(t => (int)t.tag).ToList();
                    var expected = ExpectedTags(tokens, entries);

                    Assert.Equal(expected, actual);
                    multiTokenMatches += actual.Count(t => t.tag == EntityTag.Begin);
                }
            }

            // Guards the test against becoming vacuous: it has to be finding multi-token entries, not just
            // agreeing that nothing matched.
            Assert.True(multiTokenMatches > 100, $"only {multiTokenMatches} multi-token matches were exercised");
        }

        [Fact]
        public async Task LinkedSpotter_MatchesTheOracleAndCarriesTheRightUID()
        {
            for (int seed = 0; seed < 15; seed++)
            {
                var rng = new Random(5000 + seed);
                var (entries, documents) = Generate(rng, entryCount: 50, documentCount: 15);

                var uids    = entries.ToDictionary(e => e, _ => UID128.New());
                var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
                foreach (var entry in entries) { spotter.AddEntry(entry, uids[entry]); }

                var pipeline = await PlainPipelineAsync();
                pipeline.Add(spotter);

                foreach (var tokens in documents)
                {
                    var text = string.Join(" ", tokens);
                    var doc  = new Document(text, Language.English);
                    pipeline.ProcessSingle(doc);

                    var actual   = ActualTags(doc, "Linked").OrderBy(t => t.index).ThenBy(t => (int)t.tag).ToList();
                    var expected = ExpectedTags(tokens, entries);
                    Assert.Equal(expected, actual);

                    // Every captured entity must carry the UID registered for exactly that surface form.
                    foreach (var entity in doc.SelectMany(s => s.GetEntities()).Where(e => e.EntityType.Type == "Linked"))
                    {
                        Assert.True(uids.TryGetValue(entity.Value, out var expectedUid), $"captured '{entity.Value}' which is not an entry");
                        Assert.Equal(expectedUid, entity.EntityType.TargetUID);
                    }
                }
            }
        }

        [Fact]
        public async Task Spotter_MatchesTheOracleWhenWordsArePrefixesOfEachOther()
        {
            int multiTokenMatches = 0;

            for (int seed = 0; seed < 20; seed++)
            {
                var rng     = new Random(7000 + seed);
                var entries = new HashSet<string>();

                while (entries.Count < 50)
                {
                    int words = rng.Next(1, 4);
                    entries.Add(string.Join(" ", Enumerable.Range(0, words).Select(_ => PREFIX_VOCABULARY[rng.Next(PREFIX_VOCABULARY.Length)])));
                }

                var spotter = new Spotter(Language.English, 0, "", "Entity");
                foreach (var entry in entries) { spotter.AddEntry(entry); }

                var pipeline = await PlainPipelineAsync();
                pipeline.Add(spotter);

                for (int d = 0; d < 20; d++)
                {
                    var tokens = Enumerable.Range(0, rng.Next(4, 20)).Select(_ => PREFIX_VOCABULARY[rng.Next(PREFIX_VOCABULARY.Length)]).ToArray();
                    var doc    = new Document(string.Join(" ", tokens), Language.English);
                    pipeline.ProcessSingle(doc);

                    var actual   = ActualTags(doc, "Entity").OrderBy(t => t.index).ThenBy(t => (int)t.tag).ToList();
                    var expected = ExpectedTags(tokens, entries);

                    Assert.Equal(expected, actual);
                    multiTokenMatches += actual.Count(t => t.tag == EntityTag.Begin);
                }
            }

            Assert.True(multiTokenMatches > 100, $"only {multiTokenMatches} multi-token matches were exercised");
        }

        [Fact]
        public async Task Spotter_MatchesTheOracleWhenIgnoringCase()
        {
            var rng = new Random(99);
            var (entries, documents) = Generate(rng, entryCount: 40, documentCount: 15);

            var spotter = new Spotter(Language.English, 0, "", "Entity") { IgnoreCase = true };
            foreach (var entry in entries) { spotter.AddEntry(entry.ToUpperInvariant()); }

            var pipeline = await PlainPipelineAsync();
            pipeline.Add(spotter);

            foreach (var tokens in documents)
            {
                var doc = new Document(string.Join(" ", tokens), Language.English);
                pipeline.ProcessSingle(doc);

                var actual   = ActualTags(doc, "Entity").OrderBy(t => t.index).ThenBy(t => (int)t.tag).ToList();
                var expected = ExpectedTags(tokens, entries);

                Assert.Equal(expected, actual);
            }
        }

        [Fact]
        public async Task Spotter_TagsBeginAndSingleWhenAWordIsBothAnEntryAndTheStartOfALongerOne()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("New");
            spotter.AddEntry("New York");

            var pipeline = await PlainPipelineAsync();
            pipeline.Add(spotter);

            var doc = new Document("New York is big.", Language.English);
            pipeline.ProcessSingle(doc);

            var tags = ActualTags(doc, "Entity").OrderBy(t => t.index).ThenBy(t => (int)t.tag).ToList();
            Assert.Contains((0, EntityTag.Begin), tags);
            Assert.Contains((0, EntityTag.Single), tags);
            Assert.Contains((1, EntityTag.End), tags);
        }
    }
}
