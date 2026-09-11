using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    // The entry dictionary is a sorted, front-coded byte array read through a binary search over block
    // headers. These tests push the encodings that only show up at scale - shared prefixes and suffixes past
    // the nibble the control byte carries, entries that straddle a block boundary, multi-byte UTF-8 - and the
    // lookups at the very edges of the sorted order.
    public class EntryDictionaryEdgeCaseTests
    {
        private static async Task<Pipeline> PipelineWithAsync(Spotter spotter)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(spotter);
            return nlp;
        }

        private static Spotter Build(IEnumerable<string> entries)
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            foreach (var entry in entries) { spotter.AddEntry(entry); }
            spotter.TrimExcess();
            return spotter;
        }

        private static async Task AssertAllMatchAsync(IReadOnlyList<string> entries, IEnumerable<string> nonMembers)
        {
            var spotter = Build(entries);
            var nlp     = await PipelineWithAsync(spotter);

            Assert.Equal(entries.OrderBy(e => e, StringComparer.Ordinal).ToArray(), spotter.GetEntries().ToArray());

            foreach (var entry in entries)
            {
                var doc = new Document("start " + entry + " end", Language.English);
                nlp.ProcessSingle(doc);
                var captured = doc.SelectMany(s => s.GetEntities()).Select(e => e.Value).ToArray();
                Assert.True(captured.Contains(entry), $"'{entry}' was not captured, got [{string.Join("] [", captured)}]");
            }

            foreach (var absent in nonMembers)
            {
                var doc = new Document("start " + absent + " end", Language.English);
                nlp.ProcessSingle(doc);
                Assert.DoesNotContain(absent, doc.SelectMany(s => s.GetEntities()).Select(e => e.Value));
            }
        }

        // One entry either side of every block boundary, so the binary search has to land on a header, inside
        // a block, and on the first entry of the next block.
        [Theory]
        [InlineData(1)]
        [InlineData(31)]
        [InlineData(32)]
        [InlineData(33)]
        [InlineData(64)]
        [InlineData(65)]
        [InlineData(200)]
        public async Task EveryEntryIsFoundAtAnyBlockAlignment(int count)
        {
            var entries = Enumerable.Range(0, count).Select(i => "Entry" + i.ToString("0000")).ToList();
            await AssertAllMatchAsync(entries, new[] { "Entry9999", "Entr", "EntryX", "AAAA", "zzzz" });
        }

        [Fact]
        public async Task LongSharedPrefixesAndSuffixesRoundTrip()
        {
            // 40+ shared characters and 20+ differing ones: both lengths escape the control byte's nibbles.
            const string stem = "AVIATIONPARTNUMBERWITHAVERYLONGCOMMONSTEM";
            var entries = Enumerable.Range(0, 80).Select(i => stem + i.ToString("00") + "SUFFIXTHATISALSOQUITELONG" + i.ToString("00")).ToList();
            await AssertAllMatchAsync(entries, new[] { stem, stem + "99", "AVIATION" });
        }

        [Fact]
        public async Task VeryLongEntriesRoundTrip()
        {
            var entries = new List<string>
            {
                new string('A', 300),
                new string('A', 299) + "B",
                new string('B', 150),
                "short",
            };
            await AssertAllMatchAsync(entries, new[] { new string('A', 298), new string('C', 300) });
        }

        [Fact]
        public async Task MultiByteUtf8EntriesRoundTrip()
        {
            var entries = new List<string> { "Munchen", "Muenchen", "Zurich", "Sao", "Curiosidade", "Kobenhavn" };
            // Same list with the accents that make them multi-byte in UTF-8.
            entries = new List<string> { "München", "Zürich", "São", "Köbenhavn", "naïve", "日本語", "Ελλάδα" };
            await AssertAllMatchAsync(entries, new[] { "Munchen", "Zurich", "日本", "Ελλα" });
        }

        [Fact]
        public async Task AnEntryThatIsAPrefixOfAnotherIsNotConfusedWithIt()
        {
            var entries = new List<string> { "AB", "ABC", "ABCD", "AB C", "AB CD" };
            var spotter = Build(entries);
            var nlp     = await PipelineWithAsync(spotter);

            // "AB" alone is an entry and also starts the two-token entries, so it carries Single there too.
            var doc = new Document("x AB y", Language.English);
            nlp.ProcessSingle(doc);
            Assert.Equal(new[] { "AB" }, doc.SelectMany(s => s.GetEntities()).Select(e => e.Value).ToArray());

            doc = new Document("x ABC y", Language.English);
            nlp.ProcessSingle(doc);
            Assert.Equal(new[] { "ABC" }, doc.SelectMany(s => s.GetEntities()).Select(e => e.Value).ToArray());

            // The longest two-token entry wins; the shorter "AB C" must not be reported instead.
            doc = new Document("x AB CD y", Language.English);
            nlp.ProcessSingle(doc);
            var values = doc.SelectMany(s => s.GetEntities()).Select(e => e.Value).OrderBy(v => v).ToArray();
            Assert.Contains("AB CD", values);
            Assert.DoesNotContain("AB C", values);
        }

        [Fact]
        public async Task AnEmptyModelMatchesNothingAndDoesNotThrow()
        {
            var spotter = Build(Array.Empty<string>());
            var nlp     = await PipelineWithAsync(spotter);

            var doc = new Document("Nothing here should match at all.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Empty(doc.SelectMany(s => s.GetEntities()));
            Assert.Empty(spotter.GetEntries());
        }

        [Fact]
        public async Task EntriesWithRepeatedAndSurroundingWhitespaceNormalizeToOneForm()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("  New    York  ");
            spotter.AddEntry("New York");
            spotter.TrimExcess();

            Assert.Equal(new[] { "New York" }, spotter.GetEntries().ToArray());

            var nlp = await PipelineWithAsync(spotter);
            var doc = new Document("Visiting New York now.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Contains("New York", doc.SelectMany(s => s.GetEntities()).Select(e => e.Value));
        }
    }
}
