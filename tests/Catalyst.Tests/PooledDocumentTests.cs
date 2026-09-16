using MessagePack;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    public class PooledDocumentTests
    {
        private const string TEXT = "Curiosity GmbH is based in Berlin . The quick brown fox jumps over the lazy dog .";

        private static readonly UID128 DocumentUID = UID128.Parse("22222222222222222222UP");
        private static readonly UID128 TargetUID   = UID128.Parse("11111111111111111111PH");

        private static void Fill(IDocument document)
        {
            var first = document.AddSpan(0, 34);

            for (int i = 0; i < 7; i++)
            {
                var token = first.AddTokenAsStruct(i * 4, i * 4 + 3);
                token.POS = PartOfSpeech.NOUN;

                if (i % 3 == 0)
                {
                    token.AddEntityType(new EntityType("Organization", EntityTag.Single, TargetUID));
                }
            }

            var second = document.AddSpan(36, 79);

            for (int i = 0; i < 9; i++)
            {
                var token = second.AddTokenAsStruct(36 + i * 4, 36 + i * 4 + 3);
                token.POS = PartOfSpeech.VERB;
                token.Metadata["kind"] = "test";
            }

            document.Labels.Add("label");
            document.Metadata["source"] = "test";
        }

        private static Document NewDocument()
        {
            var document = new Document(TEXT, Language.English) { UID = DocumentUID };
            Fill(document);
            return document;
        }

        private static PooledDocument NewPooled(DocumentPool pool)
        {
            var document = pool.Rent(TEXT, Language.English);
            document.UID = DocumentUID;
            Fill(document);
            return document;
        }

        private static void AssertSameContent(IDocument expected, IDocument actual)
        {
            Assert.Equal(expected.Language,      actual.Language);
            Assert.Equal(expected.Value,         actual.Value);
            Assert.Equal(expected.UID,           actual.UID);
            Assert.Equal(expected.SpansCount,    actual.SpansCount);
            Assert.Equal(expected.TokensCount,   actual.TokensCount);
            Assert.Equal(expected.EntitiesCount, actual.EntitiesCount);
            Assert.Equal(expected.Labels,        actual.Labels);

            var expectedSpans = expected.Spans.ToArray();
            var actualSpans   = actual.Spans.ToArray();

            for (int s = 0; s < expectedSpans.Length; s++)
            {
                Assert.Equal(expectedSpans[s].Begin, actualSpans[s].Begin);
                Assert.Equal(expectedSpans[s].End,   actualSpans[s].End);

                var expectedTokens = expectedSpans[s].ToArray();
                var actualTokens   = actualSpans[s].ToArray();

                Assert.Equal(expectedTokens.Length, actualTokens.Length);

                for (int t = 0; t < expectedTokens.Length; t++)
                {
                    Assert.Equal(expectedTokens[t].Begin, actualTokens[t].Begin);
                    Assert.Equal(expectedTokens[t].End,   actualTokens[t].End);
                    Assert.Equal(expectedTokens[t].POS,   actualTokens[t].POS);
                    Assert.Equal(expectedTokens[t].Value, actualTokens[t].Value);

                    var expectedEntities = expectedTokens[t].EntityTypes;
                    var actualEntities   = actualTokens[t].EntityTypes;

                    Assert.Equal(expectedEntities.Count, actualEntities.Count);

                    for (int e = 0; e < expectedEntities.Count; e++)
                    {
                        Assert.Equal(expectedEntities[e].Type,      actualEntities[e].Type);
                        Assert.Equal(expectedEntities[e].Tag,       actualEntities[e].Tag);
                        Assert.Equal(expectedEntities[e].TargetUID, actualEntities[e].TargetUID);
                    }
                }
            }
        }

        [Fact]
        public void PooledDocumentHoldsTheSameContentAsADocument()
        {
            var pool   = new DocumentPool();
            var pooled = NewPooled(pool);

            AssertSameContent(NewDocument(), pooled);

            pool.Return(pooled);
        }

        [Fact]
        public void SerializesToTheSameBytesAsTheDocumentFormatter()
        {
            var pool   = new DocumentPool();
            var pooled = NewPooled(pool);

            var expected = MessagePackSerializer.Serialize(NewDocument());

            var buffer = new ArrayBufferWriter<byte>();
            pooled.SerializeAsMessagePack(buffer);

            Assert.Equal(expected, buffer.WrittenSpan.ToArray());

            pool.Return(pooled);
        }

        [Fact]
        public void SerializedPooledDocumentDeserializesAsADocument()
        {
            var pool   = new DocumentPool();
            var pooled = NewPooled(pool);

            using var stream = new MemoryStream();
            pooled.SerializeAsMessagePack(stream);
            stream.Position = 0;

            AssertSameContent(pooled, MessagePackSerializer.Deserialize<Document>(stream));

            pool.Return(pooled);
        }

        [Fact]
        public void SerializedPooledDocumentRoundTripsBackIntoThePool()
        {
            var pool   = new DocumentPool();
            var pooled = NewPooled(pool);

            using var stream = new MemoryStream();
            pooled.SerializeAsMessagePack(stream);
            stream.Position = 0;

            var reconstructed = pool.RentFromMessagePack(stream);

            AssertSameContent(pooled, reconstructed);

            pool.Return(reconstructed);
            pool.Return(pooled);
        }

        [Fact]
        public void ADocumentFormatterPayloadReadsBackIntoThePool()
        {
            var pool  = new DocumentPool();
            var bytes = MessagePackSerializer.Serialize(NewDocument());

            var pooled = pool.RentFromMessagePack(bytes);

            AssertSameContent(NewDocument(), pooled);

            pool.Return(pooled);
        }

        [Fact]
        public void RemoveOverlapingTokensMatchesDocument()
        {
            var pool   = new DocumentPool();
            var plain  = new Document(TEXT, Language.English);
            var pooled = pool.Rent(TEXT, Language.English);

            var bounds = new (int begin, int end)[] { (0, 8), (0, 4), (0, 12), (5, 9), (5, 9), (20, 24), (13, 30), (13, 18) };

            var plainSpan  = plain.AddSpan(0, 40);
            var pooledSpan = pooled.AddSpan(0, 40);

            foreach (var (begin, end) in bounds)
            {
                plainSpan.AddToken(begin, end);
                pooledSpan.AddToken(begin, end);
            }

            plain.RemoveOverlapingTokens();
            pooled.RemoveOverlapingTokens();

            AssertSameContent(plain, pooled);

            pool.Return(pooled);
        }

        [Fact]
        public void ARentedDocumentCarriesNothingFromThePreviousRenter()
        {
            var pool = new DocumentPool();

            var first = NewPooled(pool);
            pool.Return(first);

            var second = pool.Rent("hello", Language.German);

            Assert.Equal("hello",          second.Value);
            Assert.Equal(Language.German,  second.Language);
            Assert.Equal(default,          second.UID);
            Assert.Equal(0,                second.SpansCount);
            Assert.Equal(0,                second.TokensCount);
            Assert.Equal(0,                second.EntitiesCount);
            Assert.Empty(second.Labels);
            Assert.Empty(second.Metadata);

            pool.Return(second);
        }

        [Fact]
        public void ClearEmptiesTheDocumentAndKeepsItUsable()
        {
            var pool   = new DocumentPool();
            var pooled = NewPooled(pool);

            pooled.Clear();

            Assert.Equal(0, pooled.SpansCount);
            Assert.Equal(0, pooled.TokensCount);

            Fill(pooled);

            Assert.Equal(2,  pooled.SpansCount);
            Assert.Equal(16, pooled.TokensCount);

            pool.Return(pooled);
        }

        [Fact]
        public void RecycledEntityListsDoNotLeakBetweenTokens()
        {
            var pool   = new DocumentPool();
            var pooled = pool.Rent("alpha beta", Language.English);

            var token = pooled.AddSpan(0, 9).AddTokenAsStruct(0, 4);

            token.AddEntityType(new EntityType("A", EntityTag.Single));
            token.AddEntityType(new EntityType("B", EntityTag.Single));

            Assert.Equal(2, pooled.EntitiesCount);

            token.RemoveEntityType("A");
            token.RemoveEntityType("B");

            Assert.Equal(0, pooled.EntitiesCount);

            token.AddEntityType(new EntityType("C", EntityTag.Single));

            Assert.Equal(1,   pooled.EntitiesCount);
            Assert.Equal("C", token.EntityTypes[0].Type);

            pool.Return(pooled);
        }

        [Fact]
        public void ReturningToAForeignPoolThrows()
        {
            var pool   = new DocumentPool();
            var other  = new DocumentPool();
            var pooled = pool.Rent(TEXT, Language.English);

            Assert.Throws<InvalidOperationException>(() => other.Return(pooled));

            pool.Return(pooled);
        }

        [Fact]
        public void ConcurrentRentingKeepsDocumentsIndependent()
        {
            var pool     = new DocumentPool();
            var expected = MessagePackSerializer.Serialize(NewDocument());

            Parallel.For(0, Environment.ProcessorCount * 8, _ =>
            {
                for (int i = 0; i < 25; i++)
                {
                    var pooled = NewPooled(pool);
                    var buffer = new ArrayBufferWriter<byte>();

                    pooled.SerializeAsMessagePack(buffer);

                    Assert.Equal(expected, buffer.WrittenSpan.ToArray());

                    pool.Return(pooled);
                }
            });
        }

        [Fact]
        public void ADeepDocumentRoundTripsThroughThePool()
        {
            //A batch-shaped caller rents the whole batch before returning any of it, and the documents are deep -
            //which is what the pool's element budgets, rather than its instance counts, have to survive.
            var pool      = new DocumentPool();
            var documents = new List<PooledDocument>();

            for (int d = 0; d < 64; d++)
            {
                var document = pool.Rent(TEXT, Language.English);

                document.ReserveSpans(500);

                for (int s = 0; s < 500; s++)
                {
                    var span = document.AddSpan(0, 79);

                    for (int t = 0; t < 20; t++)
                    {
                        span.AddTokenAsStruct(t * 4, t * 4 + 3);
                    }
                }

                documents.Add(document);
            }

            foreach (var document in documents)
            {
                Assert.Equal(500,      document.SpansCount);
                Assert.Equal(500 * 20, document.TokensCount);
            }

            foreach (var document in documents)
            {
                pool.Return(document);
            }

            //Everything is recycled, and a document rented afterwards is empty rather than carrying the last one's spans
            var reused = pool.Rent(TEXT, Language.English);

            Assert.Equal(0, reused.SpansCount);
            Assert.Equal(0, reused.TokensCount);

            pool.Return(reused);
        }

        [Fact]
        public void TrimDropsWhatThePoolIsHoldingAndRentingStillWorks()
        {
            var pool = new DocumentPool();

            pool.Return(NewPooled(pool));
            pool.Trim();

            var afterTrim = NewPooled(pool);

            AssertSameContent(NewDocument(), afterTrim);

            pool.Return(afterTrim);
        }

        [Fact]
        public void APooledDocumentIsUsableThroughTheDocumentApi()
        {
            var pool   = new DocumentPool();
            var pooled = NewPooled(pool);

            Document asDocument = pooled;

            Assert.Equal(NewDocument().TokenizedValue(), asDocument.TokenizedValue());
            Assert.Equal(NewDocument().ToJson(),         asDocument.ToJson());

            var clone = asDocument.Clone();

            pool.Return(pooled);

            //The clone is independent of the pool, so it still reads correctly after the return
            AssertSameContent(NewDocument(), clone);
        }
    }
}
