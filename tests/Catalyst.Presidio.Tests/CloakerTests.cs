using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace Catalyst.Presidio.Tests
{
    public class CloakerTests
    {
        [Fact]
        public async Task CloakAsync_String_WorksCorrectly()
        {
            var cloaker = new PresidioCloaker();
            var doc = new Document("My name is John Doe and my email is john.doe@example.com.", Language.English);

            var results = new List<RecognizerResult>
            {
                new RecognizerResult(11, 18, "PERSON", 1f),
                new RecognizerResult(36, 56, "EMAIL", 1f),
            };

            var finalResult = await cloaker.CloakAsync(doc, results, async (cloakedText) =>
            {
                // Verify the text is cloaked
                Assert.Contains("PERSON_1", cloakedText);
                Assert.Contains("EMAIL_1", cloakedText);
                Assert.DoesNotContain("John Doe", cloakedText);
                Assert.DoesNotContain("john.doe@example.com", cloakedText);

                // Simulate an external service modifying the text but keeping the entities
                return await Task.FromResult(cloakedText.Replace("name is", "name was"));
            });

            // Verify the entities are rehydrated
            Assert.Contains("John Doe", finalResult);
            Assert.Contains("john.doe@example.com", finalResult);
            Assert.DoesNotContain("PERSON_1", finalResult);
            Assert.DoesNotContain("EMAIL_1", finalResult);
            Assert.Contains("name was", finalResult); // Verify the modification was kept
        }

        [Fact]
        public async Task CloakAsync_ConsistentTokens()
        {
            var cloaker = new PresidioCloaker();

            var doc1 = new Document("Hello John Doe.", Language.English);
            var results1 = new List<RecognizerResult>
            {
                new RecognizerResult(6, 13, "PERSON", 1.0f)
            };

            var doc2 = new Document("Goodbye John Doe.", Language.English);
            var results2 = new List<RecognizerResult>
            {
                new RecognizerResult(8, 15, "PERSON", 1f)
            };

            string token1 = null;
            await cloaker.CloakAsync(doc1, results1, async (cloakedText) =>
            {
                token1 = cloakedText.Replace("Hello ", "").Trim('.');
                return await Task.FromResult(cloakedText);
            });

            string token2 = null;
            await cloaker.CloakAsync(doc2, results2, async (cloakedText) =>
            {
                token2 = cloakedText.Replace("Goodbye ", "").Trim('.');
                return await Task.FromResult(cloakedText);
            });

            Assert.Equal("PERSON_1", token1);
            Assert.Equal("PERSON_1", token2);
        }

        [Fact]
        public async Task CloakAsync_Streaming_WorksCorrectly()
        {
            var cloaker = new PresidioCloaker();
            var doc = new Document("My name is John Doe and my email is john.doe@example.com.", Language.English);

            var results = new List<RecognizerResult>
            {
                new RecognizerResult(11, 18, "PERSON", 1f),
                new RecognizerResult(36, 56, "EMAIL", 1f),
            };

            // We simulate a stream that splits the cloaked text into very small chunks, even splitting the token "PERSON_1"
            async IAsyncEnumerable<string> SimulateStream(string input)
            {
                // Ensure we yield small chunks to test the rolling buffer
                for (int i = 0; i < input.Length; i += 2)
                {
                    yield return input.Substring(i, System.Math.Min(2, input.Length - i));
                }

                await Task.CompletedTask;
            }

            string cloakedInputForTest = "";
            var stream = cloaker.CloakAsync(doc, results, (cloakedInput) =>
            {
                cloakedInputForTest = cloakedInput;
                return SimulateStream(cloakedInput);
            });

            string finalResult = "";
            await foreach (var chunk in stream)
            {
                finalResult += chunk;
            }

            Assert.Contains("PERSON_1", cloakedInputForTest);
            Assert.Contains("EMAIL_1", cloakedInputForTest);
            Assert.Equal(doc.Value, finalResult);
        }
    }
}