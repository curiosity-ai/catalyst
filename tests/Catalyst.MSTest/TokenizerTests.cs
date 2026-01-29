using Microsoft.VisualStudio.TestTools.UnitTesting;
using Catalyst;
using Catalyst.Models;
using System.Linq;
using System.Threading.Tasks;
using Mosaik.Core;

namespace Catalyst.MSTest
{
    [TestClass]
    public class TokenizerTests
    {
        [TestMethod]
        public async Task Tokenizer_Basic_Splitting_Test()
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            var doc = new Document("Hello, world!", Language.English);
            nlp.ProcessSingle(doc);

            // Expected: "Hello", ",", "world", "!"
            Assert.AreEqual(4, doc.TokensCount);
            var tokens = doc.ToTokenList();
            Assert.AreEqual("Hello", tokens[0].Value);
            Assert.AreEqual(",", tokens[1].Value);
            Assert.AreEqual("world", tokens[2].Value);
            Assert.AreEqual("!", tokens[3].Value);
        }

        [TestMethod]
        public async Task Tokenizer_Contractions_Test()
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            var doc = new Document("Don't do it.", Language.English);
            nlp.ProcessSingle(doc);

            // Expected: "Do", "n't", "do", "it", "."
            var tokens = doc.ToTokenList();
            Assert.AreEqual("Do", tokens[0].Value);
            // Catalyst tokenizer seems to normalize "n't" to "not" in Value property
            Assert.AreEqual("not", tokens[1].Value);
        }

        [TestMethod]
        public async Task Tokenizer_URLs_Test()
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            var url = "https://www.google.com";
            var doc = new Document($"Check {url} out.", Language.English);
            nlp.ProcessSingle(doc);

            // Check that URL is kept as one token or handled correctly
            var tokens = doc.ToTokenList();
            Assert.IsTrue(tokens.Any(t => t.Value == url), $"URL '{url}' should be preserved as a token. Tokens: {string.Join(", ", tokens.Select(t => t.Value))}");
        }
    }
}
