using Mosaik.Core;
using System;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;
using P = Catalyst.PatternUnitPrototype;

namespace Catalyst.Tests
{
    public class PipelineTests
    {
        [Fact]
        public async Task Pack_Unpack()
        {
            English.Register();
            ObjectStore.AddOtherAssembly(typeof(Pipeline).Assembly);
            var pipeline1 = await Pipeline.ForAsync(Language.English);

            pipeline1.Version = 123;
            pipeline1.Tag = "Test";

            using(var ms = new MemoryStream())
            {
                pipeline1.PackTo(ms);    
                ms.Seek(0, SeekOrigin.Begin);
                var pipeline2 = await Pipeline.LoadFromPackedAsync(ms);

                Assert.Equal(pipeline1.Version, pipeline2.Version);
                Assert.Equal(pipeline1.Tag, pipeline2.Tag);
                Assert.Equal(string.Join(";", pipeline1.GetModelsDescriptions().Select(md => md.ToString())),
                             string.Join(";", pipeline2.GetModelsDescriptions().Select(md => md.ToString())));
            }
        }

        [Theory]
        [InlineData("\u0001 \t\u0001\u0002\u0003\u0004\u0005\u0006\u0007\b\u0003\t\u0003\n\u000b\u0007\f\u0007\r\u000e\u0005\u000b\n\u000f\u0010\u0007\u0005\u0011\u0007\r\u0012\r\r\u0007\u0011\u0005\u000f\u0007\b\u0001\u0007 \u0013\u0014\b\u0015\u0007\u0016\u0017\u0018\u0007\u0016\u0019\u0007\r\u001a\u0007 \t\u0001 \u0002\u0003\u0004\u0005\u0006\u0007\b\t\t\n\u0006\u000b\u0001 \f\u0004\u000b\u000b\r\u000e\u0006\u0007\t\u000b\u000f\u0010\u000f \u0001 \t\u0001 \t\n")]
        public async Task MessyUnicode(string text)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount == 0);
        }

        [Theory]
        [InlineData(@"<HTML><HEAD><TITLE>Your Title Here</TITLE></HEAD><BODY BGCOLOR='FFFFFF'><CENTER><IMG SRC='clouds.jpg' ALIGN='BOTTOM'> </CENTER><HR><a href='http://somegreatsite.com'>Link Name</a>is a link to another nifty site<H1>This is a Header</H1><H2>This is a Medium Header</H2>Send me mail at <a href='mailto:support@yourcompany.com'>support@yourcompany.com</a>.<P> This is a new paragraph!<P> <B>This is a new paragraph!</B><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B><HR></BODY></HTML>")]
        public async Task HtmlNormalizer(string text)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(new Models.HtmlNormalizer());
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            Assert.DoesNotContain("<", doc.Value);
            Assert.DoesNotContain(">", doc.Value);
        }

        [Theory]
        [InlineData("this is an abbreviation test As Soon As Possible (ASAP) I hope this abbreviation was found")]
        public async Task Abbreviations(string text)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);

            var abbCapturer = new Models.AbbreviationCapturer(Language.English);
            var abbreviations = abbCapturer.ParseDocument(doc);
            Assert.Single(abbreviations);
            Assert.Equal("ASAP", abbreviations.Single().Abbreviation);
        }

        [Theory]
        [InlineData("Test of, previous and next char ")]
        public async Task TextPreviousNext(string text)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            var tokens = doc.SelectMany(s => s.Tokens).ToArray();

            Assert.Null(tokens.First().PreviousChar);
            Assert.Equal(' ', tokens.First().NextChar);

            Assert.Equal(' ', tokens[1].PreviousChar);
            Assert.Equal(',', tokens[1].NextChar);

            Assert.Equal(' ', tokens.Last().PreviousChar);
            Assert.Equal(' ', tokens.Last().NextChar);
            //Assert.Null(tokens.Last().NextChar);
        }

        [Theory]
        [InlineData("This is a TEST")]
        public async Task ToStringWithReplacements(string text)
        {
            English.Register();
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("TEST");

            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);

            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);

            Assert.Equal("This is a POTATO", doc.ToStringWithReplacements(t => "POTATO"));
        }

        [Theory]
        [InlineData("wiki-extract.txt")]
        public async Task TokenizerDoesNotThrowTimeoutDueToMalformedURLs(string file)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(File.ReadAllText(file), Language.English);
            nlp.ProcessSingle(doc);
        }

        [Theory]
        [InlineData("spam-extract.txt")]
        public async Task TokenizerDoesNotThrowTimeoutDueToMalformedURLs2(string file)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(File.ReadAllText(file), Language.English);
            nlp.ProcessSingle(doc);
        }

        [Theory]
        [InlineData("complex-extract.txt")]
        public async Task TokenizerHandlesComplexFile(string file)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(File.ReadAllText(file), Language.English);
            nlp.ProcessSingle(doc);
        }


        [Theory]
        [InlineData("json-extract.txt")]
        public async Task TokenizerHandlesJsonFile(string file)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document(File.ReadAllText(file), Language.English);
            nlp.ProcessSingle(doc);
        }

        [Theory]
        [InlineData("This is a very spec1fic Test")]
        public async Task PatternSpotterFuzzy(string text) {
            English.Register();
            var spotter = new PatternSpotter(Language.Any, 0, tag: "", captureTag: "Special");
            spotter.NewPattern(
                "S1",
                mp => mp.Add(
                       new PatternUnit(P.Single().WithTokenFuzzy("specific")),
                       new PatternUnit(P.Multiple().WithPOS(PartOfSpeech.NOUN, PartOfSpeech.PROPN, PartOfSpeech.AUX, PartOfSpeech.DET, PartOfSpeech.ADJ))
            ));
 
            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);
 
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            
            Assert.Equal(2, doc.EntityData.Count);
        }
    }
}
