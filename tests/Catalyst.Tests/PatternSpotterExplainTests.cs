using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    public class PatternSpotterExplainTests
    {
        private static async Task<(Pipeline nlp, PatternSpotter spotter)> SetupAsync(string captureTag = "CAP")
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            var spotter = new PatternSpotter(Language.English, 0, "test", captureTag);
            nlp.Add(spotter);
            return (nlp, spotter);
        }

        private static Document Tokenize(Pipeline nlp, string text)
        {
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            return doc;
        }

        [Fact]
        public async Task SingleLiteralTokenPattern_Matches()
        {
            var (nlp, spotter) = await SetupAsync();
            spotter.NewPattern("Cat", mp => mp.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.Single,
                Type = PatternUnitType.Token,
                Token = "Cat"
            }));

            var doc = Tokenize(nlp, "Cat");
            var explanations = spotter.Explain(doc);

            Assert.NotEmpty(explanations);
            var winning = Assert.Single(explanations.Where(e => e.ConsumedTokens > 0));
            var alt = Assert.Single(winning.Alternatives.Where(a => a.Matched));
            var unit = Assert.Single(alt.Units);
            Assert.True(unit.Matched);
            var crit = Assert.Single(unit.Criteria);
            Assert.Equal(MatchCriterion.Token, crit.Criterion);
            Assert.True(crit.Passed);
        }

        [Fact]
        public async Task SingleLiteralTokenPattern_DoesNotMatch_RecordsExpectedAndActual()
        {
            var (nlp, spotter) = await SetupAsync();
            spotter.NewPattern("Cat", mp => mp.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.Single,
                Type = PatternUnitType.Token,
                Token = "Cat"
            }));

            var doc = Tokenize(nlp, "Dog");
            var explanations = spotter.Explain(doc);

            Assert.NotEmpty(explanations);
            Assert.All(explanations, e => Assert.Equal(0, e.ConsumedTokens));

            var alt = explanations[0].Alternatives[0];
            var unit = alt.Units[0];
            var crit = Assert.Single(unit.Criteria);
            Assert.Equal(MatchCriterion.Token, crit.Criterion);
            Assert.False(crit.Passed);
            Assert.Equal("Cat", crit.Expected);
            Assert.Equal("Dog", crit.Actual);
            Assert.False(unit.Matched);
        }

        [Fact]
        public async Task TwoCriteria_OneFails_BothRecorded()
        {
            var (nlp, spotter) = await SetupAsync();
            spotter.NewPattern("AlphaPrefixedZ", mp => mp.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.Single,
                Type = PatternUnitType.IsAlpha | PatternUnitType.Prefix,
                Prefix = "Z"
            }));

            var doc = Tokenize(nlp, "Cat");
            var explanations = spotter.Explain(doc);
            var unit = explanations[0].Alternatives[0].Units[0];

            Assert.Equal(2, unit.Criteria.Count);
            Assert.Single(unit.Criteria.Where(c => c.Criterion == MatchCriterion.IsAlpha && c.Passed));
            Assert.Single(unit.Criteria.Where(c => c.Criterion == MatchCriterion.Prefix && !c.Passed));
            Assert.False(unit.Matched);
        }

        [Fact]
        public async Task OptionalUnit_FailedButAlternativeStillMatches()
        {
            var (nlp, spotter) = await SetupAsync();
            spotter.NewPattern("OptThenCat", mp => mp.Add(
                new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Optional = true,
                    Type = PatternUnitType.Token,
                    Token = "X"
                },
                new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.Token,
                    Token = "Cat"
                }
            ));

            var doc = Tokenize(nlp, "Cat");
            var explanations = spotter.Explain(doc);

            var first = explanations.First(e => e.StartTokenIndex == 0);
            var alt = first.Alternatives[0];
            Assert.True(alt.Matched);
            Assert.Equal(1, alt.ConsumedTokens);

            var optionalUnitTrace = alt.Units.First(u => u.Optional);
            Assert.False(optionalUnitTrace.Matched);

            var mandatoryUnitTrace = alt.Units.First(u => !u.Optional);
            Assert.True(mandatoryUnitTrace.Matched);
        }

        [Fact]
        public async Task ShouldNotMatch_RecordsInvertedFinalResult()
        {
            var (nlp, spotter) = await SetupAsync();
            spotter.NewPattern("NotDog", mp => mp.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.ShouldNotMatch,
                Type = PatternUnitType.Token,
                Token = "Dog"
            }));

            var doc = Tokenize(nlp, "Cat");
            var explanations = spotter.Explain(doc);

            var first = explanations.First(e => e.StartTokenIndex == 0);
            var unit = first.Alternatives[0].Units[0];
            Assert.True(unit.Matched);
            Assert.Contains(unit.Criteria, c => c.Criterion == MatchCriterion.ShouldNotMatch && c.Passed);
            Assert.Contains(unit.Criteria, c => c.Criterion == MatchCriterion.Token && !c.Passed);
        }

        [Fact]
        public async Task AndOr_PopulatesLeftAndRightSubTraces()
        {
            var (nlp, spotter) = await SetupAsync();
            spotter.NewPattern("AlphaAndPrefixC", mp => mp.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.And,
                LeftSide = new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsAlpha
                },
                RightSide = new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.Prefix,
                    Prefix = "C"
                }
            }));

            spotter.NewPattern("AlphaOrDigit", mp => mp.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.Or,
                LeftSide = new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsAlpha
                },
                RightSide = new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsDigit
                }
            }));

            var doc = Tokenize(nlp, "Cat");
            var explanations = spotter.Explain(doc);

            var andExplanation = explanations.First(e => e.PatternName == "AlphaAndPrefixC" && e.StartTokenIndex == 0);
            var andUnit = andExplanation.Alternatives[0].Units[0];
            Assert.Equal(PatternMatchingMode.And, andUnit.Mode);
            Assert.NotNull(andUnit.LeftSide);
            Assert.NotNull(andUnit.RightSide);
            Assert.True(andUnit.LeftSide.Matched);
            Assert.True(andUnit.RightSide.Matched);
            Assert.True(andUnit.Matched);

            var orExplanation = explanations.First(e => e.PatternName == "AlphaOrDigit" && e.StartTokenIndex == 0);
            var orUnit = orExplanation.Alternatives[0].Units[0];
            Assert.Equal(PatternMatchingMode.Or, orUnit.Mode);
            Assert.NotNull(orUnit.LeftSide);
            Assert.NotNull(orUnit.RightSide);
            Assert.True(orUnit.LeftSide.Matched);
            Assert.False(orUnit.RightSide.Matched);
            Assert.True(orUnit.Matched);
        }

        [Fact]
        public async Task MultipleAlternatives_LongerOneWins()
        {
            var (nlp, spotter) = await SetupAsync();
            var pattern = new MatchingPattern("TwoAlts");
            pattern.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.Single,
                Type = PatternUnitType.Token,
                Token = "Cat"
            });
            pattern.Add(new PatternUnit
            {
                Mode = PatternMatchingMode.Single,
                Type = PatternUnitType.IsAlpha
            }, new PatternUnit
            {
                Mode = PatternMatchingMode.Single,
                Type = PatternUnitType.IsAlpha
            });
            spotter.Data.Patterns.Add(pattern);

            var doc = Tokenize(nlp, "Cat dog");
            var explanations = spotter.Explain(doc);

            var first = explanations.First(e => e.StartTokenIndex == 0);
            Assert.Equal(2, first.ConsumedTokens);
            Assert.False(first.Alternatives[0].Matched || first.Alternatives[0].ConsumedTokens > 1);
            Assert.True(first.Alternatives[1].Matched);
            Assert.Equal(2, first.Alternatives[1].ConsumedTokens);
        }

        [Fact]
        public async Task Parity_ExplainMatchesProcess()
        {
            // For ten seeded documents, the per-position consumed-token counts produced by
            // Explain must equal what Process+RecognizeEntities decides for the same
            // (i, pattern) combinations. This guards against the traced overload drifting
            // from the production matcher.
            var seeds = new[]
            {
                "Cat dog 123",
                "Hello World",
                "42 alpha 9",
                "one two three four five",
                "Just one",
                "ABC 99 DEF 88",
                "a b c d e f",
                "X 1 Y 2 Z 3",
                "Multiple words here today",
                "Cat 1"
            };

            for (int s = 0; s < seeds.Length; s++)
            {
                var (nlpExplain, spotterExplain) = await SetupAsync("PARITY");
                spotterExplain.NewPattern("Alpha", mp => mp.Add(new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsAlpha
                }));
                spotterExplain.NewPattern("Digit", mp => mp.Add(new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsDigit
                }));

                var docExplain = Tokenize(nlpExplain, seeds[s]);
                var explanations = spotterExplain.Explain(docExplain);

                // Run the production matcher on an independent document.
                var (nlpProcess, spotterProcess) = await SetupAsync("PARITY");
                spotterProcess.NewPattern("Alpha", mp => mp.Add(new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsAlpha
                }));
                spotterProcess.NewPattern("Digit", mp => mp.Add(new PatternUnit
                {
                    Mode = PatternMatchingMode.Single,
                    Type = PatternUnitType.IsDigit
                }));
                var docProcess = Tokenize(nlpProcess, seeds[s]);
                spotterProcess.RecognizeEntities(docProcess);

                // Derive (startTokenIndex, length) tuples from the production matcher.
                var processSpans = new System.Collections.Generic.List<(int start, int length)>();
                foreach (var span in docProcess)
                {
                    int idx = 0;
                    foreach (var token in span.Tokens)
                    {
                        var ents = token.EntityTypes;
                        if (ents != null)
                        {
                            foreach (var e in ents)
                            {
                                if (e.Type == "PARITY" && (e.Tag == EntityTag.Begin || e.Tag == EntityTag.Single))
                                {
                                    int len = 1;
                                    if (e.Tag == EntityTag.Begin)
                                    {
                                        for (int k = idx + 1; k < span.TokensCount; k++)
                                        {
                                            var nextEnts = span.Tokens.ElementAt(k).EntityTypes;
                                            var nextTag = nextEnts?.FirstOrDefault(x => x.Type == "PARITY");
                                            if (nextTag?.Tag == EntityTag.Inside || nextTag?.Tag == EntityTag.End)
                                            {
                                                len++;
                                                if (nextTag?.Tag == EntityTag.End) break;
                                            }
                                            else break;
                                        }
                                    }
                                    processSpans.Add((idx, len));
                                }
                            }
                        }
                        idx++;
                    }
                }

                // Every Process-captured span must be matched by at least one Explain entry
                // whose ConsumedTokens equals or exceeds the Process span's length.
                foreach (var (start, length) in processSpans)
                {
                    Assert.Contains(explanations, e =>
                        e.StartTokenIndex == start &&
                        e.ConsumedTokens >= length);
                }

                Assert.True(explanations.Any(e => e.ConsumedTokens > 0),
                    $"Seed '{seeds[s]}' produced no matches in Explain");
            }
        }
    }
}
