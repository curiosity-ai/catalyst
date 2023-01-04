using UID;
using MessagePack;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections.Specialized;

namespace Catalyst.Models
{
    public class AbbreviationCapturerCommonWords
    {
        public static string[] Get(Language language)
        {
            switch (language)
            {
                case Language.English: { return English; }
                default: { return Array.Empty<string>(); }
            }
        }

        private static string[] English = new[] { "is", "was", "be", "am", "are", "were", "how", "who", "when", "where", "why", "what", "which", "whence", "whereby", "wherein", "whereupon", "aboard", "about", "above", "across", "after", "against", "along", "amid", "among", "and", "around", "as", "at", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "concerning", "considering", "despite", "down", "during", "except", "excepting", "excluding", "following", "for", "from", "in", "inside", "into", "like", "minus", "near", "of", "off", "on", "onto", "opposite", "or", "outside", "over", "past", "per", "plus", "regarding", "round", "save", "since", "than", "through", "to", "toward", "towards", "under", "underneath", "unlike", "until", "up", "upon", "versus", "via", "with", "within", "without" }.Distinct().ToArray();
    }

    public class AbbreviationCapturer
    {
        public int MinimumAbbreviationLength = 2;
        private readonly MatchingPattern _capturePattern = new MatchingPattern(new MatchingPatternPrototype(nameof(AbbreviationCapturer)).Add(PatternUnitPrototype.Single().IsOpeningParenthesis(),
                                                                                                                                              PatternUnitPrototype.And(PatternUnitPrototype.ShouldNotMatch().IsOpeningParenthesis(), 
                                                                                                                                              PatternUnitPrototype.Multiple(maxMatches: 5).IsLetterOrDigit()), 
                                                                                                                                              PatternUnitPrototype.Single().IsClosingParenthesis()));
        private static ObjectPool<StringBuilder> _stringBuilders { get; } = new ObjectPool<StringBuilder>(() => new StringBuilder(), Environment.ProcessorCount, sb => { sb.Clear(); if (sb.Capacity > 1_000_000) sb.Capacity = 0; });

        private static char[] Parenthesis = new[] { '(', ')', '[', ']', '{', '}' };
        private PatternUnit DiscardCommonWords;

        private HashSet<ulong> Stopwords;
        private PatternUnit DiscardIsSymbol = new PatternUnit(PatternUnitPrototype.ShouldNotMatch().IsLetterOrDigit());
        private PatternUnit DiscardOnlyLowerCase = new PatternUnit(PatternUnitPrototype.Single().IsLowerCase());
        private int MaximumTokensToTestForDescriptionPerLetter = 5;

        private int ContextWindow = 250; // Number of characters on both directions to take tokens as context

        public Language Language { get; private set; }

        public AbbreviationCapturer(Language language)
        {
            Language = language;
            var commonWords = AbbreviationCapturerCommonWords.Get(language);
            DiscardCommonWords = new PatternUnit(PatternUnitPrototype.Single().WithTokens(commonWords, ignoreCase: true));
            Stopwords = new HashSet<ulong>(StopWords.Spacy.For(Language).Select(w => w.AsSpan().IgnoreCaseHash64()).ToArray());
        }

        public List<AbbreviationCandidate> ParseDocument(Document doc, Func<AbbreviationCandidate, bool> shouldSkip = null)
        {
            var found = new List<AbbreviationCandidate>();
            if (doc.Language != Language && doc.Language != Language.Any && Language != Language.Any) { return found; }

            int CountUpper(ReadOnlySpan<char> sp)
            {
                int count = 0;
                foreach(var c in sp)
                {
                    if (char.IsUpper(c)) count++;
                }
                return count;
            }

            foreach (var span in doc)
            {
                var tokens = span.ToTokenSpan();
                int N = tokens.Length - 3;

                for (int i = 1; i < N; i++)
                {
                    if (_capturePattern.IsMatch(tokens.Slice(i), out var consumedTokens) && consumedTokens > 2)
                    {
                        var slice = tokens.Slice(i + 1, consumedTokens - 2); //Skips opening and closing parenthesis
                        Token innerToken = slice[0];
                        int countUpper = CountUpper(innerToken.ValueAsSpan);

                        foreach(var it in slice.Slice(1))
                        {
                            var itC = CountUpper(it.ValueAsSpan);
                            if (itC > countUpper )
                            {
                                innerToken = it;
                                countUpper = itC;
                            }
                        }

                        bool shouldDiscard = false;
                        
                        shouldDiscard |= innerToken.Length > 100; //Too long
                        shouldDiscard |= DiscardOnlyLowerCase.IsMatch(ref innerToken); //All lower case
                        shouldDiscard |= DiscardCommonWords.IsMatch(ref innerToken);
                        shouldDiscard |= DiscardIsSymbol.IsMatch(ref innerToken);

                        if (!shouldDiscard)
                        {
                            //Backtrack on the previous tokens to see if we find the explanation of the

                            var lettersToMatch = innerToken.ValueAsSpan.ToArray().Where(c => char.IsUpper(c)).ToArray();

                            if (lettersToMatch.Length >= MinimumAbbreviationLength && lettersToMatch.Length > (0.5 * innerToken.Length)) //Accept abbreviations with up to 50% lower-case letters, as long as they have enough upper-case letters
                            {
                                var matchedLetters = new bool[lettersToMatch.Length];

                                int maxTokensToTry = MaximumTokensToTestForDescriptionPerLetter * lettersToMatch.Length;
                                int min = i - 1 - maxTokensToTry;
                                if (min < 0) { min = 0; }

                                for (int j = i - 1; j > min; j--) // starts from i - 1 as tokens[i] is the opening parenthesis we found above
                                {
                                    var cur = tokens[j].ValueAsSpan;

                                    if (cur.IndexOfAny(Parenthesis) >= 0)
                                    {
                                        break;
                                    }

                                    //Try to consume tokens
                                    for (int k = 0; k < lettersToMatch.Length; k++)
                                    {
                                        if (cur.IndexOf(lettersToMatch[k]) >= 0)
                                        {
                                            matchedLetters[k] = true;
                                        }
                                    }

                                    if (matchedLetters.All(b => b))
                                    {
                                        //Found all letters, so hopefully we have a match
                                        //Make sure now that the letters appear in sequence
                                        var fullSpan = doc.Value.AsSpan().Slice(tokens[j].Begin, tokens[i].Begin - tokens[j].Begin);

                                        if (AppearsIn(innerToken.ValueAsSpan, fullSpan) && !fullSpan.IsAllUpperCase())
                                        {
                                            break;
                                        }

                                        if (IsSubSequenceOf(lettersToMatch.AsSpan(), fullSpan))
                                        {
                                            var allUpper = fullSpan.ToArray().Where(c => char.IsUpper(c)).ToList();
                                            var allUpperAbb = new List<char>(lettersToMatch);
                                            while (allUpper.Count > 0 && allUpperAbb.Count > 0)
                                            {
                                                var c = allUpper[0];
                                                if (allUpperAbb.Remove(c))
                                                {
                                                    allUpper.Remove(c);
                                                }
                                                else
                                                {
                                                    break;
                                                }
                                            }

                                            //Only add this as an abbreviation if the abbreviation contains all candidate description upper-case letters

                                            if (allUpper.Count == 0)
                                            {
                                                var context = GetContextForCandidate(doc, innerToken);

                                                var candidate = new AbbreviationCandidate
                                                {
                                                    Abbreviation = innerToken.Value,
                                                    Description = GetStandardForm(fullSpan),
                                                    Context = context
                                                };


                                                //Skip bad abbreviation captures of the form 'ABB ( ABB )', or when the description is too small
                                                if (!(candidate.Description.AsSpan().IsAllUpperCase() || candidate.Description.Length < candidate.Abbreviation.Length + 5 || candidate.Abbreviation.Contains(candidate.Description)))
                                                {
                                                    if (shouldSkip is null || !shouldSkip(candidate))
                                                    {
                                                        found.Add(candidate);
                                                    }
                                                }


                                            }

                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        i += consumedTokens - 1; //-1 as we'll do an i++ imediatelly after
                    }
                }
            }
            return found;
        }

        private bool AppearsIn(ReadOnlySpan<char> abbreviation, ReadOnlySpan<char> description)
        {
            int m = description.Length - abbreviation.Length + 1;
            for (int i = 0; i < m; i++)
            {
                if(description[i] == abbreviation[0])
                {
                    bool found = true;
                    for(int k = 1; k < abbreviation.Length; k++)
                    {
                        found &= (description[i+k] == abbreviation[k]);
                    }
                    if (found) return true;
                }
            }
            return false;
        }

        public static string GetStandardForm(ReadOnlySpan<char> fullSpan)
        {
            var sb = _stringBuilders.Rent();

            bool lastWasSpace = true;
            for (int i = 0; i < fullSpan.Length; i++)
            {
                if (fullSpan.Slice(i, 1).IsWhiteSpace())
                {
                    if (!lastWasSpace)
                    {
                        sb.Append(' '); lastWasSpace = true;
                    }
                }
                else if (fullSpan.Slice(i, 1).IsQuoteCharacters())
                {
                    if ((fullSpan[i] == '\'' || fullSpan[i] == '’') && i > 0 && !fullSpan.Slice(i - 1, 1).IsWhiteSpace())
                    {
                        sb.Append('\''); //Keep ' in the middle of a word, normalize ’ to '
                    }
                    else
                    {
                        //do nothing;
                    }
                }
                else
                {
                    sb.Append(fullSpan[i]); lastWasSpace = false;
                }
            }
            var text = sb.ToString().Trim();
            _stringBuilders.Return(sb);
            return text;
        }

        public List<string> GetContextForCandidate(Document doc, IToken innerToken)
        {
            var context = new List<string>(); // We let duplicates happen here, as they contribute to show what are the most important words after

            var contextWindowBegin = innerToken.Begin - ContextWindow;
            var contextWindowEnd = innerToken.End + ContextWindow;

            if (contextWindowBegin < 0) { contextWindowBegin = 0; }
            if (contextWindowEnd > doc.Length - 1) { contextWindowEnd = doc.Length - 1; }

            foreach (var s in doc)
            {
                bool overlap = s.Begin < contextWindowEnd && contextWindowBegin < s.End;

                if (overlap)
                {
                    foreach (var tk in s)
                    {
                        if (tk.Begin >= contextWindowBegin && tk.End < contextWindowEnd)
                        {
                            //Almost same filtering as Spotter
                            bool filterPartOfSpeech = !(tk.POS == PartOfSpeech.ADJ || tk.POS == PartOfSpeech.NOUN || tk.POS == PartOfSpeech.PROPN);

                            if (filterPartOfSpeech) continue;

                            bool skipIfHasUpperCase = !tk.ValueAsSpan.IsAllLowerCase();

                            if (skipIfHasUpperCase) continue;

                            bool skipIfTooSmall = (tk.Length < 3);

                            if (skipIfTooSmall) continue;

                            bool skipIfNotAllLetterOrDigit = !(tk.ValueAsSpan.IsAllLetterOrDigit());

                            if (skipIfNotAllLetterOrDigit) continue;

                            bool skipIfStopWordOrEntity = Stopwords.Contains(tk.ValueAsSpan.IgnoreCaseHash64()) || tk.EntityTypes.Any();

                            if (skipIfStopWordOrEntity) continue;

                            bool skipIfMaybeOrdinal = (tk.ValueAsSpan.IndexOfAny(new char[] { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }, 0) >= 0 &&
                                                       tk.ValueAsSpan.IndexOfAny(new char[] { 't', 'h', 's', 't', 'r', 'd' }, 0) >= 0 &&
                                                       tk.ValueAsSpan.IndexOfAny(new char[] { 'a', 'b', 'c', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'u', 'v', 'w', 'x', 'y', 'z' }, 0) < 0);

                            if (skipIfMaybeOrdinal) continue;

                            context.Add(tk.Value);
                        }
                    }
                }
            }

            return context;
        }

        private static bool IsSubSequenceOf(ReadOnlySpan<char> inner, ReadOnlySpan<char> original)
        {
            int j = 0;
            int m = inner.Length;
            int n = original.Length;
            for (int i = 0; i < n && j < m; i++)
            {
                if (inner[j] == original[i]) { j++; }
            }

            return (j == m);
        }
    }

    [MessagePackObject(keyAsPropertyName: true)]
    public class AbbreviationCandidate
    {
        public string Abbreviation { get; set; }
        public string Description { get; set; }
        public IReadOnlyList<string> Context { get; set; }
    }
}