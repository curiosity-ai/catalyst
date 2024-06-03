using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UID;

namespace Catalyst
{
    public static class WordNet
    {
        static WordNet()
        {
            Adjectives = ParseWordNet(GetResource("data.adj"));
            Nouns      = ParseWordNet(GetResource("data.noun"));
            Verbs      = ParseWordNet(GetResource("data.verb"));
            Adverbs    = ParseWordNet(GetResource("data.adv"), isAdverb: true);
        }

        public static  WordNetData Nouns { get; }
        public static  WordNetData Verbs { get; }
        public static  WordNetData Adjectives { get; }
        public static  WordNetData Adverbs { get; }

        private static string GetResource(string fileName)
        {
            using var stream = typeof(WordNet).Assembly.GetManifestResourceStream("Catalyst.WordNet.Resources.WNdb30." + fileName);
            using var sr = new StreamReader(stream, Encoding.UTF8);
            return sr.ReadToEnd();
        }

        internal static ulong HashWordIgnoreCaseUnderscoreIsSpace(ReadOnlySpan<char> word, ulong lexID, ulong uniqueId)
        {
            unchecked
            {
                //Implementation of Fowler/Noll/Vo (https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function), also used by the Roslyn compiler: https://github.com/dotnet/roslyn/blob/master/src/Compilers/Core/Portable/InternalUtilities/Hash.cs
                const ulong fnv_prime = 1099511628211LU;
                const ulong fnv_offset_basis = 0xcbf29ce484222325;
                ulong hash = fnv_offset_basis;
                for (int i = 0; i < word.Length; i++)
                {
                    var c = word[i];

                    if (c == '_') c = ' ';

                    if (char.IsUpper(c)) c = char.ToLowerInvariant(c);

                    hash = (hash ^ c) * fnv_prime;
                }

                return Hashes.Combine(Hashes.Combine(hash, lexID + 1UL), uniqueId + 1UL);
            }
        }

        private static WordNetData ParseWordNet(string fileContent, bool isAdverb = false)
        {
            //Data File Format
            //Format: synset_offset  lex_filenum  ss_type  w_cnt  word  lex_id  [word  lex_id...]  p_cnt  [ptr...]  [frames...]  |   gloss 
            //        https://wordnet.princeton.edu/documentation/wndb5wn

            //data.noun: 00252169 04 n 01 dry_cleaning 0 002 @ 00251013 n 0000 + 01535117 v 0101 | the act of cleaning (fabrics) with a solvent other than water  
            //           07423365 11 n 03 turn 1 turn_of_events 0 twist 0 004 @ 07423560 n 0000 + 00125841 v 0101 + 00146138 v 0102 + 02626604 v 0102 | an unforeseen development; "events suddenly took an awkward turn"  


            var lines = fileContent.AsSpan().Split(new char[] { '\r', '\n' });

            var wordCache = new StringBuilder();
            var pointers  = new List<WordNetPointers>();

            var otherWords = new List<int>();

            var terms     = new Dictionary<int, WordNetTerm>();

            var offsetMap = new Dictionary<ulong, int>();

            foreach (var line in lines)
            {
                int termWordStart_Others = otherWords.Count;
                var ls = fileContent.AsSpan(line.Start.Value, line.End.Value - line.Start.Value);

                if (ls.Length < 3) continue;

                if (ls[0] == ' ' && ls[1] == ' ') continue; //Comment from header

                var termOffset = GetInt(ls.Slice(0, "00252169".Length));
                var termLexFileNumber = GetInt(ls.Slice("00252169 ".Length, 2));
                var ss_type = ls.Slice("00252169 04 ".Length, 1)[0];
                var wordsCount = GetHex(ls.Slice("00252169 04 n ".Length, 2));
                var wordStart = ls.Slice("00252169 04 n 01 ".Length);
                var wordEnd = wordStart.IndexOf(' ');

                var (termWordStart, termWordLength) = AppendWord(wordCache, wordStart.Slice(0, wordEnd));


                var defMarker = wordStart.IndexOf('|');
                var definition = wordStart.Slice(defMarker + 2);

                var remaining = wordStart.Slice(wordEnd + 1, defMarker - wordEnd - 2);

                var word_lex_id = GetHex(remaining.Slice(0, 1));

                remaining = remaining.Slice("1 ".Length);

                var termSynsetType = GetSynsetType(ss_type);

                for (int i = 1; i < wordsCount; i++)
                {
                    var nextSpace = remaining.IndexOf(' ');
                    var other_word = AppendWord(wordCache, remaining.Slice(0, nextSpace));
                    remaining = remaining.Slice(nextSpace + 1);
                    var other_word_lex_id = GetHex(remaining.Slice(0, 1));
                    remaining = remaining.Slice("0 ".Length);

                    otherWords.Add(other_word.start);
                    otherWords.Add(other_word.length);
                    otherWords.Add(other_word_lex_id);
                }

                int termWordStart_Count = otherWords.Count - termWordStart_Others;

                var pointerCount = GetInt(remaining.Slice(0, "004".Length));

                var termPointersStart = pointers.Count;

                if (pointerCount > 0)
                {
                    remaining = remaining.Slice("004 ".Length);

                    for (int i = 0; i < pointerCount; i++)
                    {
                        var symbolEnd = remaining.IndexOf(' ');
                        var pointer_symbol = GetPointerSymbol(remaining.Slice(0, symbolEnd), isAdverb);
                        var pointer_offset = GetInt(remaining.Slice(symbolEnd + 1, "07423560".Length));
                        var pos = remaining.Slice(symbolEnd + 1 + "07423560 ".Length, 1)[0];
                        var source = GetHex(remaining.Slice(symbolEnd + 1 + "07423560 n ".Length, 2));
                        var target = GetHex(remaining.Slice(symbolEnd + 1 + "07423560 n 00".Length, 2));
                        remaining = remaining.Slice(symbolEnd + 1 + "07423560 n 0000".Length + (i < pointerCount - 1 ? 1 : 0));

                        pointers.Add(new WordNetPointers(pointer_offset, pointer_symbol, GetSynsetType(pos), source, target));
                    }
                }

                var termPointersLength = pointers.Count - termPointersStart;

                terms.Add(termOffset, new WordNetTerm(
                    termOffset,
                    word_lex_id,
                    termLexFileNumber,
                    termSynsetType,
                    termWordStart,
                    termWordLength,
                    termWordStart_Others,
                    termWordStart_Count,
                    termPointersStart,
                    termPointersLength
                    ));


                ulong trials = 0;
                //Handle colisions as words+lexID+lexFile are not unique, example from data.adj
                //  00038623 00 s 01 quiescent 0 002 & 00037757 a 0000 ;c 06060845 n 0000 | (pathology) causing no symptoms; "a quiescent tumor"  
                //  00040909 00 s 01 quiescent 0 004 & 00040685 a 0000 + 14011811 n 0103 + 14011811 n 0102 + 02190188 v 0104 | being quiet or still or inactive  

                while (true)
                {
                    var wordHash = HashWordIgnoreCaseUnderscoreIsSpace(wordStart.Slice(0, wordEnd), word_lex_id, trials);

                    if (!offsetMap.ContainsKey(wordHash))
                    {
                        offsetMap.Add(wordHash, termOffset);
                        break;
                    }
                    else
                    {
                        trials++;
                    }
                }
            }

            return new WordNetData(terms, wordCache.ToString(), pointers.ToArray(), otherWords.ToArray(), offsetMap);

            static int GetInt(ReadOnlySpan<char> span)
            {
                return int.Parse(span, NumberStyles.Number, CultureInfo.InvariantCulture.NumberFormat);
            }

            static byte GetHex(ReadOnlySpan<char> span)
            {
                return byte.Parse(span, NumberStyles.HexNumber, CultureInfo.InvariantCulture.NumberFormat);
            }


            static PointerSymbol GetPointerSymbol(ReadOnlySpan<char> readOnlySpan, bool isAdverb)
            {
                /******************************************************************
                 *  From: https://wordnet.princeton.edu/documentation/wninput5wn
                 *  
                 *   The pointer_symbol s for nouns are:
                 *
                 *       !    Antonym 
                 *       @    Hypernym 
                 *       @i    Instance Hypernym 
                 *           ~    Hyponym 
                 *           ~i    Instance Hyponym 
                 *       #m    Member holonym 
                 *       #s    Substance holonym 
                 *       #p    Part holonym 
                 *       %m    Member meronym 
                 *       %s    Substance meronym 
                 *       %p    Part meronym 
                 *       =    Attribute 
                 *       +    Derivationally related form         
                 *       ;c    Domain of synset - TOPIC 
                 *       -c    Member of this domain - TOPIC 
                 *       ;r    Domain of synset - REGION 
                 *       -r    Member of this domain - REGION 
                 *       ;u    Domain of synset - USAGE 
                 *       -u    Member of this domain - USAGE 
                 *
                 *   The pointer_symbol s for verbs are:
                 *
                 *       !    Antonym 
                 *       @    Hypernym 
                 *           ~    Hyponym 
                 *       *    Entailment 
                 *       >    Cause 
                 *       ^    Also see 
                 *       $    Verb Group 
                 *       +    Derivationally related form         
                 *       ;c    Domain of synset - TOPIC 
                 *       ;r    Domain of synset - REGION 
                 *       ;u    Domain of synset - USAGE 
                 *
                 *   The pointer_symbol s for adjectives are:
                 *
                 *       !    Antonym 
                 *       &    Similar to 
                 *       <    Participle of verb 
                 *       \    Pertainym (pertains to noun) 
                 *       =    Attribute 
                 *       ^    Also see 
                 *       ;c    Domain of synset - TOPIC 
                 *       ;r    Domain of synset - REGION 
                 *       ;u    Domain of synset - USAGE 
                 *
                 *   The pointer_symbol s for adverbs are:
                 *       !    Antonym 
                 *       \    Derived from adjective 
                 *       ;c    Domain of synset - TOPIC 
                 *       ;r    Domain of synset - REGION 
                 *       ;u    Domain of synset - USAGE 
                 ******************************************************************/

                switch (readOnlySpan[0])
                {
                    case '!': return PointerSymbol.Antonym;
                    case '@':
                        {
                            if (readOnlySpan.Length == 1) return PointerSymbol.Hypernym;
                            else if (readOnlySpan[1] == 'i') return PointerSymbol.InstanceHypernym;
                            else return PointerSymbol.UNKNOWN;
                        }
                    case '~':
                        {
                            if (readOnlySpan.Length == 1) return PointerSymbol.Hyponym;
                            else if (readOnlySpan[1] == 'i') return PointerSymbol.InstanceHyponym;
                            else return PointerSymbol.UNKNOWN;
                        }
                    case '#':
                        {
                            if (readOnlySpan.Length == 1) return PointerSymbol.UNKNOWN;
                            else if (readOnlySpan[1] == 'm') return PointerSymbol.MemberHolonym;
                            else if (readOnlySpan[1] == 's') return PointerSymbol.SubstanceHolonym;
                            else if (readOnlySpan[1] == 'p') return PointerSymbol.PartHolonym;
                            else return PointerSymbol.UNKNOWN;
                        }
                    case '%':
                        {
                            if (readOnlySpan.Length == 1) return PointerSymbol.UNKNOWN;
                            else if (readOnlySpan[1] == 'm') return PointerSymbol.MemberMeronym;
                            else if (readOnlySpan[1] == 's') return PointerSymbol.SubstanceMeronym;
                            else if (readOnlySpan[1] == 'p') return PointerSymbol.PartMeronym;
                            else return PointerSymbol.UNKNOWN;
                        }
                    case '*': return PointerSymbol.Entailment;
                    case '=': return PointerSymbol.Attribute;
                    case '+': return PointerSymbol.DerivationallyRelatedForm;
                    case ';':
                        {
                            if (readOnlySpan.Length == 1) return PointerSymbol.UNKNOWN;
                            else if (readOnlySpan[1] == 'c') return PointerSymbol.DomainOfSynsetTOPIC;
                            else if (readOnlySpan[1] == 'r') return PointerSymbol.DomainOfSynsetREGION;
                            else if (readOnlySpan[1] == 'u') return PointerSymbol.DomainOfSynsetUSAGE;
                            else return PointerSymbol.UNKNOWN;
                        }
                    case '-':
                        {
                            if (readOnlySpan.Length == 1) return PointerSymbol.UNKNOWN;
                            else if (readOnlySpan[1] == 'c') return PointerSymbol.MemberOfThisDomainTOPIC;
                            else if (readOnlySpan[1] == 'r') return PointerSymbol.MemberOfThisDomainREGION;
                            else if (readOnlySpan[1] == 'u') return PointerSymbol.MemberOfThisDomainUSAGE;
                            else return PointerSymbol.UNKNOWN;
                        }
                    case '>': return PointerSymbol.Cause;
                    case '^': return PointerSymbol.AlsoSee;
                    case '$': return PointerSymbol.VerbGroup;
                    case '&': return PointerSymbol.SimilarTo;
                    case '<': return PointerSymbol.ParticipleOfVerb;
                    case '\\': return isAdverb ? PointerSymbol.DerivedFromAdjective:  PointerSymbol.Pertainym;
                }


                return PointerSymbol.UNKNOWN;
            }

            static PartOfSpeech GetSynsetType(char ss_type)
            {
                switch (ss_type)
                {
                    case 'n': return PartOfSpeech.NOUN;
                    case 'v': return PartOfSpeech.VERB;
                    case 'a': return PartOfSpeech.ADJ;
                    case 's': return PartOfSpeech.ADJ;
                    case 'r': return PartOfSpeech.ADV;
                }
                return PartOfSpeech.X;
            }

            static (int start, int length) AppendWord(StringBuilder stringBuilder, ReadOnlySpan<char> word)
            {
                var i = stringBuilder.Length;
                foreach (var c in word)
                {
                    if (c == '_')
                    {
                        stringBuilder.Append(' ');
                    }
                    else
                    {
                        stringBuilder.Append(c);
                    }
                }
                return (i, word.Length);
            }
        }

        public enum PointerSymbol
        {
            UNKNOWN,

            Antonym,
            Hypernym,
            InstanceHypernym,
            Hyponym,
            InstanceHyponym,
            MemberHolonym,
            SubstanceHolonym,
            PartHolonym,
            MemberMeronym,
            SubstanceMeronym,
            PartMeronym,
            Attribute,
            DerivationallyRelatedForm,
            DomainOfSynsetTOPIC,
            MemberOfThisDomainTOPIC,
            DomainOfSynsetREGION,
            MemberOfThisDomainREGION,
            DomainOfSynsetUSAGE,
            MemberOfThisDomainUSAGE,

            Entailment,
            Cause,
            AlsoSee,
            VerbGroup,

            SimilarTo,
            ParticipleOfVerb,
            Pertainym,

            DerivedFromAdjective,
        }

        /// <summary>
        /// Represents a term in the <see cref="WordNetData"/>.
        /// </summary>
        public struct WordNetTerm : IEquatable<WordNetTerm>
        {
            public WordNetTerm(int termOffset, int lexID, int termLexFileNumber, PartOfSpeech termSynsetType, int termWordStart, int termWordLength, int termWordStart_Others, int termWordStart_Count, int termPointersStart, int termPointersLength)
            {
                Offset = termOffset;
                LexID = lexID;
                FileNumber = termLexFileNumber;
                PartOfSpeech = termSynsetType;
                WordStart = termWordStart;
                WordLength = termWordLength;
                WordsStart = termWordStart_Others;
                WordsLength = termWordStart_Count;
                PointersStart = termPointersStart;
                PointersLength = termPointersLength;
            }

            /// <summary>
            /// Byte offset in the data file
            /// </summary>
            public int Offset { get; }
            public int LexID { get; }
            public int FileNumber { get; }
            public PartOfSpeech PartOfSpeech { get; }

            /// <summary>
            /// Start index in the <see cref="WordNetData.WordsCache" />
            /// </summary>
            internal int WordStart { get; }

            /// <summary>
            /// Character length of the word
            /// </summary>
            public int WordLength { get; }
            public int WordsStart { get; }
            public int WordsLength { get; }
            public int PointersStart { get; }
            public int PointersLength { get; }
            public override readonly bool Equals(object obj)
                => obj is WordNetTerm term && this.Equals(term);

            public readonly bool Equals(WordNetTerm obj)
            {
                return this.PartOfSpeech == obj.PartOfSpeech && this.Offset == obj.Offset;
            }
        }

        /// <summary>
        /// Represents the pointers from a <see cref="WordNetTerm"/>.
        /// </summary>
        public struct WordNetPointers
        {
            public WordNetPointers(int pointerOffset, PointerSymbol pointerSymbol, PartOfSpeech partOfSpeech, byte source, byte target)
            {
                Offset = pointerOffset;
                Symbol = pointerSymbol;
                PartOfSpeech = partOfSpeech;
                Source = source;
                Target = target;
            }

            /// <summary>
            /// Gets or sets the (translated) source word 
            /// </summary>
            public string SourceWord { get; set; }

            /// <summary>
            /// Gets or sets the (translated) target word
            /// </summary>
            public string TargetWord { get; set; }

            /// <summary>
            /// The byte offset of the target synset in the data file corresponding to PoS.
            /// </summary>
            public int Offset { get; }
            public PointerSymbol Symbol { get; }
            public PartOfSpeech PartOfSpeech { get; }

            /// <summary>
            /// The source/target field distinguishes lexical and semantic pointers.
            /// These two digits indicates the word number in the current (source) synset.
            /// </summary>
            public byte Source { get; }

            /// <summary>
            /// The source/target field distinguishes lexical and semantic pointers.
            /// These two digits indicate the word number in the target synset.
            /// </summary>
            public byte Target { get; }
        }

        /// <summary>
        /// Represents a WordNet Data File for a single Part of Speech
        /// <see cref="https://wordnet.princeton.edu/documentation/wndb5wn"/> 
        /// </summary>
        public class WordNetData : IWordNetData
        {
            private readonly Lazy<IDictionary<string, IList<WordNetTerm>>> TermsByWord;

            /// <summary>
            /// The terms by their byte offset number
            /// </summary>
            internal readonly Dictionary<int, WordNet.WordNetTerm> Terms;
            private readonly string WordsCache;
            internal readonly WordNet.WordNetPointers[] Pointers;
            private readonly int[] OtherWordsCache;
            private readonly Dictionary<ulong, int> HashToOffset;

            public WordNetData(Dictionary<int, WordNet.WordNetTerm> terms, string wordsCache, WordNet.WordNetPointers[] pointers, int[] otherWordsCache, Dictionary<ulong, int> offsetMap)
            {
                Terms = terms;
                WordsCache = wordsCache;
                Pointers = pointers;
                OtherWordsCache = otherWordsCache;
                HashToOffset = offsetMap;
                TermsByWord = new Lazy<IDictionary<string, IList<WordNetTerm>>>(() =>
                {
                    var result = new Dictionary<string, IList<WordNetTerm>>();
                    foreach (var term in this.Terms.Values)
                    {
                        var word = GetWordFromCache(term.WordStart, term.WordLength);
                        if (result.ContainsKey(word))
                        {
                            result[word].Add(term);
                        }
                        else
                        {
                            result[word] = new List<WordNetTerm> {
                                 term
                            };
                        }
                    }
                    return result;
                });
            }

            public IEnumerable<(string Word, int LexId, PartOfSpeech PartOfSpeech)> GetAll()
            {
                foreach (var term in Terms.Values)
                {
                    yield return (GetWordFromCache(term.WordStart, term.WordLength), term.LexID, term.PartOfSpeech);
                }
            }

            /// <inheritdoc/>
            public IEnumerable<(string Word, int LexId)> GetSynonyms(string word, int lexId = -1)
            {
                ulong uniqueId = 0;

                while (true)
                {
                    if (lexId < 0)
                    {
                        ulong lexID = 0;
                        bool foundAny = false;
                        while (true)
                        {
                            var wordHash = WordNet.HashWordIgnoreCaseUnderscoreIsSpace(word, lexID, uniqueId);

                            if (HashToOffset.TryGetValue(wordHash, out var offset))
                            {
                                foundAny = true;
                                var term = Terms[offset];
                                yield return (GetWordFromCache(term.WordStart, term.WordLength), term.LexID);

                                if (term.WordsLength > 0)
                                {
                                    var others = OtherWordsCache.AsSpan(term.WordsStart, term.WordsLength).ToArray();
                                    for (int i = 0; i < others.Length; i += 3)
                                    {
                                        yield return (GetWordFromCache(others[i], others[i + 1]), others[i + 2]);
                                    }
                                }

                                lexID++;
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (!foundAny) break;
                    }
                    else
                    {
                        var wordHash = WordNet.HashWordIgnoreCaseUnderscoreIsSpace(word, (ulong)lexId, uniqueId);

                        if (HashToOffset.TryGetValue(wordHash, out var offset))
                        {
                            var term = Terms[offset];
                            yield return (GetWordFromCache(term.WordStart, term.WordLength), term.LexID);

                            if (term.WordsLength > 0)
                            {
                                var others = OtherWordsCache.AsSpan(term.WordsStart, term.WordsLength).ToArray();
                                for (int i = 0; i < others.Length; i += 3)
                                {
                                    yield return (GetWordFromCache(others[i], others[i + 1]), others[i + 2]);
                                }
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    uniqueId++;
                }
            }

            internal string GetWordFromCache(int start, int len)
            {
                return WordsCache.Substring(start, len);
            }

            /// <inheritdoc/>
            public IEnumerable<WordNetPointers> GetPointers(WordNetTerm term)
            {
                var word = GetWordFromCache(term.WordStart, term.WordLength);
                foreach (var pointer in this.GetPointers(word, term.LexID))
                {
                    yield return new WordNetPointers(pointer.Offset, pointer.Symbol, pointer.PartOfSpeech, pointer.Source, pointer.Target)
                    {
                        SourceWord = word
                    };
                }
            }

            /// <inheritdoc/>
            public IEnumerable<(int Offset, string Word, WordNet.PointerSymbol Symbol, PartOfSpeech PartOfSpeech, byte Source, byte Target)> GetPointers(string word, int lexId = -1)
            {
                ulong uniqueId = 0;

                while (true)
                {
                    if (lexId < 0)
                    {
                        ulong lexID = 0;
                        bool foundAny = false;
                        while (true)
                        {
                            var wordHash = WordNet.HashWordIgnoreCaseUnderscoreIsSpace(word, lexID, uniqueId);

                            if (HashToOffset.TryGetValue(wordHash, out var offset))
                            {
                                foundAny = true;
                                var term = Terms[offset];

                                var pointers = Pointers.AsSpan().Slice(term.PointersStart, term.PointersLength).ToArray();

                                for (int i = 0; i < pointers.Length; i++)
                                {
                                    var p = pointers[i];
                                    WordNetData otherData;
                                    switch (p.PartOfSpeech)
                                    {
                                        case PartOfSpeech.NOUN: otherData = WordNet.Nouns; break;
                                        case PartOfSpeech.VERB: otherData = WordNet.Verbs; break;
                                        case PartOfSpeech.ADJ: otherData = WordNet.Adjectives; break;
                                        case PartOfSpeech.ADV: otherData = WordNet.Adverbs; break;
                                        default: continue;
                                    }
                                    var otherTerm = otherData.Terms[p.Offset];
                                    yield return (p.Offset, otherData.GetWordFromCache(otherTerm.WordStart, otherTerm.WordLength), p.Symbol, p.PartOfSpeech, p.Source, p.Target);
                                }

                                lexID++;
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (!foundAny)
                        {
                            break;
                        }
                    }
                    else
                    {
                        var wordHash = WordNet.HashWordIgnoreCaseUnderscoreIsSpace(word, (ulong)lexId, uniqueId);

                        if (HashToOffset.TryGetValue(wordHash, out var offset))
                        {
                            var term = Terms[offset];

                            var pointers = Pointers.AsSpan().Slice(term.PointersStart, term.PointersLength).ToArray();
                            for (int i = 0; i < pointers.Length; i++)
                            {
                                var p = pointers[i];
                                WordNetData otherData;
                                switch (p.PartOfSpeech)
                                {
                                    case PartOfSpeech.NOUN: otherData = WordNet.Nouns; break;
                                    case PartOfSpeech.VERB: otherData = WordNet.Verbs; break;
                                    case PartOfSpeech.ADJ: otherData = WordNet.Adjectives; break;
                                    case PartOfSpeech.ADV: otherData = WordNet.Adverbs; break;
                                    default: continue;
                                }
                                var otherTerm = otherData.Terms[p.Offset];
                                yield return (otherTerm.Offset, otherData.GetWordFromCache(otherTerm.WordStart, otherTerm.WordLength), p.Symbol, p.PartOfSpeech, p.Source, p.Target);
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    uniqueId++;
                }
            }

            /// <inheritdoc/>
            public WordNetTerm GetTerm(int offset)
            {
                return this.Terms[offset];
            }

            public IEnumerable<WordNetTerm> GetTerms(string word)
            {
                if (this.TermsByWord.Value.TryGetValue(word, out var output))
                {
                    return output;
                }

                return Enumerable.Empty<WordNetTerm>();
            }

            /// <inheritdoc/>
            public IEnumerable<string> GetWords(WordNetTerm term)
            {
                return new[] { GetWordFromCache(term.WordStart, term.WordLength) };
            }
        }
    }
}
