using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;

namespace Catalyst.Models
{
    [FormerName("Mosaik.NLU.Models", "SimpleTokenizer")]
    public class FastTokenizer : ITokenizer, IProcess
    {
        public static bool DisableEmailOrURLCapture { get; set; } = false;
        
        public static TimeSpan Timeout { get; set; } = TimeSpan.FromHours(1);

        public Language Language { get; set; }
        public string Type => typeof(FastTokenizer).FullName;
        public string Tag => "";
        public int Version => 0;

        private readonly Dictionary<int, TokenizationException> _baseSpecialCases;
        private Dictionary<int, TokenizationException> _customSpecialCases;
        private ReaderWriterLockSlim _lockSpecialCases = new();

        public static Task<FastTokenizer> FromStoreAsync(Language language, int version, string tag)
        {
            return Task.FromResult(new FastTokenizer(language));
        }

        public static Task<bool> ExistsAsync(Language language, int version, string tag)
        {
            return Task.FromResult(true);
        } // Needs to say it exists, otherwise when calling StoredObjectInfo.ExistsAsync(Language language, int version, string tag), it will fail to load this model

        public FastTokenizer(Language language)
        {
            Language = language;
            _baseSpecialCases = TokenizerExceptions.Get(Language);
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Parse(document, cancellationToken);
        }

        public void Parse(IDocument document, CancellationToken cancellationToken = default)
        {
            if (document.SpansCount == 0)
            {
                document.AddSpan(0, document.Length - 1);
            }

            foreach (ISpan s in document.Spans)
            {
                try
                {
                    Parse(s, cancellationToken);
                }
                catch (InvalidOperationException ome)
                {
                    document.Clear();
                    throw new TokenizationFailedException($"Error tokenizing document:\n'{document.Value}'", ome);
                }
                catch (TaskCanceledException tce)
                {
                    document.Clear();
                    throw new TokenizationFailedException($"Timeout tokenizing document:\n'{document.Value}'", tce);
                }
            }
        }

        public IEnumerable<IToken> Parse(string text, CancellationToken cancellationToken = default)
        {
            var tmpDoc = new Document(text);
            Parse(tmpDoc, cancellationToken);
            return tmpDoc.Spans.First().Tokens;
        }

        public void ImportSpecialCases(IProcess process)
        {
            if (process is IHasSpecialCases cases)
            {
                _lockSpecialCases.EnterWriteLock();
                try
                {

                    _customSpecialCases ??= new();
                    foreach (var sc in cases.GetSpecialCases())
                    {
                        _customSpecialCases[sc.Key] = sc.Value;
                    }
                }
                finally
                {
                    _lockSpecialCases.ExitWriteLock();
                }
            }
        }

        public void AddSpecialCase(string word, TokenizationException exception)
        {
            _lockSpecialCases.EnterWriteLock();
            try
            {
                _customSpecialCases ??= new();
                _customSpecialCases[word.CaseSensitiveHash32()] = exception;
            }
            finally
            {
                _lockSpecialCases.ExitWriteLock();
            }
        }

        public void Parse(ISpan span, CancellationToken cancellationToken = default)
        {
            var sw = ValueStopwatch.StartNew();
            var timeout = Timeout;

            _lockSpecialCases.EnterReadLock();
            try
            {

                var customSpecialCases = _customSpecialCases;
                var baseSpecialCases = _baseSpecialCases;

                //TODO: store if a splitpoint is special case, do not try to fetch hash if not!
                var separators = CharacterClasses.WhitespaceCharacters;
                var textSpan = span.ValueAsSpan;

                bool hasEmoji = false;

                for (int i = 0; i < textSpan.Length - 1; i++)
                {
                    if (textSpan.Slice(i).IsEmoji(out _))
                    {
                        hasEmoji = true; break;
                    }
                }

                var splitPoints = new List<SplitPoint>(textSpan.Length / 4);
                var infixLocation = new List<(int index, int length)>();

                int offset = 0, sufix_offset = 0;
                int checkEvery = 0;
                while (true)
                {
                    if (splitPoints.Count > textSpan.Length)
                    {
                        throw new InvalidOperationException(); //If we found more splitting points than actual characters on the span, we hit a bug in the tokenizer
                    }

                    if (checkEvery++ % 1024 == 0)
                    {
                        if (timeout.Ticks > 0 && sw.GetElapsedTime() > timeout)
                        {
                            throw new TaskCanceledException("Timeout while parsing document");
                        }
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    offset += sufix_offset;
                    sufix_offset = 0;
                    if (offset > textSpan.Length) { break; }
                    var splitPoint = textSpan.IndexOfAny(separators, offset);
                    ReadOnlySpan<char> candidate;

                    if (splitPoint == offset)
                    {
                        //Happens on sequential separators
                        offset++; continue;
                    }

                    if (splitPoint < 0)
                    {
                        candidate = textSpan.Slice(offset);
                        splitPoint = offset + candidate.Length;
                        if (candidate.Length == 0) { break; }
                    }
                    else
                    {
                        candidate = textSpan.Slice(offset, splitPoint - offset);
                    }

                    //Special case to split also at emojis
                    if (hasEmoji)
                    {
                        for (int i = 0; i < (candidate.Length - 1); i++)
                        {
                            if (candidate.Slice(i).IsEmoji(out var emojiLength))
                            {
                                if (i == 0)
                                {
                                    splitPoint = offset + emojiLength - 1;
                                    candidate = candidate.Slice(0, emojiLength);
                                }
                                else
                                {
                                    splitPoint = offset + i - 1;
                                    candidate = candidate.Slice(0, i);
                                }
                                break;
                            }
                        }
                    }

                    while (!candidate.IsEmpty)
                    {
                        int hash = candidate.CaseSensitiveHash32();
                        if ((customSpecialCases is object && customSpecialCases.ContainsKey(hash)) || baseSpecialCases.ContainsKey(hash))
                        {
                            splitPoints.Add(new SplitPoint(offset, splitPoint - 1, SplitPointReason.Exception));
                            candidate = new ReadOnlySpan<char>();
                            offset = splitPoint + 1;
                            continue;
                        }
                        else if (candidate.IsLikeURLorEmail())
                        {
                            splitPoints.Add(new SplitPoint(offset, splitPoint - 1, SplitPointReason.EmailOrUrl));
                            candidate = new ReadOnlySpan<char>();
                            offset = splitPoint + 1;
                            continue;
                        }
                        else if (hasEmoji && candidate.IsEmoji(out var emojiLength))
                        {
                            splitPoints.Add(new SplitPoint(offset, offset + emojiLength - 1, SplitPointReason.Emoji));
                            candidate = candidate.Slice(emojiLength);
                            offset += emojiLength;
                            continue;
                        }
                        else
                        {
                            if (candidate.Length == 1)
                            {
                                splitPoints.Add(new SplitPoint(offset, offset, SplitPointReason.SingleChar));
                                candidate = new ReadOnlySpan<char>();
                                offset = splitPoint + 1;
                                continue;
                            }

                            if (!candidate.IsAllLetterOrDigit())
                            {
                                if (candidate.IsSentencePunctuation() || candidate.IsHyphen() || candidate.IsSymbol())
                                {
                                    splitPoints.Add(new SplitPoint(offset, splitPoint - 1, SplitPointReason.Punctuation));
                                    candidate = new ReadOnlySpan<char>();
                                    offset = splitPoint + 1;
                                    continue;
                                }

                                int prefixLocation = FindPrefix(candidate);
                                if (prefixLocation >= 0)
                                {
                                    splitPoints.Add(new SplitPoint(offset + prefixLocation, offset + prefixLocation, SplitPointReason.Prefix));
                                    candidate = candidate.Slice(prefixLocation + 1);
                                    offset += prefixLocation + 1;
                                    continue;
                                }

                                var (sufixIndex, sufixLength) = FindSufix(candidate);

                                if (sufixIndex > -1)
                                {
                                    splitPoints.Add(new SplitPoint(offset + sufixIndex, offset + sufixIndex + sufixLength - 1, SplitPointReason.Sufix));
                                    candidate = candidate.Slice(0, sufixIndex);
                                    splitPoint = offset + sufixIndex;
                                    sufix_offset += sufixLength;
                                    continue;
                                }

                                FindInfix(candidate, infixLocation);
                                if (infixLocation.Count > 0)
                                {
                                    int in_offset = offset;

                                    foreach (var (index, length) in infixLocation)
                                    {
                                        if ((offset + index - 1) >= in_offset)
                                        {
                                            splitPoints.Add(new SplitPoint(in_offset, offset + index - 1, SplitPointReason.Infix));
                                        }

                                        //Test if the remaining is not an exception first
                                        if ((in_offset - offset + index) <= candidate.Length)
                                        {
                                            var rest = candidate.Slice(in_offset - offset + index);
                                            int hashRest = rest.CaseSensitiveHash32();

                                            if ((customSpecialCases is object && customSpecialCases.ContainsKey(hashRest)) || baseSpecialCases.ContainsKey(hashRest))
                                            {
                                                in_offset = offset + index;
                                                break;
                                            }
                                        }
                                        in_offset = offset + index + length;
                                        splitPoints.Add(new SplitPoint(offset + index, offset + index + length - 1, SplitPointReason.Infix));
                                    }

                                    candidate = candidate.Slice(in_offset - offset);

                                    offset = in_offset;
                                    continue;
                                }
                            }
                        }

                        splitPoints.Add(new SplitPoint(offset, splitPoint - 1, SplitPointReason.Normal));
                        candidate = new ReadOnlySpan<char>();
                        offset = splitPoint + 1;
                    }
                }

                int spanBegin = span.Begin;
                int pB = int.MinValue, pE = int.MinValue;
                span.ReserveTokens(splitPoints.Count);
                splitPoints.Sort(_splitPointSorter);
                foreach (var sp in splitPoints)
                {
                    if (checkEvery++ % 1024 == 0)
                    {
                        if (timeout.Ticks > 0 && sw.GetElapsedTime() > timeout)
                        {
                            throw new TaskCanceledException("Timeout while parsing document");
                        }
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    int b = sp.Begin;
                    int e = sp.End;

                    if (pB == b && pE == e) { continue; }

                    pB = b; pE = e;

                    if (b > e)
                    {
#if DEBUG
                        throw new InvalidOperationException($"Error processing text: '{span.Value}', found token with begin={b} and end={e}");
#else
                        continue; //Ignore this token
#endif
                    }

                    while (char.IsWhiteSpace(textSpan[b]) && b < e) { b++; }

                    while (char.IsWhiteSpace(textSpan[e]) && e > b) { e--; }

                    int hash = textSpan.Slice(b, e - b + 1).CaseSensitiveHash32();

                    if (e < b)
                    {
                        continue;
                    }

                    if ((customSpecialCases is object && customSpecialCases.TryGetValue(hash, out TokenizationException exp)) || baseSpecialCases.TryGetValue(hash, out exp))
                    {
                        if (exp.Replacements is null)
                        {
                            var tk = span.AddToken(spanBegin + b, spanBegin + e);
                        }
                        else
                        {
                            //TODO: Tokens begins and ends are being artificially placed here, check in the future how to better handle this
                            int begin2 = spanBegin + b;
                            for (int i = 0; i < exp.Replacements.Length; i++)
                            {
                                //Adds replacement tokens sequentially, consuming one char from the original document at a time, and
                                //using the remaing chars in the last replacement token
                                var tk = span.AddToken(begin2, ((i == exp.Replacements.Length - 1) ? (spanBegin + e) : begin2));
                                tk.Replacement = exp.Replacements[i];
                                begin2++;
                            }
                        }
                    }
                    else
                    {
                        var tk = span.AddToken(spanBegin + b, spanBegin + e);
                        if (sp.Reason == SplitPointReason.EmailOrUrl && !DisableEmailOrURLCapture)
                        {
                            tk.AddEntityType(new EntityType("EmailOrURL", EntityTag.Single));
                        }
                    }
                }
            }
            finally
            {
                _lockSpecialCases.ExitReadLock();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (int index, int len) FindSufix(ReadOnlySpan<char> s)
        {
            if (s.Length < 2) { return (-1, 0); }

            char f = s[s.Length - 1];
            char bf = s[s.Length - 2];

            if (f == '.')
            {
                if (s.Length > 3 && s[s.Length - 3] == '°' && (bf == 'C' || bf == 'c' || bf == 'F' || bf == 'f' || bf == 'K' || bf == 'k'))
                {
                    //removes final '.' on a sequence of °[C|c|F|f|K|k].
                    return (s.Length - 1, 1);
                }
                else if (char.IsDigit(bf) || char.IsLower(bf) || CharacterClasses.Quotes.Contains(bf) || CharacterClasses.ExtraSufixesCharacters.Contains(bf))
                {
                    if (s.Length < 4)
                    {
                        int c = 0, u = 0, l = 0; // Count the number of dots and upper case
                        for (int i = 0; i < s.Length; i++)
                        {
                            if (s[i] == '.') c++;
                            else if (char.IsUpper(s[i])) u++;
                            else if (char.IsLower(s[i])) l++;
                        }
                        if (u == 1 && c == 1 && l < 3) { return (-1, 0); } // Handles abbreviations on the form of Ul. or Ull.
                    }

                    //removes final '.' on a sequence of [0-9|a-z|{Quotes}|{Extra}].
                    return (s.Length - 1, 1);
                }
                else if (bf == '.')
                {
                    return (s.Length - 1, 1);
                }
                else if (char.IsLetter(bf) && !char.IsUpper(bf))
                {
                    return (s.Length - 1, 1);
                }
                else if (s.Length > 2)
                {
                    int c = 0, u = 0, l = 0; // Count the number of dots and upper case
                    for (int i = 0; i < s.Length; i++)
                    {
                        if (s[i] == '.') c++;
                        else if (char.IsUpper(s[i])) u++;
                        else if (char.IsLower(s[i])) l++;
                    }
                    if (u == 1 && c == 1 && l < 3) { return (-1, 0); } // Handles abbreviations on the form of Ul. or Ull.
                    if (u > c + 1) { return (s.Length - 1, 1); }
                }
            }

            if (char.IsDigit(bf))
            {
                if ((f == '+' || f == '-' || f == '*' || f == '/'))
                {
                    //Remove the final +-*/ symbol on a sequence of [0-9][+-*/]
                    return (s.Length - 1, 1);
                }
                else if (CharacterClasses.CurrencyCharacters.Contains(f))
                {
                    //Remove final currency symbol from [0-9]$
                    return (s.Length - 1, 1);
                }

                //TODO: remove unit from [0-9][{units}]

                //_units = ('km km² km³ m m² m³ dm dm² dm³ cm cm² cm³ mm mm² mm³ ha µm nm yd in ft '
                //          'kg g mg µg t lb oz m/s km/h kmh mph hPa Pa mbar mb MB kb KB gb GB tb '
                //          'TB T G M K % км км² км³ м м² м³ дм дм² дм³ см см² см³ мм мм² мм³ нм '
                //          'кг г мг м/с км/ч кПа Па мбар Кб КБ кб Мб МБ мб Гб ГБ гб Тб ТБ тб')
            }
            if (CharacterClasses.PunctuationCharacters.Contains(f))
            {
                //Remove the final punctuation symbol
                return (s.Length - 1, 1);
            }

            if (CharacterClasses.CloseQuotesCharacters.Contains(f))
            {
                //Remove the final closing quotes
                return (s.Length - 1, 1);
            }

            return (-1, 0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int FindPrefix(ReadOnlySpan<char> s)
        {
            if (s.IsEmpty) { return -1; }
            char b = s[0];

            if (CharacterClasses.HyphenCharacters.Contains(b))
            {
                if (s.Length > 1 && !char.IsNumber(s[1]))
                {
                    return 0;
                }
            }

            if (b < 256)
            {
                if (CharacterClasses.ASCIIPrefixesCharacters.Contains(b)) { return 0; }
            }
            else
            {
                if (CharacterClasses.OtherPrefixInfixCharacters.Contains(b)) { return 0; }
            }
            return -1;
        }

        private static InfixLocationSorter _infixLocationSorter = new();
        private sealed class InfixLocationSorter : IComparer<(int index, int length)>
        {
            public int Compare((int index, int length) x, (int index, int length) y)
            {
                //.OrderBy(k => k.index).ThenBy(k => k.length).ToList();
                var ix = x.index.CompareTo(y.index);
                if (ix == 0)
                {
                    return x.length.CompareTo(y.length);
                }
                else
                {
                    return ix;
                }
            }
        }

        private static SplitPointSorter _splitPointSorter = new();
        private sealed class SplitPointSorter : IComparer<SplitPoint>
        {
            public int Compare(SplitPoint x, SplitPoint y)
            {
                //.OrderBy(s => s.Begin).ThenBy(s => s.End)
                var ix = x.Begin.CompareTo(y.Begin);
                if (ix == 0)
                {
                    return x.End.CompareTo(y.End);
                }
                else
                {
                    return ix;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FindInfix(ReadOnlySpan<char> s, List<(int index, int length)> infixLocation)
        {
            infixLocation.Clear();
            FindInfixNoOrder(s, infixLocation);
            infixLocation.Sort(_infixLocationSorter);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FindInfixNoOrder(ReadOnlySpan<char> s, List<(int index, int length)> infixLocation)
        {
            //Split any [...] inside a word

            int L1 = s.Length - 1;
            int index = 0;
            for (int i = 1; i < L1; i++)
            {
                char c = s[i];
                if (((c == '.' || c == '-' || c == '_') && s[i + 1] == c) || c == '…')
                {
                    index = i;
                    if (char.IsLetterOrDigit(s[i - 1]))
                    {
                        while (i < (L1 - 1) && s[i + 1] == c) { i++; }
                        if (i <= L1 && char.IsLetterOrDigit(s[i + 1]))
                        {
                            infixLocation.Add((index, i - index + 1));
                        }
                    }
                }
                else if (c == '.')
                {
                    //split any lower [.] Upper
                    if (char.IsLower(s[i - 1]) && char.IsUpper(s[i + 1]))
                    {
                        infixLocation.Add((i, 1));
                    }
                    else if (!char.IsLetterOrDigit(s[i + 1])) //split any [.] [NOT LETTER NOR DIGIT]
                    {
                        infixLocation.Add((i, 1));
                    }
                }
                else
                {
                    //Split any symbol inside a word
                    if (c < 256 && CharacterClasses.ASCIIInfixCharacters.Contains(c))
                    {
                        if (c == ':' && (i + 1 < L1) && s[i + 1] == '/' && s[i + 2] == '/')
                        {
                            continue;
                        }
                        //if (c == '$')
                        //{
                        //    if ( && ((i > 0 && char.IsLetter(s[i - 1])) || (i < s.Length && char.IsLetter(s[i + 1])))))
                        //    //Handle exceptio for currency symbols like R$, Z$, $U, TT$, RD$, $b BZ$   . List taken from: http://www.xe.com/symbols.php
                        //    //do nothing with them
                        //}
                        //else
                        //{
                        infixLocation.Add((i, 1));
                        //}
                    }
                    else if (c > 256 && CharacterClasses.SymbolCharacters.Contains(c))
                    {
                        infixLocation.Add((i, 1));
                    }
                }
            }
        }


    }
}
