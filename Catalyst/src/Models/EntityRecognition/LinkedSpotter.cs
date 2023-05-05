using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Models
{
    public class LinkedSpotterModel : StorableObjectData
    {
        public Dictionary<ulong, UID128> Hashes { get; set; } = new Dictionary<ulong, UID128>();
        public List<HashSet<ulong>> MultiGramHashes { get; set; } = new List<HashSet<ulong>>();
        public string CaptureTag { get; set; }
        public HashSet<int> TokenizerExceptionsSet { get; set; } = new HashSet<int>();
        public bool IgnoreOnlyNumeric { get; set; }
        public bool IgnoreCase { get; set; }
    }

    public class LinkedSpotter : StorableObjectV2<LinkedSpotter, LinkedSpotterModel>, IEntityRecognizer, IProcess, IHasSimpleSpecialCases, ICanOptimizeMemory
    {
        public string CaptureTag => Data.CaptureTag;

        public bool IgnoreCase { get { return Data.IgnoreCase; } set { Data.IgnoreCase = value; } }

        public const string Separator = "_";

        private LinkedSpotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public LinkedSpotter(Language language, int version, string tag, string captureTag) : this(language, version, tag)
        {
            Data.CaptureTag = captureTag;
        }

        public new static async Task<LinkedSpotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new LinkedSpotter(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return new[] { CaptureTag };
        }

        public bool RecognizeEntities(IDocument document)
        {
            var foundAny = false;
            foreach (var span in document)
            {
                foundAny |= RecognizeEntities(span);
            }
            return foundAny;
        }

        public bool HasAnyEntity(IDocument document)
        {
            foreach (var span in document)
            {
                if (RecognizeEntities(span, stopOnFirstFound: true))
                {
                    return true;
                }
            }
            return false;
        }

        public void OptimizeMemory()
        {
            Data.TokenizerExceptionsSet?.Clear();
            Data.TokenizerExceptionsSet = null;
        }

        public static ulong HashCombine64(ulong rhs, ulong lhs)
        {
            lhs ^= rhs + 0x9e3779b97f492000 + (lhs << 6) + (lhs >> 2);
            return lhs;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Hash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += key[i];
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong IgnoreCaseHash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += char.ToLowerInvariant(key[i]);
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        public void ClearModel()
        {
            Data.Hashes.Clear();
            Data.MultiGramHashes.Clear();
            Data.TokenizerExceptionsSet.Clear();
        }

        public bool RecognizeEntities(Span ispan, bool stopOnFirstFound = false)
        {
            var pooledTokens = ispan.ToTokenSpanPolled(out var actualLength);
            var tokens = pooledTokens.AsSpan(0, actualLength);

            int N = tokens.Length;
            bool hasMultiGram = Data.MultiGramHashes.Any();
            bool foundAny = false;
            for (int i = 0; i < N; i++)
            {
                var tk = tokens[i];
                //if (tk.POS != PartOfSpeechEnum.NOUN && tk.POS != PartOfSpeechEnum.ADJ && tk.POS != PartOfSpeechEnum.PROPN) { continue; }

                var tokenHash = Data.IgnoreCase ? IgnoreCaseHash64(tk.ValueAsSpan) : Hash64(tk.ValueAsSpan);

                if (hasMultiGram && Data.MultiGramHashes[0].Contains(tokenHash))
                {
                    int window = Math.Min(N - i, Data.MultiGramHashes.Count);
                    ulong hash = tokenHash;
                    bool someTokenHasReplacements = tk.Replacement is object;
                    int i_final = i;
                    UID128 uid_final = default;

                    for (int n = 1; n < window; n++)
                    {
                        var next = tokens[n + i];
                        someTokenHasReplacements |= (next.Replacement is object);

                        var nextHash = Data.IgnoreCase ? IgnoreCaseHash64(next.ValueAsSpan) : Hash64(next.ValueAsSpan);
                        if (Data.MultiGramHashes[n].Contains(nextHash))
                        {
                            //txt += " " + next.Value;
                            //var hashTxt = Hash64(txt);
                            hash = HashCombine64(hash, nextHash);
                            if (Data.Hashes.TryGetValue(hash, out var uid_multi))
                            {
                                i_final = i + n;
                                uid_final = uid_multi;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (i_final > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return foundAny; } //Used for checking if the document contains any entity
                        tk.AddEntityType(new EntityType(CaptureTag, EntityTag.Begin, uid_final));
                        tokens[i_final].AddEntityType(new EntityType(CaptureTag, EntityTag.End, uid_final));

                        for (int m = i + 1; m < (i_final); m++)
                        {
                            tokens[m].AddEntityType(new EntityType(CaptureTag, EntityTag.Inside, uid_final));
                        }
                    }

                    i = i_final;
                }

                if (Data.Hashes.TryGetValue(tokenHash, out var uid))
                {
                    foundAny = true;
                    if (stopOnFirstFound) { return foundAny; } //Used for checking if the document contains any entity
                    tk.AddEntityType(new EntityType(CaptureTag, EntityTag.Single, uid));
                }
            }
            
            ArrayPool<Token>.Shared.Return(pooledTokens);

            return foundAny;
        }

        private ReaderWriterLockSlim TrainLock = new ReaderWriterLockSlim();

        public IEnumerable<int> GetSimpleSpecialCases()
        {
            if (Data.TokenizerExceptionsSet is object)
            {
                foreach (var sc in Data.TokenizerExceptionsSet)
                {
                    yield return sc;
                }
            }
        }

        public void AddEntry(string entry, UID128 uid)
        {
            void AddSingleTokenConcept(ulong entryHash)
            {
                Data.Hashes[entryHash] =  uid;
            }

            if (string.IsNullOrWhiteSpace(entry)) { return; }


            if (Data.IgnoreOnlyNumeric && int.TryParse(entry, out _)) { return; } //Ignore pure numerical entries

            var words = entry.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 1)
            {
                var hash = Data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[0].AsSpan()) : Spotter.Hash64(words[0].AsSpan());
                AddSingleTokenConcept(hash);

                if (!words[0].AsSpan().IsAllLetterOrDigit())
                {
                    Data.TokenizerExceptionsSet.Add(words[0].CaseSensitiveHash32());
                }

                return;
            }

            ulong combinedHash = 0;
            for (int n = 0; n < words.Length; n++)
            {
                var word_hash = Data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[n].AsSpan()) : Spotter.Hash64(words[n].AsSpan());
                if (n == 0) { combinedHash = word_hash; } else { combinedHash = Spotter.HashCombine64(combinedHash, word_hash); }
                if (Data.MultiGramHashes.Count < n + 1)
                {
                    Data.MultiGramHashes.Add(new HashSet<ulong>());
                }

                if (!Data.MultiGramHashes[n].Contains(word_hash))
                {
                    Data.MultiGramHashes[n].Add(word_hash);
                }

                if (!words[n].AsSpan().IsAllLetterOrDigit())
                {
                    Data.TokenizerExceptionsSet.Add(words[n].CaseSensitiveHash32());
                }
            }

            AddSingleTokenConcept(combinedHash);
        }

        public void AppendList(IEnumerable<(string word, UID128 uid)> words)
        {
            foreach (var (word, uid) in words)
            {
                AddEntry(word, uid);
            }
        }
    }
}