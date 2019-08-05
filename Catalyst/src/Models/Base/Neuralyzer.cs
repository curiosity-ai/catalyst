// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class NeuralyzerModel : StorableObjectData
    {
        public List<string> UncaptureTags { get; set; } = new List<string>();
        public Dictionary<string, MatchingPattern> RemovePatternPerEntityType { get; set; } = new Dictionary<string, MatchingPattern>();
        public Dictionary<string, MatchingPattern> AddPatternPerEntityType { get; set; } = new Dictionary<string, MatchingPattern>();
        public Dictionary<string, MatchingPattern> ApproximateRemovePatternPerEntityType { get; set; } = new Dictionary<string, MatchingPattern>();
        public Dictionary<string, MatchingPattern> ApproximateAddPatternPerEntityType { get; set; } = new Dictionary<string, MatchingPattern>();
    }

    public class Neuralyzer : StorableObject<Neuralyzer, NeuralyzerModel>, IEntityRecognizer, IProcess
    {
        private readonly ReaderWriterLockSlim RWLock = new ReaderWriterLockSlim();

        public Neuralyzer(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public new static async Task<Neuralyzer> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new Neuralyzer(language, version, tag);
            await a.LoadDataAsync();
            a.Data.AddPatternPerEntityType = a.Data.AddPatternPerEntityType ?? new Dictionary<string, MatchingPattern>();
            a.Data.RemovePatternPerEntityType = a.Data.RemovePatternPerEntityType ?? new Dictionary<string, MatchingPattern>();
            a.Data.ApproximateAddPatternPerEntityType = a.Data.ApproximateAddPatternPerEntityType ?? new Dictionary<string, MatchingPattern>();
            a.Data.ApproximateRemovePatternPerEntityType = a.Data.ApproximateRemovePatternPerEntityType ?? new Dictionary<string, MatchingPattern>();
            return a;
        }

        public void Process(IDocument document)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return Data.AddPatternPerEntityType.Keys.ToArray();
        }

        public bool RecognizeEntities(IDocument document)
        {
            RWLock.EnterReadLock();
            try
            {
                var foundAny = false;
                foreach (var span in document)
                {
                    foundAny |= DoRecognizeEntities(span);
                }
                return foundAny;
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public bool Neuralyze(IDocument document, List<UID128> removedEntityTargetUIDs = null)
        {
            RWLock.EnterReadLock();
            try
            {
                var foundAny = false;
                foreach (var span in document)
                {
                    foundAny |= DoRecognizeEntities(span, removedEntityTargetUIDs: removedEntityTargetUIDs);
                }
                return foundAny;
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public bool HasAnyEntity(IDocument document)
        {
            RWLock.EnterReadLock();
            try
            {
                foreach (var span in document)
                {
                    if (DoRecognizeEntities(span, stopOnFirstFound: true))
                    {
                        return true;
                    }
                }
                return false;
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        private bool DoRecognizeEntities(ISpan ispan, bool stopOnFirstFound = false, List<UID128> removedEntityTargetUIDs = null)
        {
            var tokens = ispan.ToTokenSpan();
            int N = tokens.Length;
            bool foundAny = false;

            for (int i = 0; i < N; i++)
            {
                foreach (var kv in Data.ApproximateAddPatternPerEntityType)
                {
                    var et = kv.Key;
                    var p = kv.Value;
                    if (p.IsMatch(tokens.Slice(i), out var _) && Data.AddPatternPerEntityType[et].IsMatch(tokens.Slice(i), out var consumedTokens))
                    {
                        if (stopOnFirstFound) { return true; }

                        if (consumedTokens == 1)
                        {
                            tokens[i].AddEntityType(new EntityType(et, EntityTag.Single));
                        }
                        else
                        {
                            for (int j = i; j < (i + consumedTokens); j++)
                            {
                                tokens[j].AddEntityType(new EntityType(et, (j == i ? EntityTag.Begin : (j == (i + consumedTokens - 1) ? EntityTag.End : EntityTag.Inside))));
                            }
                        }

                        i += consumedTokens - 1; //-1 as we'll do an i++ imediatelly after
                        foundAny = true;
                        break;
                    }
                }

                foreach (var kv in Data.ApproximateRemovePatternPerEntityType)
                {
                    var et = kv.Key;
                    var p = kv.Value;
                    {
                        if (p.IsMatch(tokens.Slice(i), out var _) && Data.RemovePatternPerEntityType[et].IsMatch(tokens.Slice(i), out var consumedTokens))
                        {
                            if (stopOnFirstFound) { return true; }

                            if (removedEntityTargetUIDs is object)
                            {
                                removedEntityTargetUIDs.AddRange(tokens[i].EntityTypes.Where(etype => etype.Type == et && etype.Tag == (consumedTokens == 1 ? EntityTag.Single : EntityTag.Begin))
                                                                           .Where(etype => etype.TargetUID.IsNotNull())
                                                                           .Select(etype => etype.TargetUID));
                            }

                            for (int j = i; j < (i + consumedTokens); j++)
                            {
                                tokens[j].RemoveEntityType(et);
                            }

                            i += consumedTokens - 1; //-1 as we'll do an i++ imediatelly after
                            foundAny = true;
                            break;
                        }
                    }
                }
            }

            return foundAny;
        }

        public void ClearAllPatterns()
        {
            RWLock.EnterWriteLock();
            Data.AddPatternPerEntityType.Clear();
            Data.RemovePatternPerEntityType.Clear();
            Data.ApproximateAddPatternPerEntityType.Clear();
            Data.ApproximateRemovePatternPerEntityType.Clear();
            RWLock.ExitWriteLock();
        }

        public void ClearPatternsFor(string entityType)
        {
            RWLock.EnterWriteLock();
            Data.AddPatternPerEntityType.Remove(entityType);
            Data.RemovePatternPerEntityType.Remove(entityType);
            Data.ApproximateAddPatternPerEntityType.Remove(entityType);
            Data.ApproximateRemovePatternPerEntityType.Remove(entityType);
            RWLock.ExitWriteLock();
        }

        public void TeachForgetPattern(string entityType, string name, Action<MatchingPattern> pattern)
        {
            RWLock.EnterWriteLock();
            try
            {
                if (!Data.RemovePatternPerEntityType.TryGetValue(entityType, out var mp))
                {
                    mp = new MatchingPattern(name);
                    Data.RemovePatternPerEntityType.Add(entityType, mp);
                }
                pattern(mp);

                if (!Data.ApproximateRemovePatternPerEntityType.TryGetValue(entityType, out var mpApp))
                {
                    mpApp = new MatchingPattern(name);
                    Data.ApproximateRemovePatternPerEntityType.Add(entityType, mpApp);
                }
                pattern(mpApp);
                OptimizeMatchingPattern(entityType, mpApp, isAdd: false);
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        public void TeachAddPattern(string entityType, string name, Action<MatchingPattern> pattern)
        {
            RWLock.EnterWriteLock();
            try
            {
                if (!Data.AddPatternPerEntityType.TryGetValue(entityType, out var mp))
                {
                    mp = new MatchingPattern(name);
                    Data.AddPatternPerEntityType.Add(entityType, mp);
                }
                pattern(mp);

                if (!Data.ApproximateAddPatternPerEntityType.TryGetValue(entityType, out var mpApp))
                {
                    mpApp = new MatchingPattern(name);
                    Data.ApproximateAddPatternPerEntityType.Add(entityType, mpApp);
                }
                pattern(mpApp);
                OptimizeMatchingPattern(entityType, mpApp, isAdd: true);
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        //public void RemoveForgetPattern(string entityType, string name)
        //{
        //    if (!Data.AutoOptimizePatterns.Value) { return; }
        //    RWLock.EnterWriteLock();
        //    try
        //    {
        //        if (Data.RemovePatternsPerEntityType.TryGetValue(entityType, out var list))
        //        {
        //            list.RemoveAll(p => p.Name == name);
        //            if (list.Count == 0) { Data.RemovePatternsPerEntityType.Remove(entityType); }
        //        }
        //    }
        //    finally
        //    {
        //        RWLock.ExitWriteLock();
        //    }
        //}

        //public void RemoveAddPattern(string entityType, string name)
        //{
        //    RWLock.EnterWriteLock();

        //    try
        //    {
        //        if (Data.AddPatternPerEntityType.TryGetValue(entityType, out var mp))
        //        {
        //            mp.Patterns.RemoveAll(p => p.Name == name);
        //            if (mp.Patterns.Count == 0) { Data.AddPatternPerEntityType.Remove(entityType); }

        //            RWLock.ExitWriteLock();
        //        }
        //    }
        //    finally
        //    {
        //        RWLock.ExitWriteLock();
        //    }
        //}

        public void OptimizeMatchingPattern(string entityType, MatchingPattern mp, bool isAdd)
        {
            var patterns = mp.Patterns.ToArray();
            mp.Patterns.Clear();

            var ignorePatterns = patterns.Where(p => p.All(pu => isAdd ? !IsSimpleAddEntityType(pu, entityType) : !IsSimpleForgetEntityType(pu, entityType))).ToList();
            mp.Patterns.AddRange(ignorePatterns);

            foreach (var ignoreCase in new bool[] { false, true })
            {
                var words = new Dictionary<int, List<List<string[]>>>();
                foreach (var mergePatterns in patterns.Where(p => p.All(pu => isAdd ? IsSimpleAddEntityType(pu, entityType, ignoreCase) : IsSimpleForgetEntityType(pu, entityType, ignoreCase))))
                {
                    if (!words.TryGetValue(mergePatterns.Length, out var wordList))
                    {
                        wordList = new List<List<string[]>>();
                        words[mergePatterns.Length] = wordList;
                    }
                    wordList.Add(mergePatterns.Select(pu => GetTokens(pu).ToArray()).ToList());
                }

                foreach (var kv in words)
                {
                    var len = kv.Key; var wordList = kv.Value;
                    var newPatterns = new List<IPatternUnit>();

                    var hs = new HashSet<string>[len];
                    for (int i = 0; i < len; i++) { hs[i] = new HashSet<string>(); }
                    foreach (var wl in wordList)
                    {
                        for (int i = 0; i < len; i++)
                        {
                            foreach (var w in wl[i])
                            {
                                hs[i].Add(w);
                            }
                        }
                    }
                    newPatterns.AddRange(hs.Select(h => PatternUnitPrototype.Single().WithTokens(h, ignoreCase)));

                    foreach (var pu in newPatterns)
                    {
                        if (isAdd)
                        {
                            pu.WithoutEntityType(entityType);
                        }
                        else
                        {
                            pu.WithEntityType(entityType);
                        }
                    }
                    mp.Patterns.Add(newPatterns.Select(prot => new PatternUnit(prot)).ToArray());
                }
            }
        }

        private IEnumerable<string> GetTokens(PatternUnit pu)
        {
            if (pu.Token is object) { yield return pu.Token; }
            if (pu.Set is object)
            {
                foreach (var t in pu.Set)
                {
                    yield return t;
                }
            }
        }

        public bool IsSimpleAddEntityType(PatternUnit pu, string entityType)
        {
            return pu.Mode == PatternMatchingMode.Single && ((pu.Type == (PatternUnitType.Token | PatternUnitType.NotEntity)) || (pu.Type == (PatternUnitType.Set | PatternUnitType.NotEntity))) && pu.EntityType == entityType;
        }

        public bool IsSimpleForgetEntityType(PatternUnit pu, string entityType)
        {
            return pu.Mode == PatternMatchingMode.Single && ((pu.Type == (PatternUnitType.Token | PatternUnitType.Entity)) || (pu.Type == (PatternUnitType.Set | PatternUnitType.Entity))) && pu.EntityType == entityType;
        }

        public bool IsSimpleAddEntityType(PatternUnit pu, string entityType, bool ignoreCase)
        {
            return IsSimpleAddEntityType(pu, entityType) && pu.CaseSensitive == !ignoreCase;
        }

        public bool IsSimpleForgetEntityType(PatternUnit pu, string entityType, bool ignoreCase)
        {
            return IsSimpleForgetEntityType(pu, entityType) && pu.CaseSensitive == !ignoreCase;
        }
    }
}