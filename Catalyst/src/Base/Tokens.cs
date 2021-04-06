using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst
{
    public struct Tokens : ITokens
    {
        public int Begin { get { return Parent.TokensData[SpanIndex][ChildrenIndexes[0]].LowerBound; } set { throw new NotImplementedException(); } }

        public int End { get { return Parent.TokensData[SpanIndex][ChildrenIndexes[ChildrenIndexes.Length - 1]].UpperBound; } set { throw new NotImplementedException(); } }

        public int Length { get { return End - Begin + 1; } }

        public string Value
        {
            get
            {
                if ((Length + ChildrenIndexes.Length) < 1)
                {
                    return string.Empty;
                }

                var sb = Pools.StringBuilder.Rent();

                foreach (var token in Children)
                {
                    bool isHyphen = token.ValueAsSpan.IsHyphen();
                    if (isHyphen)
                    {
                        if (sb.Length > 0 && sb[sb.Length - 1] == ' ') { sb.Length--; }
                        sb.Append(token.ValueAsSpan);
                    }
                    else
                    {
                        sb.Append(token.ValueAsSpan).Append(' ');
                    }
                }

                if (sb.Length > 0 && sb[sb.Length - 1] == ' ') { sb.Length--; }

                var val = sb.ToString();
                
                if (sb.Length < 10000) Pools.StringBuilder.Return(sb);

                return val;
            }
        }

        public int Hash
        {
            get
            {
                int h = Children.First().Hash;
                foreach (var c in Children.Skip(1))
                {
                    h = Hashes.CombineWeak(h, c.Hash);
                }
                return h;
            }
            set { throw new NotImplementedException(); }
        }

        public int IgnoreCaseHash
        {
            get
            {
                int h = Children.First().IgnoreCaseHash;
                foreach (var c in Children.Skip(1))
                {
                    h = Hashes.CombineWeak(h, c.IgnoreCaseHash);
                }

                if (EntityType.Type is object)
                {
                    h = Hashes.CombineWeak(h, EntityType.Type.CaseSensitiveHash32());
                }

                return h;
            }
            set { throw new NotImplementedException(); }
        }

        public EntityType EntityType { get; set; }

        public float Frequency { get; set; }

        public int[] ChildrenIndexes { get; set; }

        private int SpanIndex { get; set; }
        private Document Parent { get; set; }

        public IEnumerable<IToken> Children
        {
            get
            {
                var spanData = Parent.TokensData[SpanIndex];
                foreach (var ix in ChildrenIndexes)
                {
                    var td = spanData[ix];
                    yield return new Token(Parent, ix, SpanIndex, td.Replacement is object, td.LowerBound, td.UpperBound);
                }
            }
        }

        public int Index => throw new NotImplementedException();

        public ReadOnlySpan<char> ValueAsSpan => Value.AsSpan();

        public string Replacement { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public Dictionary<string, string> Metadata => throw new NotImplementedException();

        public PartOfSpeech POS { get => PartOfSpeech.X; set => throw new NotImplementedException(); }

        public EntityType[] EntityTypes => new EntityType[] { EntityType };

        public int Head { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public string DependencyType { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public string Lemma
        {
            get
            {
                if ((Length + ChildrenIndexes.Length) < 1)
                {
                    return string.Empty;
                }

                var sb = Pools.StringBuilder.Rent();

                foreach (var token in Children)
                {
                    bool isHyphen = token.LemmaAsSpan.IsHyphen();
                    if (isHyphen)
                    {
                        if (sb.Length > 0 && sb[sb.Length - 1] == ' ') { sb.Length--; }
                        sb.Append(token.LemmaAsSpan);
                    }
                    else
                    {
                        sb.Append(token.LemmaAsSpan).Append(' ');
                    }
                }

                if (sb.Length > 0 && sb[sb.Length - 1] == ' ') { sb.Length--; }

                var val = sb.ToString();

                if (sb.Length < 10000) Pools.StringBuilder.Return(sb);

                return val;
            }
        }

        public ReadOnlySpan<char> LemmaAsSpan => Lemma.AsSpan();

        public Tokens(Document parent, int spanIndex, int[] children, EntityType entityType = default(EntityType))
        {
            Parent = parent;
            SpanIndex = spanIndex;
            ChildrenIndexes = children;
            Frequency = 0;
            EntityType = entityType;
        }

        public void AddEntityType(EntityType entityType) => throw new NotImplementedException();

        public void UpdateEntityType(int ix, ref EntityType entityType) => throw new NotImplementedException();

        public void RemoveEntityType(string entityType) => throw new NotImplementedException();

        public void RemoveEntityType(int ix) => throw new NotImplementedException();

        public void ClearEntities() => throw new NotImplementedException();
    }
}