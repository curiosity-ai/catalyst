using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst
{
    /// <summary>
    /// Represents a group of tokens, typically representing an entity.
    /// </summary>
    public struct Tokens : ITokens
    {
        /// <inheritdoc />
        public int Begin { get { return Parent.TokensData[SpanIndex][ChildrenIndexes[0]].LowerBound; } set { throw new NotImplementedException(); } }

        /// <inheritdoc />
        public int End { get { return Parent.TokensData[SpanIndex][ChildrenIndexes[ChildrenIndexes.Length - 1]].UpperBound; } set { throw new NotImplementedException(); } }

        /// <inheritdoc />
        public int Length { get { return End - Begin + 1; } }

        /// <inheritdoc />
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

        /// <inheritdoc />
        public string OriginalValue
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
                    bool isHyphen = token.OriginalValueAsSpan.IsHyphen();
                    if (isHyphen)
                    {
                        if (sb.Length > 0 && sb[sb.Length - 1] == ' ') { sb.Length--; }
                        sb.Append(token.OriginalValueAsSpan);
                    }
                    else
                    {
                        sb.Append(token.OriginalValueAsSpan).Append(' ');
                    }
                }

                if (sb.Length > 0 && sb[sb.Length - 1] == ' ') { sb.Length--; }

                var val = sb.ToString();

                if (sb.Length < 10000) Pools.StringBuilder.Return(sb);

                return val;
            }
        }

        /// <inheritdoc />
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

        /// <inheritdoc />
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

        /// <inheritdoc />
        public EntityType EntityType { get; set; }

        /// <inheritdoc />
        public float Frequency { get; set; }

        /// <summary>Gets or sets the indexes of the child tokens within their span.</summary>
        public int[] ChildrenIndexes { get; set; }

        private int SpanIndex { get; set; }
        private Document Parent { get; set; }

        /// <inheritdoc />
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

        /// <inheritdoc />
        public int Index => throw new NotImplementedException();

        /// <inheritdoc />
        public ReadOnlySpan<char> ValueAsSpan => Value.AsSpan();
        /// <inheritdoc />
        public ReadOnlySpan<char> OriginalValueAsSpan => OriginalValue.AsSpan();

        /// <inheritdoc />
        public string Replacement { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        /// <inheritdoc />
        public Dictionary<string, string> Metadata => throw new NotImplementedException();

        /// <inheritdoc />
        public PartOfSpeech POS { get => PartOfSpeech.X; set => throw new NotImplementedException(); }

        /// <inheritdoc />
        public IReadOnlyList<EntityType> EntityTypes => new EntityType[] { EntityType };

        /// <inheritdoc />
        public int Head { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        /// <inheritdoc />
        public string DependencyType { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        /// <inheritdoc />
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

        /// <inheritdoc />
        public ReadOnlySpan<char> LemmaAsSpan => Lemma.AsSpan();

        /// <inheritdoc />
        public char? PreviousChar => Parent.GetPreviousChar(ChildrenIndexes[0], SpanIndex);
        /// <inheritdoc />
        public char? NextChar => Parent.GetNextChar(ChildrenIndexes[ChildrenIndexes.Length - 1], SpanIndex);

        /// <summary>
        /// Initializes a new instance of the <see cref="Tokens"/> struct.
        /// </summary>
        /// <param name="parent">The parent document.</param>
        /// <param name="spanIndex">The index of the span within the document.</param>
        /// <param name="children">The indexes of the child tokens.</param>
        /// <param name="entityType">The entity type.</param>
        public Tokens(Document parent, int spanIndex, int[] children, EntityType entityType = default(EntityType))
        {
            Parent = parent;
            SpanIndex = spanIndex;
            ChildrenIndexes = children;
            Frequency = 0;
            EntityType = entityType;
        }

        /// <inheritdoc />
        public void AddEntityType(EntityType entityType) => throw new NotImplementedException();

        /// <inheritdoc />
        public void UpdateEntityType(int ix, ref EntityType entityType) => throw new NotImplementedException();

        /// <inheritdoc />
        public void RemoveEntityType(string entityType) => throw new NotImplementedException();

        /// <inheritdoc />
        public void RemoveEntityType(int ix) => throw new NotImplementedException();

        /// <inheritdoc />
        public void ClearEntities() => throw new NotImplementedException();
    }
}