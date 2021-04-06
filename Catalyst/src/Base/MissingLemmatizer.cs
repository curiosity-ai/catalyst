using Mosaik.Core;
using System;

namespace Catalyst
{
    public sealed class MissingLemmatizer : ILemmatizer
    {
        public Language Language => Language.Any;

        public string GetLemma(IToken token) => token.Value;

        public ReadOnlySpan<char> GetLemmaAsSpan(IToken token) => token.ValueAsSpan;

        public bool IsBaseForm(IToken token) => false;
    }
}