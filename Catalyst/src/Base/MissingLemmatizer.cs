using Mosaik.Core;
using System;

namespace Catalyst
{
    public sealed class MissingLemmatizer : ILemmatizer
    {
        public Language Language => Language.Any;

        public string GetLemma(ReadOnlySpan<char> value) => new string(value);

        public ReadOnlySpan<char> GetLemmaAsSpan(ReadOnlySpan<char> value) => value;

        public bool IsBaseForm(ReadOnlySpan<char> value) => false;
    }
}