
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Church_Slavic_Slavonic_Old_Bulgarian
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.Church_Slavic_Slavonic_Old_Bulgarian;

            public string GetLemma(IToken token)
            {
                return token.Value;
            }

            public ReadOnlySpan<char> GetLemmaAsSpan(IToken token)
            {
                return token.ValueAsSpan;
            }

            public bool IsBaseForm(IToken token)
            {
                return false;
            }
        }
    }
}
