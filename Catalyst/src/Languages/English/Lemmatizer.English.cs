using Mosaik.Core;
using System;
using System.Collections.Generic;

namespace Catalyst
{

    public static partial class Lemmatizer
    {
        private class English : ILemmatizer
        {
            public Language Language => Language.English;

            public string GetLemma(IToken token)
            {
#if NETCOREAPP3_0 || NETCOREAPP3_1 || NET5_0
                return new string(GetLemmaAsSpan(token));
#else
                return new string(GetLemmaAsSpan(token).ToArray());
#endif
            }

            public ReadOnlySpan<char> GetLemmaAsSpan(IToken token)
            {
                throw new NotImplementedException();
            }

            public bool IsBaseForm(IToken token)
            {
                throw new NotImplementedException();
            }
        }
    }
}