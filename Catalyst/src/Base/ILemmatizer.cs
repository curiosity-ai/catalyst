using Mosaik.Core;
using System;

namespace Catalyst
{
    public interface ILemmatizer
    {
        Language Language { get; }
     
        /// <summary>
        /// Check if the token is an uninflected paradigm, so we can avoid lemmatization entirely.
        /// </summary>
        /// <param name="token"></param>
        /// <returns></returns>
        bool IsBaseForm(IToken token);
        
        string GetLemma(IToken token);
        
        ReadOnlySpan<char> GetLemmaAsSpan(IToken token);
    }
}