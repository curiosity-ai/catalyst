using System;

namespace Catalyst.Models
{
#pragma warning disable CA1032 // Implement standard exception constructors (2020-02-19 DWR: There are no options or additional properties here, so we don't need any of the other standard exception constructors)
    public sealed class EmptyCorpusException : Exception
    {
        private const string _message = "Empty corpus, nothing to train LDA model";

        public EmptyCorpusException() : base(_message) { }
    }
#pragma warning restore CA1032 // Implement standard exception constructors
}