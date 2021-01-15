using System;

namespace Catalyst
{
    public static partial class Spacy
    {
        public sealed class PythonLock : IDisposable
        {
            internal PythonLock()
            {

            }

            public void Dispose()
            {
                Spacy.Shutdown();
            }
        }
    }
}
