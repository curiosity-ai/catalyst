using System;
using System.Runtime.Serialization;

namespace Catalyst.Models
{
    [Serializable]
    internal class TokenizationFailedException : Exception
    {
        public TokenizationFailedException()
        {
        }

        public TokenizationFailedException(string message) : base(message)
        {
        }

        public TokenizationFailedException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected TokenizationFailedException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}