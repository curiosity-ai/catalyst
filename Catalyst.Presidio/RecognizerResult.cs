using System;

namespace Catalyst.Presidio
{
    /// <summary>
    /// Represents the result of a PII recognition.
    /// </summary>
    public struct RecognizerResult
    {
        public RecognizerResult(int start, int end, string entityType, float score)
        {
            Start = start;
            End = end;
            EntityType = entityType;
            Score = score;
        }

        /// <summary>
        /// Gets or sets the start index of the identified PII (inclusive).
        /// </summary>
        public int Start { get; }

        /// <summary>
        /// Gets or sets the end index of the identified PII (inclusive).
        /// </summary>
        public int End { get; }

        /// <summary>
        /// Gets or sets the confidence score of the recognition.
        /// </summary>
        public float Score { get; }

        /// <summary>
        /// Gets or sets the type of the identified PII (e.g., EMAIL_ADDRESS).
        /// </summary>
        public string EntityType { get; }

        /// <summary>
        /// Gets the length of the identified PII.
        /// </summary>
        public int Length => End - Start + 1;
    }


    public static class ITokensHelper
    {
        public static RecognizerResult AsRecognizerResult(this ITokens entity)
        {
            return new RecognizerResult(entity.Begin, entity.End, entity.EntityType.Type, 1.0f);
        }
    }
}
