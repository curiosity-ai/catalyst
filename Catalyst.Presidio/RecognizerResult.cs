using System;

namespace Catalyst.Presidio
{
    /// <summary>
    /// Represents the result of a PII recognition.
    /// </summary>
    public class RecognizerResult
    {
        /// <summary>
        /// Gets or sets the start index of the identified PII.
        /// </summary>
        public int Start { get; set; }

        /// <summary>
        /// Gets or sets the end index of the identified PII.
        /// </summary>
        public int End { get; set; }

        /// <summary>
        /// Gets or sets the confidence score of the recognition.
        /// </summary>
        public double Score { get; set; }

        /// <summary>
        /// Gets or sets the type of the identified PII (e.g., EMAIL_ADDRESS).
        /// </summary>
        public string EntityType { get; set; }

        /// <summary>
        /// Gets the length of the identified PII.
        /// </summary>
        public int Length => End - Start + 1;
    }
}
