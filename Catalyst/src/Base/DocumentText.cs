using System;
using System.Runtime.InteropServices;

namespace Catalyst
{
    /// <summary>
    /// Turns a document's text back into a string. Shared by <see cref="Document"/> and
    /// <see cref="ImmutableDocument"/>, which both store their text as <see cref="ReadOnlyMemory{T}"/>.
    /// </summary>
    internal static class DocumentText
    {
        /// <summary>
        /// Returns <paramref name="value"/> as a string, without copying when it already is one.
        /// </summary>
        public static string Materialize(ReadOnlyMemory<char> value)
        {
            if (value.IsEmpty) { return string.Empty; }

            //When the memory covers a whole string, hand that string back instead of copying it
            if (MemoryMarshal.TryGetString(value, out var text, out int start, out int length) && start == 0 && length == text.Length)
            {
                return text;
            }

            return value.Span.ToString();
        }
    }
}
