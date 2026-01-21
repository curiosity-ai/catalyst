using System.Collections.Generic;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a unit within a matching pattern.
    /// </summary>
    public interface IPatternUnit
    {
        /// <summary>Matches if the token consists of alphabetic characters only.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsAlpha();

        /// <summary>Matches if the token is a closing parenthesis.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsClosingParenthesis();

        /// <summary>Matches if the token consists of digits only.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsDigit();

        /// <summary>Matches if the token is an emoji.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsEmoji();

        /// <summary>Matches if the token consists of letters or digits only.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsLetterOrDigit();

        /// <summary>Matches if the token is in lower case.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsLowerCase();

        /// <summary>Matches if the token is numeric.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsNumeric();

        /// <summary>Matches if the token contains at least one numeric character.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit HasNumeric();

        /// <summary>Matches if the token is an opening parenthesis.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsOpeningParenthesis();

        /// <summary>Matches if the token is punctuation.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsPunctuation();

        /// <summary>Matches if the token is in title case.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsTitleCase();

        /// <summary>Matches if the token is in upper case.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit IsUpperCase();

        /// <summary>Matches if the token looks like an email address.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit LikeEmail();

        /// <summary>Matches if the token looks like a URL.</summary>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit LikeURL();

        /// <summary>Matches if the token has the specified entity type.</summary>
        /// <param name="entityType">The entity type.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithEntityType(string entityType);

        /// <summary>Matches if the token does not have the specified entity type.</summary>
        /// <param name="entityType">The entity type.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithoutEntityType(string entityType);

        /// <summary>Matches if the token has any of the specified part-of-speech tags.</summary>
        /// <param name="pos">The part-of-speech tags.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithPOS(params PartOfSpeech[] pos);

        /// <summary>Matches if the token has the specified prefix.</summary>
        /// <param name="prefix">The prefix.</param>
        /// <param name="ignoreCase">Whether to ignore case.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithPrefix(string prefix, bool ignoreCase = false);

        /// <summary>Matches if the token has the specified character shape.</summary>
        /// <param name="shape">The shape string.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithShape(string shape);

        /// <summary>Matches if the token has the specified suffix.</summary>
        /// <param name="suffix">The suffix.</param>
        /// <param name="ignoreCase">Whether to ignore case.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithSuffix(string suffix, bool ignoreCase = false);

        /// <summary>Matches if the token matches the specified text.</summary>
        /// <param name="token">The text to match.</param>
        /// <param name="ignoreCase">Whether to ignore case.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithToken(string token, bool ignoreCase = false);

        /// <summary>Matches if the token matches any of the specified texts.</summary>
        /// <param name="tokens">The collection of texts to match.</param>
        /// <param name="ignoreCase">Whether to ignore case.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithTokens(IEnumerable<string> tokens, bool ignoreCase = false);

        /// <summary>Matches if the token consists of the specified characters.</summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithChars(IEnumerable<char> chars);

        /// <summary>Matches if the token's length is within the specified range.</summary>
        /// <param name="minLength">The minimum length.</param>
        /// <param name="maxLength">The maximum length.</param>
        /// <returns>The updated pattern unit.</returns>
        IPatternUnit WithLength(int minLength, int maxLength);
    }
}