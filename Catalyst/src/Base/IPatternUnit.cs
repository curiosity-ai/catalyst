using System.Collections.Generic;

namespace Catalyst
{
    public interface IPatternUnit
    {
        IPatternUnit IsAlpha();
        IPatternUnit IsClosingParenthesis();
        IPatternUnit IsDigit();
        IPatternUnit IsEmoji();
        IPatternUnit IsLetterOrDigit();
        IPatternUnit IsLowerCase();
        IPatternUnit IsNumeric();
        IPatternUnit HasNumeric();
        IPatternUnit IsOpeningParenthesis();
        IPatternUnit IsPunctuation();
        IPatternUnit IsTitleCase();
        IPatternUnit IsUpperCase();
        IPatternUnit LikeEmail();
        IPatternUnit LikeURL();
        IPatternUnit WithEntityType(string entityType);
        IPatternUnit WithoutEntityType(string entityType);
        IPatternUnit WithPOS(params PartOfSpeech[] pos);
        IPatternUnit WithPrefix(string prefix, bool ignoreCase = false);
        IPatternUnit WithShape(string shape);
        IPatternUnit WithSuffix(string suffix, bool ignoreCase = false);
        IPatternUnit WithToken(string token, bool ignoreCase = false);
        IPatternUnit WithTokens(IEnumerable<string> tokens, bool ignoreCase = false);
        IPatternUnit WithTokenFuzzy(string token, float confidence = 0.8f, bool ignoreCase = false);
        IPatternUnit WithChars(IEnumerable<char> chars);
        IPatternUnit WithLength(int minLength, int maxLength);
    }
}