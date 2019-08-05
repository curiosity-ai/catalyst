// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst.Models
{
    internal enum SplitPointReason
    {
        Normal,
        Prefix,
        Sufix,
        Infix,
        SingleChar,
        EmailOrUrl,
        Exception,
        Punctuation,
        Emoji
    }
}