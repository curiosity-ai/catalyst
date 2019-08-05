// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst
{
    public interface ITokenizer
    {
        void Parse(IDocument document);

        void Parse(ISpan span);
    }
}