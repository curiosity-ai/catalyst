// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst
{
    public interface ITagger
    {
        void Predict(IDocument document);

        void Predict(ISpan span);
    }
}