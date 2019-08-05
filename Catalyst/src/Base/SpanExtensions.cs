// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst
{
    public static class SpanExtensions
    {
        public static bool IsVerbRoot(this ISpan span)
        {
            int[] heads = span.Select(tk => tk.Head).ToArray();

            int root = Array.IndexOf(heads, -1);
            return (span[root].POS == PartOfSpeech.VERB);
        }

        public static bool IsProjective(this ISpan span)
        {
            int[] heads = span.Select(tk => tk.Head).ToArray();

            for (int i = 0; i < span.TokensCount; i++)
            {
                int head = heads[i];
                if (head == i) { continue; } //ROOT
                if (head < 0) { continue; }  //UNATTACHED or ROOT
                int start = head < i ? (head + 1) : (i + 1);
                int end = head < i ? (i) : (head);
                for (int k = start; k < end; k++)
                {
                    var ancestors = GetAncestors(k, heads).ToArray();
                    if (!ancestors.Contains(head)) { return false; }
                }
            }
            return true;
        }

        internal static IEnumerable<int> GetAncestors(int i, int[] heads)
        {
            int head = i, count = 0;
            while (heads[head] != head && count < heads.Length)
            {
                head = heads[head]; count++;
                yield return head;
                if (head < 0) { break; }
            }
        }
    }
}