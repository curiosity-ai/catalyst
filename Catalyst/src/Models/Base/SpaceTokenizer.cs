using Mosaik.Core;
using System;
using System.Linq;
using System.Threading;

namespace Catalyst.Models
{
    public class SpaceTokenizer : ITokenizer, IProcess
    {
        public SpaceTokenizer()
        {
        }

        public Language Language => Language.Any;
        public string Type => typeof(SpaceTokenizer).FullName;
        public string Tag => "";
        public int Version => 0;

        public void Parse(IDocument document)
        {
            if (!document.Spans.Any())
            {
                document.AddSpan(0, document.Length - 1);
            }
            foreach (ISpan s in document.Spans)
            {
                Parse(s);
            }
        }

        public void Parse(ISpan span)
        {
            var textSpan = span.ValueAsSpan;
            int spanBegin = span.Begin;
            int begin = 0, end = textSpan.IndexOf(' ');

            while (end >= 0)
            {
                if (!textSpan.Slice(0, end).IsNullOrWhiteSpace())
                {
                    span.AddToken(spanBegin + begin, spanBegin + begin + end - 1);
                }
                textSpan = textSpan.Slice(end + 1);
                begin += end + 1;
                end = textSpan.IndexOf(' ');
            }

            if (begin < span.Length)
            {
                if (!span.ValueAsSpan.Slice(begin, span.Length - begin).IsNullOrWhiteSpace())
                {
                    span.AddToken(spanBegin + begin, spanBegin + span.Length - 1);
                }
            }
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Parse(document);
        }
    }
}