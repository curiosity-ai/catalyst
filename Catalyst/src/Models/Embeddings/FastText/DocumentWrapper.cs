using UID;
using Mosaik.Core;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;

namespace Catalyst.Models
{
    internal class DocumentWrapper : IDocument
    {
        private Document Source;
        private int IgnoreSpan;

        public DocumentWrapper(IDocument source)
        {
            Source = (Document)source;
        }

        public void SetInvisibleSpan(int index)
        {
            IgnoreSpan = index;
        }

        public ISpan this[int key] { get => Source[key]; set => throw new NotImplementedException(); }

        public Language Language { get => Source.Language; set => throw new NotImplementedException(); }

        public int Length => Source.Length;

        public string Value { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public string TokenizedValue(bool mergeEntities = false) => throw new NotImplementedException();

        public UID128 UID { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public List<string> Labels => throw new NotImplementedException();

        public Dictionary<string, string> Metadata => throw new NotImplementedException();

        public IEnumerable<ISpan> Spans
        {
            get
            {
                for (int i = 0; i < Source.SpansCount; i++)
                {
                    if (i != IgnoreSpan)
                    {
                        yield return new Span(Source, i);
                    }
                }
            }
        }

        public int SpansCount => throw new NotImplementedException();

        public int TokensCount => throw new NotImplementedException();

        public int EntitiesCount => throw new NotImplementedException();

        public bool IsParsed => throw new NotImplementedException();

        public ISpan AddSpan(int begin, int end)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            throw new NotImplementedException();
        }

        public IEnumerator<ISpan> GetEnumerator()
        {
            return Spans.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return Spans.GetEnumerator();
        }

        public void RemoveOverlapingTokens()
        {
            throw new NotImplementedException();
        }

        public List<IToken> ToTokenList()
        {
            throw new NotImplementedException();
        }

        public void WriteAsJson(IJsonWriter jw)
        {
            throw new NotImplementedException();
        }

        public string ToJson()
        {
            throw new NotImplementedException();
        }
    }
}