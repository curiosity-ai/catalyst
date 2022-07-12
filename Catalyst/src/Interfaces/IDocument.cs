using UID;
using Mosaik.Core;
using Newtonsoft.Json;
using System.Collections.Generic;

//using MessagePack;

namespace Catalyst
{
    [MessagePack.Union(0, typeof(Document))]
    public interface IDocument : IEnumerable<Span>
    {
        Language Language { get; set; }
        int Length { get; }
        string Value { get; set; }
        string TokenizedValue(bool mergeEntities = false);
        UID128 UID { get; set; }
        List<string> Labels { get; }
        Dictionary<string, string> Metadata { get; }
        IEnumerable<Span> Spans { get; }
        int SpansCount { get; }
        int TokensCount { get; }
        int EntitiesCount { get; }
        bool IsParsed { get; }
        Span this[int key] { get; set; }
        Span AddSpan(int begin, int end);
        List<IToken> ToTokenList();
        void Clear();
        void RemoveOverlapingTokens();
        void WriteAsJson(IJsonWriter jw);
        string ToJson();
    }
}