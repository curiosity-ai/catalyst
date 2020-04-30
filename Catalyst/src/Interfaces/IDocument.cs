using UID;
using Mosaik.Core;
using Newtonsoft.Json;
using System.Collections.Generic;

//using MessagePack;

namespace Catalyst
{
    [MessagePack.Union(0, typeof(Document))]
    public interface IDocument : IEnumerable<ISpan>
    {
        Language Language { get; set; }
        int Length { get; }
        string Value { get; set; }
        string TokenizedValue(bool mergeEntities = false);
        UID128 UID { get; set; }
        List<string> Labels { get; }
        Dictionary<string, string> Metadata { get; }
        IEnumerable<ISpan> Spans { get; }
        int SpansCount { get; }
        int TokensCount { get; }
        int EntitiesCount { get; }
        bool IsParsed { get; }
        ISpan this[int key] { get; set; }
        ISpan AddSpan(int begin, int end);
        List<IToken> ToTokenList();
        void Clear();
        void RemoveOverlapingTokens();
        void WriteAsJson(IJsonWriter jw);
        string ToJson();
    }
}