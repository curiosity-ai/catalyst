using UID;
using Mosaik.Core;
using Newtonsoft.Json;
using System.Collections.Generic;

//using MessagePack;

namespace Catalyst
{
    /// <summary>
    /// Represents a document that can be processed by Catalyst.
    /// </summary>
    [MessagePack.Union(0, typeof(Document))]
    public interface IDocument : IEnumerable<Span>
    {
        /// <summary>
        /// Gets or sets the language of the document.
        /// </summary>
        Language Language { get; set; }

        /// <summary>
        /// Gets the length of the document's text.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Gets or sets the text value of the document.
        /// </summary>
        string Value { get; set; }

        /// <summary>
        /// Returns the tokenized value of the document.
        /// </summary>
        /// <param name="mergeEntities">Whether to merge entities into a single token.</param>
        /// <returns>The tokenized text.</returns>
        string TokenizedValue(bool mergeEntities = false);

        /// <summary>
        /// Gets or sets the unique identifier of the document.
        /// </summary>
        UID128 UID { get; set; }

        /// <summary>
        /// Gets the labels associated with the document.
        /// </summary>
        List<string> Labels { get; }

        /// <summary>
        /// Gets the metadata associated with the document.
        /// </summary>
        Dictionary<string, string> Metadata { get; }

        /// <summary>
        /// Gets the spans in the document.
        /// </summary>
        IEnumerable<Span> Spans { get; }

        /// <summary>
        /// Gets the number of spans in the document.
        /// </summary>
        int SpansCount { get; }

        /// <summary>
        /// Gets the number of tokens in the document.
        /// </summary>
        int TokensCount { get; }

        /// <summary>
        /// Gets the number of entities in the document.
        /// </summary>
        int EntitiesCount { get; }

        /// <summary>
        /// Gets a value indicating whether the document has been parsed.
        /// </summary>
        bool IsParsed { get; }

        /// <summary>
        /// Gets or sets the span at the specified index.
        /// </summary>
        /// <param name="key">The index of the span.</param>
        /// <returns>The span at the specified index.</returns>
        Span this[int key] { get; set; }

        /// <summary>
        /// Adds a new span to the document.
        /// </summary>
        /// <param name="begin">The beginning character index of the span.</param>
        /// <param name="end">The ending character index of the span.</param>
        /// <returns>The newly created span.</returns>
        Span AddSpan(int begin, int end);

        /// <summary>
        /// Converts the document to a list of tokens.
        /// </summary>
        /// <returns>A list of tokens.</returns>
        List<IToken> ToTokenList();

        /// <summary>
        /// Clears the document's content.
        /// </summary>
        void Clear();

        /// <summary>
        /// Removes overlapping tokens from the document.
        /// </summary>
        void RemoveOverlapingTokens();

        /// <summary>
        /// Writes the document as JSON to the specified writer.
        /// </summary>
        /// <param name="jw">The JSON writer.</param>
        void WriteAsJson(IJsonWriter jw);

        /// <summary>
        /// Converts the document to its JSON representation.
        /// </summary>
        /// <returns>A JSON string representing the document.</returns>
        string ToJson();
    }
}