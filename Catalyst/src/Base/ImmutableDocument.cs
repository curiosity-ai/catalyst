using UID;
using Mosaik.Core;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Linq;

//using MessagePack;

namespace Catalyst
{
    /// <summary>
    /// Represents an immutable document in the Catalyst NLP pipeline.
    /// </summary>
    public class ImmutableDocument
    {
        /// <summary>Gets or sets the language of the document.</summary>
        public Language Language { get; set; }

        /// <summary>Gets or sets the text value of the document.</summary>
        public string Value { get; set; }

        /// <summary>Gets or sets the token data for each span in the document.</summary>
        public TokenData[][] TokensData { get; set; }

        /// <summary>Gets or sets the bounds for each span in the document, packed as long (begin &lt;&lt; 32 | end).</summary>
        public long[] SpanBounds { get; set; }

        /// <summary>Gets or sets the metadata for the document.</summary>
        public Dictionary<string, string> Metadata { get; set; }

        /// <summary>Gets or sets the unique identifier for the document.</summary>
        public UID128 UID { get; set; }

        /// <summary>Gets or sets the labels for the document.</summary>
        public string[] Labels { get; set; }

        /// <summary>Gets or sets the entity data for the document.</summary>
        public Dictionary<long, EntityType[]> EntityData { get; set; }

        /// <summary>Gets or sets the token metadata for the document.</summary>
        public Dictionary<long, Dictionary<string, string>> TokenMetadata { get; set; }

        /// <summary>Gets the length of the document's text value.</summary>
        public int Length { get { return Value.Length; } }

        private static long GetTokenIndex(int spanIndex, int tokenIndex)
        {
            return (long)spanIndex << 32 | (long)(uint)tokenIndex;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImmutableDocument"/> class.
        /// </summary>
        public ImmutableDocument(Language language, string value, TokenData[][] tokensData, long[] spanBounds, Dictionary<string, string> metadata, UID128 uID, string[] labels, Dictionary<long, EntityType[]> entityData, Dictionary<long, Dictionary<string, string>> tokenMetadata)
        {
            Language = language;
            Value = value;
            TokensData = tokensData;
            SpanBounds = spanBounds;
            Metadata = metadata;
            UID = uID;
            Labels = labels;
            EntityData = entityData;
            TokenMetadata = tokenMetadata;
        }

        /// <summary>
        /// Converts the current immutable document to a mutable <see cref="Document"/>.
        /// </summary>
        /// <returns>A <see cref="Document"/> instance.</returns>
        public Document ToMutable()
        {
            return Document.FromImmutable(this);
        }

        /// <summary>
        /// Writes the document to an <see cref="IJsonWriter"/>.
        /// </summary>
        /// <param name="jw">The JSON writer.</param>
        public void WriteAsJson(IJsonWriter jw)
        {
            jw.WriteStartObject();

            jw.WritePropertyName(nameof(Language)); jw.WriteStringValue(Languages.EnumToCode(Language));
            jw.WritePropertyName(nameof(Length)); jw.WriteNumberValue(Length);
            jw.WritePropertyName(nameof(Value)); jw.WriteStringValue(Value);

            if (UID.IsNotNull())
            {
                jw.WritePropertyName(nameof(UID)); jw.WriteStringValue(UID);
            }

            if (Metadata is object && Metadata.Any())
            {
                jw.WritePropertyName(nameof(Metadata));
                jw.WriteStartObject();
                foreach (var kv in Metadata)
                {
                    jw.WritePropertyName(kv.Key); jw.WriteStringValue(kv.Value);
                }
                jw.WriteEndObject();
            }

            if (Labels is object && Labels.Length > 0)
            {
                jw.WritePropertyName(nameof(Labels));
                jw.WriteStartArray();
                foreach (var l in Labels)
                {
                    jw.WriteStringValue(l);
                }
                jw.WriteEndArray();
            }

            jw.WritePropertyName(nameof(TokensData));
            jw.WriteStartArray();
            for (int i = 0; i < TokensData.Length; i++)
            {
                var spanData = TokensData[i];
                jw.WriteStartArray();
                for (int j = 0; j < spanData.Length; j++)
                {
                    var tk = spanData[j];
                    long ix = GetTokenIndex(i, j);

                    jw.WriteStartObject();
                    jw.WritePropertyName(nameof(TokenData.Bounds));
                    jw.WriteStartArray();
                    jw.WriteNumberValue(tk.LowerBound);
                    jw.WriteNumberValue(tk.UpperBound);
                    jw.WriteEndArray();

                    if (tk.Tag != PartOfSpeech.NONE)
                    {
                        jw.WritePropertyName(nameof(TokenData.Tag)); jw.WriteStringValue(tk.Tag.ToString());
                    }

                    if (tk.Head >= 0)
                    {
                        jw.WritePropertyName(nameof(TokenData.Head)); jw.WriteNumberValue(tk.Head);
                    }

                    if (tk.Frequency != 0)
                    {
                        jw.WritePropertyName(nameof(TokenData.Frequency)); jw.WriteNumberValue(tk.Frequency);
                    }

                    if (!string.IsNullOrEmpty(tk.Replacement))
                    {
                        jw.WritePropertyName(nameof(TokenData.Replacement)); jw.WriteStringValue(tk.Replacement);
                    }

                    if (TokenMetadata is object)
                    {
                        if (TokenMetadata.TryGetValue(ix, out var tokenMetadata))
                        {
                            if (!(tokenMetadata is null) && tokenMetadata.Any())
                            {
                                jw.WritePropertyName(nameof(Metadata));
                                jw.WriteStartObject();
                                foreach (var kv in tokenMetadata)
                                {
                                    jw.WritePropertyName(kv.Key); jw.WriteStringValue(kv.Value);
                                }
                                jw.WriteEndObject();
                            }
                        }
                    }

                    if (EntityData is object)
                    {
                        if (EntityData.TryGetValue(ix, out var entities))
                        {
                            if (!(entities is null) && entities.Any())
                            {
                                jw.WritePropertyName(nameof(EntityType));
                                jw.WriteStartArray();
                                for (int k = 0; k < entities.Length; k++)
                                {
                                    jw.WriteStartObject();
                                    jw.WritePropertyName(nameof(EntityType.Type)); jw.WriteStringValue(entities[k].Type);
                                    jw.WritePropertyName(nameof(EntityType.Tag)); jw.WriteStringValue(entities[k].Tag.ToString());

                                    if (entities[k].TargetUID.IsNotNull())
                                    {
                                        jw.WritePropertyName(nameof(EntityType.TargetUID)); jw.WriteStringValue(entities[k].TargetUID);
                                    }

                                    if (!(entities[k].Metadata is null) && entities[k].Metadata.Any())
                                    {
                                        jw.WritePropertyName(nameof(EntityType.Metadata));
                                        jw.WriteStartObject();
                                        foreach (var kv in entities[k].Metadata)
                                        {
                                            jw.WritePropertyName(kv.Key); jw.WriteStringValue(kv.Value);
                                        }
                                        jw.WriteEndObject();
                                    }

                                    jw.WriteEndObject();
                                }
                                jw.WriteEndArray();
                            }
                        }
                    }
                    jw.WriteEndObject();
                }
                jw.WriteEndArray();
            }
            jw.WriteEndArray();

            jw.WriteEndObject();
        }
    }
}