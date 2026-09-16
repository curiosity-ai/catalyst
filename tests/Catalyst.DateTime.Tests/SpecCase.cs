using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>One case of the Microsoft.Recognizers.Text `Specs/DateTime/&lt;language&gt;/DateTimeModel.json` suite.</summary>
    public sealed class SpecCase
    {
        public string                Input        { get; set; }
        public SpecContext           Context      { get; set; }
        public string                NotSupported { get; set; }
        public List<SpecResult>      Results      { get; set; } = new List<SpecResult>();

        public DateTime ReferenceDateTime => Context?.ReferenceDateTime ?? new DateTime(2016, 11, 7);
    }

    public sealed class SpecContext
    {
        public DateTime? ReferenceDateTime { get; set; }
    }

    public sealed class SpecResult
    {
        public string                     Text       { get; set; }
        public int                        Start      { get; set; }
        public int                        End        { get; set; }
        public string                     TypeName   { get; set; }
        public SpecResolution             Resolution { get; set; }

        public string Type => TypeName;
    }

    public sealed class SpecResolution
    {
        public List<Dictionary<string, string>> Values { get; set; } = new List<Dictionary<string, string>>();
    }

    public static class SpecLoader
    {
        private static readonly JsonSerializerOptions Options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling         = JsonCommentHandling.Skip,
            AllowTrailingCommas         = true,
        };

        public static string SpecFolder => Path.Combine(AppContext.BaseDirectory, "Specs");

        public static IReadOnlyList<SpecCase> Load(string language)
        {
            var path = Path.Combine(SpecFolder, language + ".json");

            if (!File.Exists(path)) return Array.Empty<SpecCase>();

            var json = File.ReadAllText(path);
            var all  = JsonSerializer.Deserialize<List<SpecCase>>(json, Options) ?? new List<SpecCase>();

            // The suite marks a handful of cases as unsupported outside .NET; those still run here.
            return all.Where(c => !string.IsNullOrEmpty(c.Input)).ToList();
        }
    }
}
