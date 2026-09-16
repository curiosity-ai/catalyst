using System;
using System.Collections.Generic;
using System.Linq;
using Catalyst.DateTimeRecognition;
using Microsoft.Recognizers.Text;
using Microsoft.Recognizers.Text.DateTime;
using Microsoft.Recognizers.Text.DateTime.English;
using Mosaik.Core;
using MsDateTimeModel = Microsoft.Recognizers.Text.DateTime.DateTimeModel;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>A recognised expression, normalised so the two engines and the spec can be compared field by field.</summary>
    public sealed class Hit
    {
        public string                            Text   { get; set; }
        public int                               Start  { get; set; }
        public int                               End    { get; set; }
        public string                            Type   { get; set; }
        public List<Dictionary<string, string>>  Values { get; set; } = new List<Dictionary<string, string>>();

        public string Key => $"{Start}..{End}:{Type}";

        public string Describe()
        {
            var vs = string.Join(" | ", Values.Select(v => string.Join(",", v.OrderBy(k => k.Key).Select(k => k.Key + "=" + k.Value))));
            return $"[{Start}..{End}] {Type} \"{Text}\" {{{vs}}}";
        }
    }

    public static class Engines
    {
        // ------------------------------------------------------------------ the new engine

        private static readonly Dictionary<string, Catalyst.DateTimeRecognition.DateTimeModel> _catalyst = new Dictionary<string, Catalyst.DateTimeRecognition.DateTimeModel>();

        public static List<Hit> RunCatalyst(string language, string input, System.DateTime reference)
        {
            Catalyst.DateTimeRecognition.DateTimeModel model;

            lock (_catalyst)
            {
                if (!_catalyst.TryGetValue(language, out model))
                {
                    model = language switch
                    {
                        "English"       => Catalyst.DateTimeRecognition.DateTimeModel.For(Language.English, useUsEnglishForEnglish: true),
                        "EnglishOthers" => Catalyst.DateTimeRecognition.DateTimeModel.For(Language.English, useUsEnglishForEnglish: false),
                        _               => Catalyst.DateTimeRecognition.DateTimeModel.For(ParseLanguage(language)),
                    };

                    _catalyst[language] = model;
                }
            }

            return model.Parse(input, reference)
                        .Select(e => new Hit
                        {
                            Text   = e.Text,
                            Start  = e.Start,
                            End    = e.End,
                            Type   = e.TypeName,
                            Values = e.Values.Select(v => v.ToDictionary()).ToList(),
                        })
                        .ToList();
        }

        private static Language ParseLanguage(string name) => Enum.Parse<Language>(name, ignoreCase: true);

        // ------------------------------------------------------------------ Microsoft.Recognizers.Text

        private static readonly Dictionary<string, MsDateTimeModel> _microsoft = new Dictionary<string, MsDateTimeModel>();

        public static List<Hit> RunMicrosoft(string language, string input, System.DateTime reference)
        {
            MsDateTimeModel model;

            lock (_microsoft)
            {
                if (!_microsoft.TryGetValue(language, out model))
                {
                    model = BuildMicrosoft(language);
                    _microsoft[language] = model;
                }
            }

            return model.Parse(input, reference)
                        .Select(r => new Hit
                        {
                            Text   = r.Text,
                            Start  = r.Start,
                            End    = r.End,
                            Type   = r.TypeName,
                            Values = ExtractValues(r),
                        })
                        .ToList();
        }

        private static List<Dictionary<string, string>> ExtractValues(ModelResult r)
        {
            if (r.Resolution is object && r.Resolution.Count > 0 && r.Resolution.First().Value is List<Dictionary<string, string>> list)
            {
                return list;
            }

            return new List<Dictionary<string, string>>();
        }

        private static MsDateTimeModel BuildMicrosoft(string language)
        {
            var options = DateTimeOptions.None;

            var culture = language switch
            {
                "English"       => Culture.English,
                "EnglishOthers" => Culture.EnglishOthers,
                _               => Culture.English,
            };

            return new MsDateTimeModel(
                new BaseMergedDateTimeParser(new EnglishMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(culture, options, dmyDateFormat: false))),
                new BaseMergedDateTimeExtractor(new EnglishMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(culture, options, dmyDateFormat: false))));
        }

        public static List<Hit> FromSpec(SpecCase c)
        {
            return c.Results.Select(r => new Hit
            {
                Text   = r.Text,
                Start  = r.Start,
                End    = r.End,
                Type   = r.Type,
                Values = r.Resolution?.Values ?? new List<Dictionary<string, string>>(),
            }).ToList();
        }
    }
}
