using Microsoft.Recognizers.Text;
using Microsoft.Recognizers.Text.DateTime;
using Microsoft.Recognizers.Text.DateTime.English;
using Microsoft.Recognizers.Text.DateTime.Chinese;
using Microsoft.Recognizers.Text.DateTime.Dutch;
using Microsoft.Recognizers.Text.DateTime.French;
using Microsoft.Recognizers.Text.DateTime.German;
using Microsoft.Recognizers.Text.DateTime.Hindi;
using Microsoft.Recognizers.Text.DateTime.Italian;
using Microsoft.Recognizers.Text.DateTime.Portuguese;
using Microsoft.Recognizers.Text.DateTime.Spanish;
using Microsoft.Recognizers.Text.DateTime.Turkish;

using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Catalyst.External
{
    public enum RecognizerTypes
    {
        DateTime
    }

    public class DateTimeRecognizer : StorableObject<DateTimeRecognizer, DateTimeRecognizerModel>, IEntityRecognizer, IProcess
    {
        private readonly Lazy<DateTimeModel> _dateTimeModel;

        public DateTimeRecognizer(Language language, bool useUsEnglishForEnglish = false) : base(language, 0, "", false)
        {
            _dateTimeModel = new Lazy<DateTimeModel>(() => GetModel(language, useUsEnglishForEnglish), isThreadSafe: true);
        }

        private DateTimeModel GetModel(Language language, bool useUsEnglishForEnglish)
        {
            //  We use a similar logic to the official code, but instead of pre-allocating all languages, we only load the one we care for here
            // https://github.com/microsoft/Recognizers-Text/blob/ed73b6604ea7140799831f73d6e31f764629bafe/.NET/Microsoft.Recognizers.Text.DateTime/DateTimeRecognizer.cs#L50
            // This is due the way they build the models, that ends up allocating all static fields of the per-language class by mistake.

            var options = DateTimeOptions.None;

            switch (language)
            {
                case Language.English:    return GetEnglishModel(options, useUsEnglishForEnglish);
                case Language.Chinese:    return GetChineseModel(options);
                case Language.Spanish:    return GetSpanishModel(options);
                case Language.French:     return GetFrenchModel(options);
                case Language.Portuguese: return GetPortugueseModel(options);
                case Language.German:     return GetGermanModel(options);
                case Language.Italian:    return GetItalianModel(options);
                case Language.Turkish:    return GetTurkishModel(options);
                case Language.Hindi:      return GetHindiModel(options);
                case Language.Dutch:      return GetDutchModel(options);
                default: throw new Exception($"Language not supported: {language}");
            }
        }

        private static DateTimeModel GetEnglishModel(DateTimeOptions options, bool useUsEnglishForEnglish)
        {
            if (useUsEnglishForEnglish)
            {
                return new DateTimeModel(
                        new BaseMergedDateTimeParser(
                            new EnglishMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.English, options, dmyDateFormat: false))),
                        new BaseMergedDateTimeExtractor(
                            new EnglishMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.English, options, dmyDateFormat: false))));
            }
            else
            {
                return new DateTimeModel(
                        new BaseMergedDateTimeParser(
                            new EnglishMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.EnglishOthers, options, dmyDateFormat: false))),
                        new BaseMergedDateTimeExtractor(
                            new EnglishMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.EnglishOthers, options, dmyDateFormat: false))));
            }
        }

        private static DateTimeModel GetChineseModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseCJKMergedDateTimeParser(
                        new ChineseMergedParserConfiguration(new ChineseCommonDateTimeParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Chinese, options)))),
                    new BaseCJKMergedDateTimeExtractor(
                        new ChineseMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Chinese, options))));
        }

        private static DateTimeModel GetSpanishModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new SpanishMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Spanish, options))),
                    new BaseMergedDateTimeExtractor(
                        new SpanishMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Spanish, options))));
        }

        private static DateTimeModel GetFrenchModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new FrenchMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.French, options))),
                    new BaseMergedDateTimeExtractor(
                        new FrenchMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.French, options))));
        }

        private static DateTimeModel GetPortugueseModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new PortugueseMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Portuguese, options))),
                    new BaseMergedDateTimeExtractor(
                        new PortugueseMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Portuguese, options))));
        }

        private static DateTimeModel GetGermanModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new GermanMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.German, options))),
                    new BaseMergedDateTimeExtractor(
                        new GermanMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.German, options))));
        }

        private static DateTimeModel GetItalianModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new ItalianMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Italian, options))),
                    new BaseMergedDateTimeExtractor(
                        new ItalianMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Italian, options))));
        }

        private static DateTimeModel GetTurkishModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new TurkishMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Turkish, options))),
                    new BaseMergedDateTimeExtractor(
                        new TurkishMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Turkish, options))));
        }

        private static DateTimeModel GetHindiModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                    new BaseMergedDateTimeParser(
                        new HindiMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Hindi, options))),
                    new BaseMergedDateTimeExtractor(
                        new HindiMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Hindi, options))));
        }

        private static DateTimeModel GetDutchModel(DateTimeOptions options)
        {
            return new DateTimeModel(
                     new BaseMergedDateTimeParser(
                         new DutchMergedParserConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Dutch, options))),
                     new BaseMergedDateTimeExtractor(
                         new DutchMergedExtractorConfiguration(new BaseDateTimeOptionsConfiguration(Culture.Dutch, options))));
        }

        public void Process(IDocument document)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return new[] { nameof(RecognizerTypes.DateTime) };
        }

        public static new Task<bool> ExistsAsync(Language language, int version, string tag)
        {
            return Task.FromResult(true);
        } // Needs to say it exists, otherwise when calling StoredObjectInfo.ExistsAsync(Language language, int version, string tag), it will fail to load this model

        public static new Task<DateTimeRecognizer> FromStoreAsync(Language language, int version, string tag)
        {
            return Task.FromResult(new DateTimeRecognizer(language));
        }

        public bool RecognizeEntities(IDocument document)
        {
            var result = _dateTimeModel.Value.Parse(document.Value, DateTime.Now);

            bool found = result.Any();
            if (found)
            {
                foreach (var r in result)
                {
                    foreach (var span in document)
                    {
                        if (span.Begin <= r.Start && span.End >= r.End)
                        {
                            //Found, add tokens now
                            int begin = -1;
                            int end = -1;
                            foreach (var tk in span)
                            {
                                if (tk.End < r.Start)
                                {
                                    continue;
                                }

                                if (begin < 0 && tk.Begin >= r.Start)
                                {
                                    begin = tk.Index;
                                }

                                if (begin >= 0 && tk.End <= r.End)
                                {
                                    end = tk.Index;
                                }

                                if (begin >= 0 && tk.Begin >= r.End)
                                {
                                    //end = tk.Index-1;
                                    break;
                                }

                                if (tk.Begin > r.End) { break; }
                            }

                            if (begin >= 0 && end >= 0)
                            {
                                if (r.Resolution?.FirstOrDefault().Value is List<Dictionary<string, string>> list)
                                {
                                    var md = list.First();
                                    if (begin == end)
                                    {
                                        span[begin].AddEntityType(new EntityType(nameof(RecognizerTypes.DateTime), EntityTag.Single) { Metadata = md });
                                    }
                                    else
                                    {
                                        span[begin].AddEntityType(new EntityType(nameof(RecognizerTypes.DateTime), EntityTag.Begin) { Metadata = md });
                                        span[end].AddEntityType(new EntityType(nameof(RecognizerTypes.DateTime), EntityTag.End));
                                        for (int i = begin + 1; i < end; i++)
                                        {
                                            span[i].AddEntityType(new EntityType(nameof(RecognizerTypes.DateTime), EntityTag.Inside));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return found;
        }
    }

    public class DateTimeRecognizerModel : StorableObjectData
    {
    }
}