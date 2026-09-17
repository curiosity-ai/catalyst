using Catalyst.DateTimeRecognition;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.External
{
    public enum RecognizerTypes
    {
        DateTime
    }

    /// <summary>
    /// Finds dates, times, periods and durations in a document and tags the tokens they cover, attaching the
    /// resolved value as entity metadata (timex, type, value / start / end) in the datetimeV2 shape.
    /// </summary>
    public class DateTimeRecognizer : StorableObject<DateTimeRecognizer, DateTimeRecognizerModel>, IEntityRecognizer, IProcess
    {
        private readonly Lazy<DateTimeModel> _dateTimeModel;

        public DateTimeRecognizer(Language language, bool useUsEnglishForEnglish = false) : base(language, 0, "", false)
        {
            _dateTimeModel = new Lazy<DateTimeModel>(() => GetModel(language, useUsEnglishForEnglish), LazyThreadSafetyMode.ExecutionAndPublication);
        }

        private static DateTimeModel GetModel(Language language, bool useUsEnglishForEnglish)
        {
            if (language == Language.Any || language == Language.Unknown) language = Language.English;

            if (!Lexicons.IsSupported(language)) throw new NotSupportedException($"Language not supported: {language}");

            return DateTimeModel.For(language, useUsEnglishForEnglish);
        }

        /// <summary>Whether a date/time vocabulary exists for this language.</summary>
        public static bool IsLanguageSupported(Language language) => Lexicons.IsSupported(language);

        public void Process(IDocument document, CancellationToken cancellationToken = default)
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

            bool found = result.Count > 0;

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
                                    break;
                                }

                                if (tk.Begin > r.End) { break; }
                            }

                            if (begin >= 0 && end >= 0)
                            {
                                var md = r.Values.FirstOrDefault()?.ToDictionary();

                                if (md is object)
                                {
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
