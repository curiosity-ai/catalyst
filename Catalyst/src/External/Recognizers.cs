using Microsoft.Recognizers.Text;
using Mosaik.Core;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst.External
{
    public enum RecognizerTypes
    {
        DateTime
    }

    public class DateTimeRecognizer : StorableObject<DateTimeRecognizer, DateTimeRecognizerModel>, IEntityRecognizer, IProcess
    {
        public DateTimeRecognizer(Language language) : base(language, 0, "", false)
        {
            var lang = Languages.EnumToCode(language);
            Data.Culture = Culture.GetSupportedCultureCodes().Where(l => l.StartsWith(lang)).FirstOrDefault();
        }

        public void Process(IDocument document)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return new[] { nameof(RecognizerTypes.DateTime) };
        }

        public bool RecognizeEntities(IDocument document)
        {
            var result = Microsoft.Recognizers.Text.DateTime.DateTimeRecognizer.RecognizeDateTime(document.Value, Data.Culture);
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
                                var list = r.Resolution.First().Value as List<Dictionary<string, string>>;
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
            return found;
        }
    }

    public class DateTimeRecognizerModel : StorableObjectData
    {
        public string Culture { get; set; }
    }
}