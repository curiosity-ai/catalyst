using ICSharpCode.SharpZipLib.BZip2;
using Mosaik.Core;
using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using UID;


namespace Catalyst.Training
{
    public class TrainWikiNER
    {
        private static ILogger Logger = ApplicationLogging.CreateLogger<TrainWikiNER>();

        public static async Task TrainAsync(string basePath, Language language, int version, string tag)
        {
            var langMarker = "-" + Languages.EnumToCode(language) + "-";
            var files = Directory.EnumerateFiles(basePath, "*.bz2").Where(f => f.Contains(langMarker));

            var documents = new List<IDocument>();

            foreach (var f in files)
            {
                documents.AddRange(ReadFile(f));
            }

            var pos = await AveragePerceptronTagger.FromStoreAsync(language, -1, "");

            using (var m = new Measure(Logger, "Tagging documents"))
            {
                Parallel.ForEach(documents, doc => pos.Predict(doc));
            }

            var aper = new AveragePerceptronEntityRecognizer(language, version, tag, new string[] { "Person", "Organization", "Location" }, ignoreCase:false );

            aper.Train(documents);
            await aper.StoreAsync();
        }

        private static IEnumerable<IDocument> ReadFile(string fn)
        {
            var MapTypes = new Dictionary<string, string>()
            {
                ["I-ORG"] = "Organization",
                ["I-PER"] = "Person",
                ["I-MISC"] = AveragePerceptronEntityRecognizer.TagOutside.ToString(),
                ["I-LOC"] = "Location",
                ["O"] = AveragePerceptronEntityRecognizer.TagOutside.ToString(),
            };

            var sb = new StringBuilder();
            var tagHashToTag = new Dictionary<uint, string>();
            using (var stream = File.OpenRead(fn))
            {
                Stream decompressedStream = stream;
                if (fn.EndsWith("bz2")) { decompressedStream = new BZip2InputStream(stream); }
                using (var reader = new StreamReader(decompressedStream))
                {
                    //Original tags are following the Inside Outside model
                    string line;

                    while ((line = reader.ReadLine()) is object)
                    {
                        if (!string.IsNullOrWhiteSpace(line))
                        {
                            var lineSpan = line.AsSpan();
                            int pos = 0;

                            //First assemble the document text
                            sb.Clear();

                            while(true)
                            {
                                var space = lineSpan.Slice(pos).IndexOf(' ');

                                if (space < 0)
                                {
                                    break;
                                }
                                var token = lineSpan.Slice(pos, space - 1);
                                var sep = token.IndexOf('|');
                                var word = token.Slice(0, sep);
                                sb.Append(word).Append(' ');
                                pos += space+1;
                            }

                            var text = sb.ToString();

                            var doc = new Document(text);
                            if (doc.Value.Length != text.Length)
                            {
                                continue;//There were control characters in the text, we just ignore it here...
                            }
                            var span = doc.AddSpan(0, text.Length - 1);


                            pos = 0;
                            int curPos = 0;
                            var prevTag = "O";
                            var nextTag = "O";

                            //Now add the tags
                            while(true)
                            {
                                var space = lineSpan.Slice(pos).IndexOf(' ');

                                if (space < 0)
                                {
                                    break;
                                }

                                var nextSpace = lineSpan.Slice(pos + space + 1).IndexOf(' ');

                                if (nextSpace > 0)
                                {
                                    var nextToken = lineSpan.Slice(pos + space +1, nextSpace);
                                    var nextSep = nextToken.LastIndexOf('|');
                                    nextTag = nextToken.Slice(nextSep + 1).ToString();
                                }
                                else
                                {
                                    nextTag = "O";
                                }

                                var token    = lineSpan.Slice(pos, space);
                                var sep      = token.IndexOf('|');
                                var lastSep = token.LastIndexOf('|');
                                var word = token.Slice(0, sep);
                                var rawtag = token.Slice(lastSep + 1);

                                string tag;

                                if(rawtag.StartsWith("B-"))
                                {
                                    var h = rawtag.Hash32();

                                    if(!tagHashToTag.TryGetValue(h, out tag))
                                    {
                                        tag = "I-" + rawtag.Slice(2).ToString();
                                        tagHashToTag[h] = tag;
                                    }
                                    //Replace with "I-"
                                }
                                else if (rawtag.EndsWith("MISC"))
                                {
                                    //Treat as "O";
                                    tag = "O";
                                }
                                else
                                {
                                    var h = rawtag.Hash32();

                                    if (!tagHashToTag.TryGetValue(h, out tag))
                                    {
                                        tag = rawtag.ToString();
                                        tagHashToTag[h] = tag;
                                    }
                                }

                                int begin = curPos - span.Begin;
                                int end = begin + word.Length - 1;
                                curPos += word.Length + 1;
                                EntityTag extra = EntityTag.Outside;

                                bool hasTag = tag != "O";

                                if (hasTag)
                                {
                                    bool equalBefore = tag == prevTag;
                                    bool equalNext = tag == nextTag;

                                    if (!(equalBefore || equalNext)) { extra = EntityTag.Single; }
                                    if (equalBefore && equalNext)    { extra = EntityTag.Inside; }
                                    if (equalBefore && !equalNext)   { extra = EntityTag.End; }
                                    if (!equalBefore && equalNext)   { extra = EntityTag.Begin; }
                                }

                                var newToken = span.AddToken(begin, end);

                                if (!MapTypes.ContainsKey(tag))
                                {
                                    Logger.LogError("Missing tag: {TAG}", tag);
                                    throw new Exception($"Missing tag: {tag}");
                                }

                                newToken.AddEntityType(new EntityType(MapTypes[tag], extra));
                                prevTag = tag;
                                pos += space + 1;
                            }

                            doc.TrimTokens();
                            yield return doc;
                        }
                    }
                }
            }
        }
    }
}
