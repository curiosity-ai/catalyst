using ICSharpCode.SharpZipLib.BZip2;
using Mosaik.Core;
using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Training
{
    public class TrainWikiNER
    {
        public static void Train(string basePath, Language language, int version, string tag)
        {
            var langMarker = "-" + Languages.EnumToCode(language) + "-";
            var files = Directory.EnumerateFiles(basePath, "*.bz2").Where(f => f.Contains(langMarker));

            var documents = new List<IDocument>();

            foreach (var f in files)
            {
                documents.AddRange(ReadFile(f));
            }

            var pos = AveragePerceptronTagger.FromStoreAsync(language, -1, "").WaitResult(); //TODO optional version to get always the latest model

            Console.Write("Tagging documents");
            Parallel.ForEach(documents, doc => pos.Predict(doc));
            Console.WriteLine("...done");

            var aper = new AveragePerceptronEntityRecognizer(language, version, tag, new string[] { "Person", "Organization", "Location" }, ignoreCase:false );

            aper.Train(documents);
            aper.StoreAsync().Wait();
        }

        private static IEnumerable<IDocument> ReadFile(string fn)
        {
            var lines = new List<string>();
            using (var stream = File.OpenRead(fn))
            {
                Stream decompressedStream = stream;
                if (fn.EndsWith("bz2")) { decompressedStream = new BZip2InputStream(stream); }
                using (var reader = new StreamReader(decompressedStream))
                {

                    string line;
                    while ((line = reader.ReadLine()) is object)
                    {
                        if (!string.IsNullOrWhiteSpace(line)) { lines.Add(line); }
                    }
                }
            }

            var MapTypes = new Dictionary<string, string>()
            {
                ["I-ORG"] = "Organization",
                ["I-PER"] = "Person",
                ["I-MISC"] = AveragePerceptronEntityRecognizer.TagOutside.ToString(),
                ["I-LOC"] = "Location",
                ["O"] = AveragePerceptronEntityRecognizer.TagOutside.ToString(),
            };

            //Original tags are following the Inside Outside model
            foreach (var line in lines)
            {
                var tokens = line.Split(' ').Select(l => l.Split('|').First()).ToArray();
                var tags = line.Split(' ').Select(l => l.Split('|').Last()).ToArray();

                for (int i = 0; i < tags.Length; i++)
                {
                    if (tags[i].StartsWith("B-")) { tags[i] = tags[i].Replace("B-", "I-"); }
                    if (tags[i].EndsWith("MISC")) { tags[i] = "O"; }
                }


                var text = string.Join(" ", tokens);

                var doc = new Document(text);
                if(doc.Value.Length != text.Length)
                {
                    continue;//There were control characters in the text, we just ignore it here...
                }
                var span = doc.AddSpan(0, text.Length - 1);

                int curPos = 0;
                var prevTag = "O";
                var nextTag = "O";

                for (int i = 0; i < tokens.Length; i++)
                {
                    if (i < tokens.Length - 1) { nextTag = tags[i + 1]; }
                    else { nextTag = "O"; }

                    int begin = curPos - span.Begin;
                    int end = begin + tokens[i].Length - 1;
                    curPos += tokens[i].Length + 1;

                    var tag = tags[i];
                    EntityTag extra = EntityTag.Outside;


                    bool hasTag = tag != "O";

                    if (hasTag)
                    {
                        bool equalBefore = tag == prevTag;
                        bool equalNext = tag == nextTag;

                        if (!(equalBefore || equalNext)) { extra = EntityTag.Single; }
                        if (equalBefore && equalNext)    { extra = EntityTag.Inside; }
                        if (equalBefore && !equalNext)   { extra = EntityTag.End;    }
                        if (!equalBefore && equalNext)   { extra = EntityTag.Begin;  }
                    }

                    var newToken = span.AddToken(begin, end);
                    if (!MapTypes.ContainsKey(tag))
                    {
                        Console.WriteLine(tag);
                        throw new Exception();
                    }
                    newToken.AddEntityType(new EntityType(MapTypes[tag], extra));
                    prevTag = tags[i];
                }
                doc.TrimTokens();
                yield return doc;
            }

        }
    }
}
