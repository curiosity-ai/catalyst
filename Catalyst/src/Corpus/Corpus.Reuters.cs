using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;

namespace Catalyst
{
    public static partial class Corpus
    {
        public static class Reuters
        {
            private const string URL = @"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip";

            public static async Task DownloadAsync()
            {
                var response = await Client.GetAsync(URL);
                using (var dest = await Storage.Current.OpenLockedStreamAsync(GetDataPath(), FileAccess.Write))
                {
                    await response.Content.CopyToAsync(dest);
                    await dest.FlushAsync();
                }
            }

            private static string GetDataPath()
            {
                return Storage.Current.GetDataPath(Language.English, nameof(Reuters), 0, "reuters-dataset-nltk");
            }

            public static async Task<(IDocument[] trainDocuments, IDocument[] testDocuments)> GetAsync()
            {
                var path = GetDataPath();
                
                if (! await Storage.Current.ExistsAsync(path))
                {
                    await DownloadAsync();
                }

                var categories = new Dictionary<string, string[]>();
                var trainDocs = new List<IDocument>();
                var testDocs  = new List<IDocument>();

                using (var dest = await Storage.Current.OpenLockedStreamAsync(GetDataPath(), System.IO.FileAccess.Read))
                using (var zip = new ZipArchive(dest, ZipArchiveMode.Read))
                {
                    using (var cat = zip.GetEntry(@"reuters/cats.txt").Open())
                    using(var sr = new StreamReader(cat))
                    {
                        while(!sr.EndOfStream)
                        {
                            var line = await sr.ReadLineAsync();
                            var parts = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            categories.Add(parts[0], parts.Skip(1).ToArray());
                        }
                    }

                    foreach(var kv in categories)
                    {
                        IDocument doc;
                        using (var cat = zip.GetEntry(@"reuters/" + kv.Key).Open())
                        using (var sr = new StreamReader(cat))
                        {
                            var txt = await sr.ReadToEndAsync();
                            txt = txt.Replace('\n', ' ');
                            doc = new Document(txt, Language.English);
                            doc.Labels.AddRange(kv.Value);
                        }

                        if (kv.Key.StartsWith("train"))
                        {
                            trainDocs.Add(doc);
                        }
                        else
                        {
                            testDocs.Add(doc);
                        }
                    }
                }

                return (trainDocs.ToArray(), testDocs.ToArray());
            }
        }
    }
}
