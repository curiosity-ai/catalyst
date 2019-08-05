using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public static class TranslationDictionary
    {
        public static string Path = @"./Corpus/MUSE";

        public static async Task DownloadMUSEDataAsync()
        {
            Directory.CreateDirectory(Path);
            using(var c = new HttpClient())
            {
                var response = await c.GetAsync("https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz");
                var stream = await response.Content.ReadAsStreamAsync();
                using(var tempStream = Storage.Current.GetTempStream())
                {
                    await stream.CopyToAsync(tempStream);
                    tempStream.Seek(0, SeekOrigin.Begin);
                    using (var gzipStream = new GZipInputStream(tempStream))
                    using (var tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
                    {
                        tarArchive.ExtractContents(Path);
                        tarArchive.Close();
                        gzipStream.Close();
                        tempStream.Close();
                    }
                }
            }
        }

        public static Dictionary<string, string> GetDictionary(Language from, Language to, int N)
        {
            var f = Languages.EnumToCode(from);
            var t = Languages.EnumToCode(to);
            string path = System.IO.Path.Combine(Path, $"{f}-{t}.txt");

            if (!File.Exists(path))
            {
                throw new Exception("Need to download MUSE data first. Set TranslationDictionary.Path to the desired directory, and call await TranslationDictionary.DownloadMUSEDataAsync()");
            }

            var lines = File.ReadAllLines(path);
            var d = new Dictionary<string, string>(lines.Length);
            var unique = lines.Select(l => l.Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries))
                              .GroupBy(s => s[0])
                              .Where(g => g.Count() == 1)
                              .Take(N)
                              .ToDictionary(g => g.First()[0], g => g.First()[1]);
            return unique;
        }
    }
}