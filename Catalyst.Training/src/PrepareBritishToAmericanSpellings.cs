using Catalyst.Models;
using MessagePack;
using Mosaik.Core;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Training
{
    public static class PrepareBritishToAmericanSpellings
    {
        internal static async Task RunAsync(string holdOffHungerData, string languagesDirectory)
        {
            var rootFolder = Path.Combine(holdOffHungerData, @"lib\Words\AmericanBritish");


            var resDir = Path.Combine(languagesDirectory, Language.English.ToString(), "Resources");

            Directory.CreateDirectory(resDir);

            if (!Directory.Exists(rootFolder)) throw new Exception("data directory not found");

            //Files are American -> British mapping

            var mapUK2US = new Dictionary<string, string>();

            var mapUS2UK = new Dictionary<string, string>();

            foreach (char c in Enumerable.Range('A', 'Z' - 'A' + 1))
            {
                var lines = await File.ReadAllLinesAsync(Path.Combine(rootFolder, $"AmericanBritishWords_{c}.php"));
                int count = 0;
                foreach (var line in lines.SkipWhile(l => !l.EndsWith("return [")).Skip(1).TakeWhile(l => !l.EndsWith("];")))
                {
                    // 'abakumov'=>['abakumov', 'abakumoff',],
                    // 'analogization'=>'analogisation',

                    //US is always Z

                    var l = line.Replace("=>", "§").Replace("[", "").Replace("]", "").Replace("\\'", "‡").Replace("'", "\"").Replace("‡", "'").Replace(",", "").Trim('\t');

                    var parts = l.Split(new char[] { '"' }, StringSplitOptions.RemoveEmptyEntries);
                    var spellingUS = parts[0];
                    var sep = parts[1];
                    var spellingsUK = parts.Skip(2).Where(p => !string.IsNullOrWhiteSpace(p));

                    mapUS2UK[spellingUS] = spellingsUK.First();

                    foreach (var v in spellingsUK)
                    {
                        mapUK2US[v] = spellingUS;
                    }
                    count++;
                }
                Console.WriteLine($"{c} = {count}");
            }

            await CreateLookup(resDir, "en_us2uk", mapUS2UK);
            await CreateLookup(resDir, "en_uk2us", mapUK2US);

            Console.WriteLine("Done");
        }

        private static async Task CreateLookup(string resDir, string name, Dictionary<string, string> map)
        {
            var outputFile = Path.Combine(resDir, name + ".bin");

            var buffer = new char[map.Values.Sum(k => k.Length)];
            var bufferLength = 0;

            var entries = new Dictionary<ulong, Lookups.Entry>();
            int count = 0;

            foreach (var (k, v) in map.OrderByDescending(kv => kv.Value.Length).ThenBy(kv => kv.Value))
            {
                var keyHash = Lookups.Hash(k);
                var invKeyHash = Lookups.InvariantHash(k);

                var index = buffer.AsSpan(0, bufferLength).IndexOf(v);

                if (index < 0)
                {
                    v.AsSpan().CopyTo(buffer.AsSpan(bufferLength, v.Length));
                    entries.TryAdd(keyHash, new Lookups.Entry((byte)v.Length, (uint)bufferLength));
                    if (invKeyHash != keyHash)
                    {
                        entries.TryAdd(invKeyHash, new Lookups.Entry((byte)v.Length, (uint)bufferLength));
                    }
                    bufferLength += v.Length;
                    Console.Write("+");
                }
                else
                {
                    entries.TryAdd(keyHash, new Lookups.Entry((byte)v.Length, (uint)index));
                    if (invKeyHash != keyHash)
                    {
                        entries.TryAdd(invKeyHash, new Lookups.Entry((byte)v.Length, (uint)index));
                    }
                    //Console.Write(".");
                }
                count++;
                if (count % 1000 == 0) Console.WriteLine($"\nAt {count} of {map.Count}");
            }

            Array.Resize(ref buffer, bufferLength);

            var lookup = new Lookups(name, Language.English, new string(buffer), entries);

            using (var f = File.OpenWrite(outputFile))
            {
                await lookup.SerializeAsync(f);
            }
        }
    }
}
