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
    public static class PrepareSpacyLookups
    {
        internal static async Task RunAsync(string spacyLookupsData)
        {
            var rootLangFolder = Path.Combine(spacyLookupsData, @"spacy_lookups_data\data\");
            if (!Directory.Exists(rootLangFolder)) throw new Exception("data directory not found");
            
            var lookupsFolder = Path.Combine(Directory.GetCurrentDirectory(), "Lookups");
            
            Directory.CreateDirectory(lookupsFolder);

            //TODO Handle rules data

            foreach(var (file, language) in Directory.GetFiles(rootLangFolder, "*_lemma_lookup*.json").Select(f => (file:f, language: Languages.CodeToEnum(Path.GetFileName(f).Substring(0,2)))))
            {
                Console.WriteLine($"\n\n\nBegin processing {file}\n\n");

                var name = Path.GetFileNameWithoutExtension(file);
                var outputFile = Path.Combine(lookupsFolder, name + ".bin");
                
                if (File.Exists(outputFile))
                {
                    Console.WriteLine("Skipping...");
                    continue;
                }

                var map = JsonConvert.DeserializeObject<Dictionary<string, string>>(FixWordsAsArrays(await File.ReadAllTextAsync(file)));
                var buffer = new char[map.Values.Sum(k => k.Length)];
                var bufferLength = 0;

                var entries = new Dictionary<ulong, Lookups.Entry>();
                int count = 0;

                foreach(var (k,v) in map.OrderByDescending(kv => kv.Value.Length).ThenBy(kv => kv.Value))
                {
                    var keyHash = Lookups.Hash(k);
                    var invKeyHash = Lookups.InvariantHash(k);

                    var index = buffer.AsSpan(0, bufferLength).IndexOf(v);

                    if(index < 0)
                    {
                        v.AsSpan().CopyTo(buffer.AsSpan(bufferLength, v.Length));
                        entries.TryAdd(keyHash, new Lookups.Entry((byte)v.Length, (uint)bufferLength));
                        if(invKeyHash != keyHash)
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
                        Console.Write(".");
                    }
                    count++;
                    if (count % 1000 == 0) Console.WriteLine($"\nAt {count} of {map.Count}");
                }

                Array.Resize(ref buffer, bufferLength);

                var lookup = new Lookups(name, language, new string(buffer), entries);
                

                using (var f = File.OpenWrite(outputFile))
                {
                    await lookup.SerializeAsync(f);
                    Console.WriteLine($"\n\n\nWrote {outputFile}\n\n");
                }
            }
        }

        private static string FixWordsAsArrays(string v)
        {
            if(v.IndexOf("[\"") > 0)
            {
                return v.Replace("[", "").Replace("]", ""); 
            }
            return v;
        }
    }
}
