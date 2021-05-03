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
        internal static async Task RunAsync(string spacyLookupsData, string languagesDirectory)
        {
            var rootLangFolder = Path.Combine(spacyLookupsData, @"spacy_lookups_data\data\");
            if (!Directory.Exists(rootLangFolder)) throw new Exception("data directory not found");
            
            //TODO Handle rules data

            var tasks = new List<Task>();

            foreach(var (file, language) in Directory.GetFiles(rootLangFolder, "*_lemma_lookup*.json").Select(f => (file:f, language: Languages.CodeToEnum(Path.GetFileName(f).Substring(0,2)))))
            {
                tasks.Add(Task.Run(async () =>
                {
                    Console.WriteLine($"\n\n\nBegin processing {file}\n\n");

                    var name = Path.GetFileNameWithoutExtension(file);

                    var resDir = Path.Combine(languagesDirectory, language.ToString(), "Resources");
                    Directory.CreateDirectory(resDir);

                    var outputFile = Path.Combine(resDir, name + ".bin");

                    if (File.Exists(outputFile))
                    {
                        Console.WriteLine("Skipping...");
                        return;
                    }

                    var map = JsonConvert.DeserializeObject<Dictionary<string, string>>(FixWordsAsArrays(await File.ReadAllTextAsync(file)));
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

                    var lookup = new Lookups(name, language, new string(buffer), entries);

                    using (var f = File.OpenWrite(outputFile))
                    {
                        await lookup.SerializeAsync(f);
                    }
                    Console.WriteLine($"\n\n\nWrote {outputFile}\n\n");
                }));
            }

            foreach (var (file, language) in Directory.GetFiles(rootLangFolder, "*_lexeme_cluster*.json").Select(f => (file: f, language: Languages.CodeToEnum(Path.GetFileName(f).Substring(0, 2)))))
            {
                tasks.Add(Task.Run(async () =>
                {
                    var probFile = file.Replace("_lexeme_cluster", "_lexeme_prob");
                    
                    if (!File.Exists(probFile))
                    {
                        return;
                    }

                    Console.WriteLine($"\n\n\nBegin processing {file} + {probFile}\n\n");

                    var name = Path.GetFileNameWithoutExtension(file);

                    var resDir = Path.Combine(languagesDirectory, language.ToString(), "Resources");
                    Directory.CreateDirectory(resDir);

                    var outputFile = Path.Combine(resDir, name + "_prob.bin");

                    if (File.Exists(outputFile))
                    {
                        Console.WriteLine("Skipping...");
                        return;
                    }

                    var cluster = JsonConvert.DeserializeObject<Dictionary<string, uint>>(FixWordsAsArrays(await File.ReadAllTextAsync(file)));
                    var prob = JsonConvert.DeserializeObject<Dictionary<string, float>>(FixWordsAsArrays(await File.ReadAllTextAsync(probFile)));

                    var entries = new Dictionary<ulong, Lookups.Entry>();
                    int count = 0;

                    foreach (var (k, v) in cluster)
                    {
                        var keyHash = Lookups.Hash(k);
                        var invKeyHash = Lookups.InvariantHash(k);

                        var probVal = prob.TryGetValue(k, out var p) ? p : -25f;

                        entries.TryAdd(keyHash, new Lookups.Entry(probVal, v));
                        if (invKeyHash != keyHash)
                        {
                            entries.TryAdd(invKeyHash, new Lookups.Entry(probVal, v));
                        }
                        count++;
                        if (count % 1000 == 0) Console.WriteLine($"\nAt {count} of {cluster.Count}");
                    }

                    var lookup = new Lookups(name, language, null, entries);

                    using (var f = File.OpenWrite(outputFile))
                    {
                        await lookup.SerializeAsync(f);
                    }

                    Console.WriteLine($"\n\n\nWrote {outputFile}\n\n");
                }));
            }


            await Task.WhenAll(tasks);
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
