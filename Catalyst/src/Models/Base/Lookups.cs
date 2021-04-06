using MessagePack;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    [MessagePackObject]
    public class Lookups
    {
        internal static readonly MessagePackSerializerOptions LZ4Standard = MessagePackSerializerOptions.Standard.WithCompression(MessagePackCompression.Lz4Block);

        public Lookups(string name, Language language, string cache, Dictionary<ulong, Entry> entries)
        {
            Name = name;
            Language = language;
            Cache = cache;
            Entries = entries;
        }

        [Key(0)] public string Name { get; set; }
        [Key(1)] public Language Language { get; set; }
        [Key(2)] public string Cache { get; set; }
        [Key(3)] public Dictionary<ulong, Entry> Entries { get; set; }

        [MessagePackObject]
        public struct Entry
        {
            public Entry(byte length, uint begin)
            {
                Length = length;
                Begin = begin;
            }

            public Entry(float probabilityLog10, uint cluster)
            {
                Length = (byte)(-probabilityLog10 * 10);
                Begin = cluster;
            }

            [Key(0)] public byte Length { get; set; }
            [Key(1)] public uint Begin { get; set; }

            [IgnoreMember]
            public float Probability
            {
                get
                {
#if NETCOREAPP3_0 || NETCOREAPP3_1 || NET5_0

                    return MathF.Pow(10f, -(Length / 10f));
#else
                    return (float)Math.Pow(10, -(Length / 10f));
#endif
                }
            }

            [IgnoreMember] public uint Cluster => Begin;
        }

        public static ulong Hash(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += key[i];
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        public static ulong InvariantHash(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += char.ToLowerInvariant(key[i]);
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        public ReadOnlySpan<char> Get(IToken token)
        {
            var hash = Hash(token.ValueAsSpan);

            if(Entries.TryGetValue(hash, out var entry))
            {
                return Cache.AsSpan().Slice((int)entry.Begin, entry.Length);
            }
            
            var invHash = InvariantHash(token.ValueAsSpan);
            
            if(invHash != hash && Entries.TryGetValue(invHash, out entry))
            {
                return Cache.AsSpan().Slice((int)entry.Begin, entry.Length);
            }

            return token.ValueAsSpan;
        }

        public async Task SerializeAsync(Stream stream)
        {
            await MessagePackSerializer.SerializeAsync(stream, this, LZ4Standard);
        }

        public static async Task<Lookups> FromStream(Stream stream)
        {
            return await MessagePackSerializer.DeserializeAsync<Lookups>(stream, LZ4Standard);
        }
    }
}
