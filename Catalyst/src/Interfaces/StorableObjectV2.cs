using MessagePack;
using Mosaik.Core;
using System.IO;
using System.Threading.Tasks;

namespace Catalyst
{
    public abstract class StorableObjectV2<TModel, TData> : StorableObject<TModel, TData> where TData : StorableObjectData, new()
    {
        private static readonly MessagePackSerializerOptions LZ4Standard = MessagePackSerializerOptions.Standard.WithCompression(MessagePackCompression.Lz4Block);

        public StorableObjectV2(Language language, int version, string tag = "", bool compress = true) : base(language, version, tag, compress)
        {

        }

        public virtual Task StoreAsync(Stream stream)
        {
            return MessagePackSerializer.SerializeAsync(stream, Data, LZ4Standard);
        }
        public virtual async Task LoadAsync(Stream stream)
        {
            Data = await MessagePackSerializer.DeserializeAsync<TData>(stream, LZ4Standard);
        }
    }
}