using MessagePack;
using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Catalyst
{
    internal static class SerializationHelper
    {
        private static readonly MessagePackSerializerOptions LZ4Standard = MessagePackSerializerOptions.Standard.WithCompression(MessagePackCompression.Lz4Block);
        internal static Func<object, byte[]> CreateSerializer(Type type) => source => MessagePackSerializer.Serialize(type, source, LZ4Standard);
        internal static Func<byte[], object> CreateDeserializer(Type type) => serialized => MessagePackSerializer.Deserialize(type, serialized, LZ4Standard);
    }
}