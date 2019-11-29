using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Catalyst
{
    internal static class SerializationHelper
    {
        internal static Func<object, byte[]> CreateSerializer(Type type) => source => MessagePack.MessagePackSerializer.Serialize(type, source, MessagePack.MessagePackSerializerOptions.LZ4Standard);
        internal static Func<byte[], object> CreateDeserializer(Type type) => serialized => MessagePack.MessagePackSerializer.Deserialize(type, serialized, MessagePack.MessagePackSerializerOptions.LZ4Standard);
    }
}