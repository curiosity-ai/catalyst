using UID;
using MessagePack;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Catalyst
{
    [JsonObject]
    [MessagePackObject]
    public partial struct EntityType
    {
        [Key(0)] public string Type { get; set; }
        [Key(1)] public EntityTag Tag { get; set; }
        [Key(2)] public Dictionary<string, string> Metadata { get; set; }
        [Key(3)] public UID128 TargetUID { get; set; }

        public EntityType(string type, EntityTag tag)
        {
            Type = type; Tag = tag; Metadata = null; TargetUID = default(UID128);
        }

        public EntityType(string type, EntityTag tag, UID128 targetUID)
        {
            Type = type; Tag = tag; Metadata = null; TargetUID = targetUID;
        }

        [SerializationConstructor]
        public EntityType(string type, EntityTag tag, Dictionary<string, string> metadata, UID128 targetUID)
        {
            Type = type;
            Tag = tag;
            Metadata = metadata;
            TargetUID = targetUID;
        }
    }

    public enum EntityTag
    {
        Begin = 'B',
        Inside = 'I',
        End = 'L', //Last
        Outside = 'O',
        Single = 'U', //Unit
    }
}