using UID;
using MessagePack;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Catalyst
{
    /// <summary>
    /// Represents the type and tag of an entity.
    /// </summary>
    [JsonObject]
    [MessagePackObject]
    public struct EntityType
    {
        /// <summary>
        /// Gets or sets the type of the entity (e.g., PERSON, ORGANIZATION).
        /// </summary>
        [Key(0)] public string Type { get; set; }

        /// <summary>
        /// Gets or sets the tag of the entity (e.g., Begin, Inside, End, Single).
        /// </summary>
        [Key(1)] public EntityTag Tag { get; set; }

        /// <summary>
        /// Gets or sets the metadata associated with the entity.
        /// </summary>
        [Key(2)] public Dictionary<string, string> Metadata { get; set; }

        /// <summary>
        /// Gets or sets the unique identifier of the target entity if this is a linked entity.
        /// </summary>
        [Key(3)] public UID128 TargetUID { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="EntityType"/> struct.
        /// </summary>
        /// <param name="type">The type of the entity.</param>
        /// <param name="tag">The tag of the entity.</param>
        public EntityType(string type, EntityTag tag)
        {
            Type = type; Tag = tag; Metadata = null; TargetUID = default(UID128);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="EntityType"/> struct with a target UID.
        /// </summary>
        /// <param name="type">The type of the entity.</param>
        /// <param name="tag">The tag of the entity.</param>
        /// <param name="targetUID">The target UID.</param>
        public EntityType(string type, EntityTag tag, UID128 targetUID)
        {
            Type = type; Tag = tag; Metadata = null; TargetUID = targetUID;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="EntityType"/> struct for serialization.
        /// </summary>
        /// <param name="type">The type of the entity.</param>
        /// <param name="tag">The tag of the entity.</param>
        /// <param name="metadata">The metadata.</param>
        /// <param name="targetUID">The target UID.</param>
        [SerializationConstructor]
        public EntityType(string type, EntityTag tag, Dictionary<string, string> metadata, UID128 targetUID)
        {
            Type = type;
            Tag = tag;
            Metadata = metadata;
            TargetUID = targetUID;
        }
    }

    /// <summary>
    /// Specifies the tag of an entity in IOB, BIO or similar schemes.
    /// </summary>
    public enum EntityTag
    {
        /// <summary>
        /// The beginning of an entity.
        /// </summary>
        Begin = 'B',

        /// <summary>
        /// Inside an entity.
        /// </summary>
        Inside = 'I',

        /// <summary>
        /// The last token of an entity.
        /// </summary>
        End = 'L', //Last

        /// <summary>
        /// Outside an entity.
        /// </summary>
        Outside = 'O',

        /// <summary>
        /// A single-token entity.
        /// </summary>
        Single = 'U', //Unit
    }
}