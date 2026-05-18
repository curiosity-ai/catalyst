using System;
using System.Collections.Generic;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a token.
    /// </summary>
    public interface IToken
    {
        /// <summary>
        /// Gets or sets the beginning character index of the token.
        /// </summary>
        int Begin { get; set; }

        /// <summary>
        /// Gets or sets the ending character index of the token.
        /// </summary>
        int End { get; set; }

        /// <summary>
        /// Gets the length of the token.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Gets the index of the token within its parent span.
        /// </summary>
        int Index { get; }

        /// <summary>
        /// Gets the text value of the token.
        /// </summary>
        string Value { get; }

        /// <summary>
        /// Gets the original text value of the token, before any replacements.
        /// </summary>
        string OriginalValue { get; }

        /// <summary>
        /// Gets the text value of the token as a read-only span of characters.
        /// </summary>
        ReadOnlySpan<char> ValueAsSpan { get; }

        /// <summary>
        /// Gets the original text value of the token as a read-only span of characters.
        /// </summary>
        ReadOnlySpan<char> OriginalValueAsSpan { get; }

        /// <summary>
        /// Gets the lemma of the token.
        /// </summary>
        string Lemma { get; }

        /// <summary>
        /// Gets the lemma of the token as a read-only span of characters.
        /// </summary>
        ReadOnlySpan<char> LemmaAsSpan { get; }

        /// <summary>
        /// Gets or sets the replacement text for the token.
        /// </summary>
        string Replacement { get; set; }

        /// <summary>
        /// Gets or sets the hash of the token's value.
        /// </summary>
        int Hash { get; set; }

        /// <summary>
        /// Gets or sets the case-insensitive hash of the token's value.
        /// </summary>
        int IgnoreCaseHash { get; set; }

        /// <summary>
        /// Gets the metadata associated with the token.
        /// </summary>
        Dictionary<string, string> Metadata { get; }

        /// <summary>
        /// Gets or sets the part-of-speech tag of the token.
        /// </summary>
        PartOfSpeech POS { get; set; }

        /// <summary>
        /// Gets the entity types associated with the token.
        /// </summary>
        IReadOnlyList<EntityType> EntityTypes { get; }

        /// <summary>
        /// Gets or sets the head of the token in the dependency tree.
        /// </summary>
        int Head { get; set; }

        /// <summary>
        /// Gets or sets the dependency relation type.
        /// </summary>
        string DependencyType { get; set; }

        /// <summary>
        /// Gets or sets the frequency of the token.
        /// </summary>
        float Frequency { get; set; }

        /// <summary>
        /// Gets the character immediately preceding the token.
        /// </summary>
        char? PreviousChar {get;}

        /// <summary>
        /// Gets the character immediately following the token.
        /// </summary>
        char? NextChar { get; }

        /// <summary>
        /// Adds an entity type to the token.
        /// </summary>
        /// <param name="entityType">The entity type to add.</param>
        void AddEntityType(EntityType entityType);

        /// <summary>
        /// Updates an entity type at the specified index.
        /// </summary>
        /// <param name="ix">The index of the entity type to update.</param>
        /// <param name="entityType">The new entity type.</param>
        void UpdateEntityType(int ix, ref EntityType entityType);

        /// <summary>
        /// Removes the entity type with the specified name.
        /// </summary>
        /// <param name="entityType">The name of the entity type to remove.</param>
        void RemoveEntityType(string entityType);

        /// <summary>
        /// Removes the entity type at the specified index.
        /// </summary>
        /// <param name="ix">The index of the entity type to remove.</param>
        void RemoveEntityType(int ix);

        /// <summary>
        /// Clears all entity types from the token.
        /// </summary>
        void ClearEntities();
    }
}