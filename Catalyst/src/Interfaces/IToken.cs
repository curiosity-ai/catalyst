using System;
using System.Collections.Generic;

namespace Catalyst
{
    public interface IToken
    {
        int Begin { get; set; }
        int End { get; set; }
        int Length { get; }
        int Index { get; }
        string Value { get; }
        ReadOnlySpan<char> ValueAsSpan { get; }
        string Replacement { get; set; }
        int Hash { get; set; }
        int IgnoreCaseHash { get; set; }
        Dictionary<string, string> Metadata { get; }
        PartOfSpeech POS { get; set; }
        EntityType[] EntityTypes { get; }
        int Head { get; set; }
        string DependencyType { get; set; }
        float Frequency { get; set; }

        void AddEntityType(EntityType entityType);

        void UpdateEntityType(int ix, ref EntityType entityType);

        void RemoveEntityType(string entityType);

        void RemoveEntityType(int ix);

        void ClearEntities();
    }
}