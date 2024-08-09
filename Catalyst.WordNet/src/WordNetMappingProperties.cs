using System.Collections.Generic;
using Mosaik.Core;

namespace Catalyst
{
    public class WordNetMappingProperties : StorableObjectData
    {
        /// <summary>
        /// From localized word to WordNet entries (synset_offset+PoS). The entries
        /// follow the order of the mapping file.
        /// </summary>
        public Dictionary<string, List<(int Offset, PartOfSpeech PartOfSpeech)>> Mapping { get; set; }

        /// <summary>
        /// From WordNet entry (word+lexId) to localized word
        /// </summary>
        public Dictionary<(string Word, int LexId), HashSet<string>> InverseMapping { get; set; }
        public bool Loaded { get; set; }
    }
}
