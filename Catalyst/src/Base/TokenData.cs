using MessagePack;

namespace Catalyst
{
    /// <summary>
    /// Represents the raw data of a token, used for serialization and storage within a <see cref="Document"/>.
    /// </summary>
    [MessagePackObject]
    public struct TokenData
    {
        /// <summary>The lower bound (start index) of the token in the document text.</summary>
        [IgnoreMember] public int LowerBound;
        /// <summary>The upper bound (end index) of the token in the document text.</summary>
        [IgnoreMember] public int UpperBound;

        /// <summary>Gets or sets the bounds of the token as an array [LowerBound, UpperBound].</summary>
        [Key(0)] public int[] Bounds { get { return new int[2] { LowerBound, UpperBound }; } }

        /// <summary>Gets or sets the part-of-speech tag.</summary>
        [Key(1)] public PartOfSpeech Tag;

        /// <summary>Gets or sets the token hash.</summary>
        [Key(2)] public int Hash;

        /// <summary>Gets or sets the case-insensitive token hash.</summary>
        [Key(3)] public int IgnoreCaseHash;

        /// <summary>Gets or sets the head index for dependency parsing.</summary>
        [Key(4)] public int Head;

        /// <summary>Gets or sets the frequency of the token.</summary>
        [Key(5)] public float Frequency;

        /// <summary>Gets or sets the dependency relation type.</summary>
        [Key(6)] public string DependencyType;

        /// <summary>Gets or sets the replacement text for the token.</summary>
        [Key(7)] public string Replacement;

        /// <summary>
        /// Initializes a new instance of the <see cref="TokenData"/> struct for serialization.
        /// </summary>
        /// <param name="bounds">The token bounds.</param>
        /// <param name="tag">The part-of-speech tag.</param>
        /// <param name="hash">The hash.</param>
        /// <param name="ignoreCaseHash">The case-insensitive hash.</param>
        /// <param name="head">The head index.</param>
        /// <param name="frequency">The frequency.</param>
        /// <param name="dependencyType">The dependency type.</param>
        /// <param name="replacement">The replacement text.</param>
        [SerializationConstructor]
        public TokenData(int[] bounds, PartOfSpeech tag, int hash, int ignoreCaseHash, int head, float frequency, string dependencyType, string replacement)
        {
            LowerBound = bounds[0];
            UpperBound = bounds[1];
            Tag = tag;
            Replacement = replacement;
            Hash = hash;
            IgnoreCaseHash = ignoreCaseHash;
            Head = head;
            DependencyType = dependencyType;
            Frequency = frequency;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TokenData"/> struct with full details.
        /// </summary>
        /// <param name="lowerBound">The lower bound.</param>
        /// <param name="upperBound">The upper bound.</param>
        /// <param name="tag">The part-of-speech tag.</param>
        /// <param name="hash">The hash.</param>
        /// <param name="ignoreCaseHash">The case-insensitive hash.</param>
        /// <param name="head">The head index.</param>
        /// <param name="frequency">The frequency.</param>
        /// <param name="dependencyType">The dependency type.</param>
        /// <param name="replacement">The replacement text.</param>
        public TokenData(int lowerBound, int upperBound, PartOfSpeech tag, int hash, int ignoreCaseHash, int head, float frequency, string dependencyType, string replacement)
        {
            LowerBound = lowerBound;
            UpperBound = upperBound;
            Tag = tag;
            Replacement = replacement;
            Hash = hash;
            IgnoreCaseHash = ignoreCaseHash;
            Head = head;
            DependencyType = dependencyType;
            Frequency = frequency;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TokenData"/> struct with specified bounds.
        /// </summary>
        /// <param name="bounds">The token bounds.</param>
        public TokenData(int[] bounds)
        {
            LowerBound = bounds[0];
            UpperBound = bounds[1];
            Tag = PartOfSpeech.NONE;
            Replacement = null;
            Hash = 0;
            IgnoreCaseHash = 0;
            Head = -1;
            DependencyType = null;
            Frequency = 0f;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TokenData"/> struct with specified bounds.
        /// </summary>
        /// <param name="lowerBound">The lower bound.</param>
        /// <param name="upperBound">The upper bound.</param>
        public TokenData(int lowerBound, int upperBound)
        {
            LowerBound = lowerBound;
            UpperBound = upperBound;
            Tag = PartOfSpeech.NONE;
            Replacement = null;
            Hash = 0;
            IgnoreCaseHash = 0;
            Head = -1;
            DependencyType = null;
            Frequency = 0f;
        }
    }
}