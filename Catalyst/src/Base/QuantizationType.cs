namespace Catalyst
{
    /// <summary>
    /// Specifies the quantization type used for storing matrices.
    /// </summary>
    public enum QuantizationType
    {
        /// <summary>No quantization (32-bit float).</summary>
        None,
        /// <summary>1-bit quantization.</summary>
        OneBit,
        /// <summary>2-bit quantization (not implemented).</summary>
        TwoBits,
        /// <summary>4-bit quantization (not implemented).</summary>
        FourBits
    }
}