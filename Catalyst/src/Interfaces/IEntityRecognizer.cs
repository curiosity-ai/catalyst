using Mosaik.Core;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a model that can recognize entities in a document.
    /// </summary>
    public interface IEntityRecognizer : IModel
    {
        /// <summary>
        /// Recognizes entities in the specified document.
        /// </summary>
        /// <param name="document">The document to process.</param>
        /// <returns>True if entities were successfully recognized, false otherwise.</returns>
        bool RecognizeEntities(IDocument document);

        /// <summary>
        /// Returns an array of entity types that this recognizer can produce.
        /// </summary>
        /// <returns>An array of entity type names.</returns>
        string[] Produces();
    }
}