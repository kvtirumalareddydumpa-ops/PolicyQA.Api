/// <summary>
/// Splits text into word-based chunks with overlap.
/// ~200 words is roughly 250-300 tokens: one idea per chunk, well under the embedding model's input limit.
/// Overlap keeps a sentence or rule from being cut in half at a chunk boundary.
/// </summary>
public static class TextChunker
{
    public static IEnumerable<string> Chunk(string text, int chunkSize = 200, int overlap = 40)
    {
        // Validated eagerly (outside the iterator) so bad arguments fail immediately
        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize), "Chunk size must be positive.");
        if (overlap < 0 || overlap >= chunkSize)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap must be between 0 and chunkSize - 1.");

        return ChunkIterator(text, chunkSize, overlap);
    }

    private static IEnumerable<string> ChunkIterator(string text, int chunkSize, int overlap)
    {
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        for (int start = 0; start < words.Length; start += chunkSize - overlap)
        {
            yield return string.Join(' ', words.Skip(start).Take(chunkSize));
            if (start + chunkSize >= words.Length) break;
        }
    }
}
