public static class TextChunker
{
    // ~200 words ≈ 250-300 tokens per chunk, with 40 words of overlap
    public static IEnumerable<string> Chunk(string text, int chunkSize = 200, int overlap = 40)
    {
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        for (int start = 0; start < words.Length; start += chunkSize - overlap)
        {
            yield return string.Join(' ', words.Skip(start).Take(chunkSize));
            if (start + chunkSize >= words.Length) break;
        }
    }
}