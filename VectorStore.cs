public record DocumentChunk(
    Guid Id, string FileName, string ContentHash, int PageNumber, string Text, float[] Embedding);

public record SearchResult(DocumentChunk Chunk, double Score);

public interface IVectorStore
{
    bool HasDocument(string contentHash);

    /// <summary>Removes any chunks for this file name and adds the new ones in one atomic step. Returns chunks removed.</summary>
    int ReplaceDocument(string fileName, IReadOnlyCollection<DocumentChunk> chunks);

    IReadOnlyList<SearchResult> Search(float[] queryVector, int topK);

    int Count { get; }
}

/// <summary>
/// Simple in-memory store with brute-force cosine search. Fine for hundreds of chunks.
/// At scale: swap for Azure AI Search or SQL Server vectors (approximate nearest-neighbor indexes)
/// by implementing IVectorStore. Registered as a singleton, so all access is locked.
/// </summary>
public class InMemoryVectorStore : IVectorStore
{
    private readonly List<DocumentChunk> _chunks = new();
    private readonly object _lock = new();

    public int Count
    {
        get { lock (_lock) return _chunks.Count; }
    }

    public bool HasDocument(string contentHash)
    {
        lock (_lock) return _chunks.Any(c => c.ContentHash == contentHash);
    }

    public int ReplaceDocument(string fileName, IReadOnlyCollection<DocumentChunk> chunks)
    {
        lock (_lock)
        {
            int removed = _chunks.RemoveAll(c => c.FileName.Equals(fileName, StringComparison.OrdinalIgnoreCase));
            _chunks.AddRange(chunks);
            return removed;
        }
    }

    public IReadOnlyList<SearchResult> Search(float[] queryVector, int topK)
    {
        lock (_lock)
        {
            return _chunks
                .Select(c => new SearchResult(c, CosineSimilarity(queryVector, c.Embedding)))
                .OrderByDescending(r => r.Score)
                .Take(topK)
                .ToList();
        }
    }

    /// <summary>1.0 = same direction (same meaning), 0 = unrelated. Ignores vector length.</summary>
    public static double CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same dimensions (same embedding model and settings).");

        double dot = 0, magA = 0, magB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return dot / (Math.Sqrt(magA) * Math.Sqrt(magB) + 1e-10);
    }
}
