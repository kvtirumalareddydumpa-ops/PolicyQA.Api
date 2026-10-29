public record DocumentChunk(Guid Id, string FileName, int PageNumber, string Text, float[] Embedding);

public record SearchResult(DocumentChunk Chunk, double Score);

public interface IVectorStore
{
    void AddRange(IEnumerable<DocumentChunk> chunks);
    IReadOnlyList<SearchResult> Search(float[] queryVector, int topK);
    int Count { get; }
}

public class InMemoryVectorStore : IVectorStore
{
    private readonly List<DocumentChunk> _chunks = new();
    private readonly object _lock = new();

    public void AddRange(IEnumerable<DocumentChunk> chunks)
    {
        lock (_lock) _chunks.AddRange(chunks);
    }

    public int Count
    {
        get { lock (_lock) return _chunks.Count; }
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

    // 1.0 = same meaning direction, 0 = unrelated
    private static double CosineSimilarity(float[] a, float[] b)
    {
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