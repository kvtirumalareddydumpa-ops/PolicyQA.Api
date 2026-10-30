using System.Text.Json;

public interface IEmbeddingClient
{
    Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
    Task<float[]> EmbedQueryAsync(string text, CancellationToken ct = default);
}

/// <summary>
/// Gemini embeddings (gemini-embedding-001).
/// Documents and questions use different task types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY),
/// which improves matching between short questions and long passages.
/// </summary>
public class GeminiEmbeddingClient(HttpClient http, IConfiguration config) : IEmbeddingClient
{
    private const int MaxBatchSize = 100; // API limit per batch call

    private readonly string _model = config["Gemini:EmbeddingModel"]!;
    private readonly int _dimensions = config.GetValue("Gemini:EmbeddingDimensions", 768);

    public async Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var results = new List<float[]>(texts.Count);

        // LINQ Chunk() = batches of up to 100 texts per call: fewer round trips, fewer rate-limit hits
        foreach (var batch in texts.Chunk(MaxBatchSize))
        {
            var body = new
            {
                requests = batch.Select(t => new
                {
                    model = $"models/{_model}",
                    content = new { parts = new[] { new { text = t } } },
                    taskType = "RETRIEVAL_DOCUMENT",
                    outputDimensionality = _dimensions
                })
            };

            using var doc = await GeminiHttp.PostJsonAsync(http, $"models/{_model}:batchEmbedContents", body, ct);
            results.AddRange(doc.RootElement.GetProperty("embeddings").EnumerateArray().Select(ReadValues));
        }

        return results;
    }

    public async Task<float[]> EmbedQueryAsync(string text, CancellationToken ct = default)
    {
        var body = new
        {
            content = new { parts = new[] { new { text } } },
            taskType = "RETRIEVAL_QUERY",
            outputDimensionality = _dimensions
        };

        using var doc = await GeminiHttp.PostJsonAsync(http, $"models/{_model}:embedContent", body, ct);
        return ReadValues(doc.RootElement.GetProperty("embedding"));
    }

    private static float[] ReadValues(JsonElement embedding) =>
        embedding.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
}
