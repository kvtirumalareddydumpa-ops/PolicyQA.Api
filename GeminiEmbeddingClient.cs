using System.Text.Json;

public interface IEmbeddingClient
{
    Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
    Task<float[]> EmbedQueryAsync(string text, CancellationToken ct = default);
}

public class GeminiEmbeddingClient(HttpClient http, IConfiguration config) : IEmbeddingClient
{
    private readonly string _model = config["Gemini:EmbeddingModel"]!;
    private readonly int _dims = int.Parse(config["Gemini:EmbeddingDimensions"] ?? "768");

    public async Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var results = new List<float[]>();

        // LINQ's Chunk() splits our list into batches of 100 per API call
        foreach (var batch in texts.Chunk(100))
        {
            var body = new
            {
                requests = batch.Select(t => new
                {
                    model = $"models/{_model}",
                    content = new { parts = new[] { new { text = t } } },
                    taskType = "RETRIEVAL_DOCUMENT",
                    outputDimensionality = _dims
                })
            };

            using var doc = await PostAsync($"models/{_model}:batchEmbedContents", body, ct);
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
            outputDimensionality = _dims
        };

        using var doc = await PostAsync($"models/{_model}:embedContent", body, ct);
        return ReadValues(doc.RootElement.GetProperty("embedding"));
    }

    private async Task<JsonDocument> PostAsync(string path, object body, CancellationToken ct)
    {
        using var response = await http.PostAsJsonAsync(path, body, ct);
        var json = await response.Content.ReadAsStringAsync(ct);

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Gemini embedding error {(int)response.StatusCode}: {json}");

        return JsonDocument.Parse(json);
    }

    private static float[] ReadValues(JsonElement e) =>
        e.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
}