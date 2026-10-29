using System.Text.Json;

public interface ILlmClient
{
    Task<LlmResult> GenerateAsync(string systemInstruction, string input, CancellationToken ct = default);
}

public record LlmResult(string Text, int InputTokens, int OutputTokens);

public class GeminiClient(HttpClient http, IConfiguration config) : ILlmClient
{
    private readonly string _model = config["Gemini:ChatModel"]!;

    public async Task<LlmResult> GenerateAsync(string systemInstruction, string input, CancellationToken ct = default)
    {
        var body = new
        {
            model = _model,
            system_instruction = systemInstruction,
            input,
            generation_config = new { thinking_level = "low" }
        };        

        using var response = await http.PostAsJsonAsync("interactions", body, ct);
        var json = await response.Content.ReadAsStringAsync(ct);

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Gemini error {(int)response.StatusCode}: {json}");

        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // The final answer lives in the "model_output" step(s)
        var text = string.Concat(
            root.GetProperty("steps").EnumerateArray()
                .Where(s => s.GetProperty("type").GetString() == "model_output")
                .SelectMany(s => s.GetProperty("content").EnumerateArray())
                .Where(c => c.GetProperty("type").GetString() == "text")
                .Select(c => c.GetProperty("text").GetString()));

        int inputtoken = 0, output = 0;
        if (root.TryGetProperty("usage", out var usage))
        {
            if (usage.TryGetProperty("total_input_tokens", out var i)) inputtoken = i.GetInt32();
            if (usage.TryGetProperty("total_output_tokens", out var o)) output = o.GetInt32();
        }

        return new LlmResult(text, inputtoken, output);
    }
}